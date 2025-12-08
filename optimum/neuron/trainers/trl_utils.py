# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Literal

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from optimum.utils import logging
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from trl.trainer.utils import RepeatSampler

from ..utils import is_precompilation


logger = logging.get_logger()

TRL_VERSION = "0.24.0"


def pad(
    tensors: list[torch.Tensor],
    padding_value: int = 0,
    padding_side: str = "right",
    max_length: int | None = None,
) -> torch.Tensor:
    """
    Pads a list of tensors to the same shape along the first dimension.
    It differs from `trl` by enfoncing the same sequence length for all tensors, which is required to avoid
    recompilation.
    """
    batch_size = len(tensors)
    if max_length is None:
        max_length = np.max([t.shape[0] for t in tensors]).tolist()

    output_shape = (max_length,) + tensors[0].shape[1:]

    # Create an output tensor filled with the padding value
    output = torch.full((batch_size, *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)

    for i, t in enumerate(tensors):
        if padding_side == "left":
            seq_start = output_shape[0] - t.shape[0]
        elif padding_side == "right":
            seq_start = 0
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        # Define the slices
        seq_slice = slice(seq_start, seq_start + t.shape[0])
        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output


def pad_or_truncate_to_length(
    tensor: torch.Tensor,
    length: int,
    dim: int = 0,
    padding_value: int = 0,
    padding_or_truncate_side: Literal["left", "right"] = "right",
) -> torch.Tensor:
    """
    Pads or truncates a tensor to a given length along the provided dimension.

    Args:
        tensor: Input tensor to pad or truncate
        length: Target length
        dim: Dimension along which to pad/truncate
        padding_value: Value to use for padding
        padding_or_truncate_side: Side for both padding and truncation
            - "left": Pads on left, truncates from left (keeps last tokens)
            - "right": Pads on right, truncates from right (keeps first tokens)
    """
    current_length = tensor.shape[dim]
    if current_length == length:
        return tensor
    elif current_length > length:
        # Truncate
        slice_ = [slice(None)] * tensor.dim()
        if padding_or_truncate_side == "left":
            # Keep last tokens (truncate from left)
            slice_[dim] = slice(current_length - length, current_length)
        elif padding_or_truncate_side == "right":
            # Keep first tokens (truncate from right)
            slice_[dim] = slice(0, length)
        else:
            raise ValueError("padding_or_truncate_side must be 'left' or 'right'")
        return tensor[slice_]
    else:
        # Pad
        padding_shape = list(tensor.shape)
        padding_shape[dim] = length - current_length
        padding = torch.full(padding_shape, padding_value, dtype=tensor.dtype, device=tensor.device)
        if padding_or_truncate_side == "left":
            return torch.cat([padding, tensor], dim=dim)
        elif padding_or_truncate_side == "right":
            return torch.cat([tensor, padding], dim=dim)
        else:
            raise ValueError("padding_or_truncate_side must be 'left' or 'right'")


def entropy_from_logits(logits: torch.Tensor, chunk_size: int = 128) -> torch.Tensor:
    """
    Compute the Shannon entropy (in nats) for each row of *logits* in a memory-efficient way.

    Instead of materializing the full softmax for all rows at once, the logits are flattened to shape (N, num_classes),
    where N is the product of all leading dimensions. Computation is then performed in chunks of size `chunk_size`
    along this flattened dimension, reducing peak memory usage. The result is reshaped back to match the input's
    leading dimensions.

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`. Entropy is taken along the last axis; all leading dimensions
            are preserved in the output.
        chunk_size (`int`, *optional*, defaults to `128`):
            Number of rows from the flattened logits to process per iteration. Smaller values reduce memory usage at
            the cost of more iterations.

    Returns:
        `torch.Tensor`:
            Entropy values with shape `logits.shape[:-1]`.
    """
    original_shape = logits.shape[:-1]  # all dims except num_classes
    num_classes = logits.shape[-1]

    # Flatten all leading dimensions into one
    flat_logits = logits.reshape(-1, num_classes)

    entropies = []
    for chunk in flat_logits.split(chunk_size, dim=0):
        logps = F.log_softmax(chunk, dim=-1)
        chunk_entropy = -(torch.exp(logps) * logps).sum(-1)
        entropies.append(chunk_entropy)

    entropies = torch.cat(entropies, dim=0)
    return entropies.reshape(original_shape)


def neuron_parallel_compile_tokenizer_decoder_method(
    self,
    token_ids: int | list[int],
    skip_special_tokens: bool = False,
    clean_up_tokenization_spaces: bool | None = None,
    **kwargs,
) -> str:
    """
    Patched `tokenizer._decode` method for `neuron_parallel_compile`.
    This is needed because any tensor operation on the XLA device during `neuron_parallel_compile` produces rubbish
    results, which is not an issue in general, but causes failure when the token IDS end up being out of range for the
    tokenizer vocabulary.
    """
    if not is_precompilation():
        raise RuntimeError("This patch method should only be used with `neuron_parallel_compile`.")

    # We log the token IDs to force the data mouvement to CPU, which would happen during actual decoding.
    logger.debug("Using patched tokenizer.decode method for Neuron parallel compilation, token_ids = ", token_ids)

    # Returns a dummy string, we do not care about the value in this context.
    return "dummy"


def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """
    XLA-compatible version of nanmin that doesn't use dynamic indexing.
    Compute the minimum value of a tensor, ignoring NaNs.
    """
    mask = torch.isnan(tensor)
    if mask.all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    filled = torch.where(mask, torch.tensor(float("inf"), dtype=tensor.dtype, device=tensor.device), tensor)
    return torch.min(filled)


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    XLA-compatible version of nanmax that doesn't use dynamic indexing.
    Compute the maximum value of a tensor, ignoring NaNs.
    """
    mask = torch.isnan(tensor)
    if mask.all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    filled = torch.where(mask, torch.tensor(float("-inf"), dtype=tensor.dtype, device=tensor.device), tensor)
    return torch.max(filled)


def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    XLA-compatible version of nanstd.
    Compute the standard deviation of a tensor, ignoring NaNs.
    """
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)
    return torch.sqrt(variance)


class DistributedRepeatSampler(RepeatSampler, DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        # Initialize RepeatSampler with the actual sampling logic
        RepeatSampler.__init__(
            self,
            data_source=dataset,
            mini_repeat_count=mini_repeat_count,
            batch_size=batch_size,
            repeat_count=repeat_count,
            shuffle=shuffle,
            seed=seed,
        )

        # Store DistributedSampler attributes for interface compatibility
        # (but we don't use them for actual distribution)
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() else 0

        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
