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
        max_length = max(t.shape[0] for t in tensors)

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


def entropy_from_logits(logits: torch.Tensor, chunk_size: int = 128) -> torch.Tensor:
    """
    Compute the Shannon entropy (in nats) for each row of *logits*.

    Original implementation from trl.trainer.utils.entropy_from_logits provide a memory efficient alternative,
    but it accumulates results in a list which can lead to graph fragmentation on XLA devices.
    Here we keep things simple and compute the entropy in one go.
    """
    logps = F.log_softmax(logits, dim=-1)
    entropy = -(torch.exp(logps) * logps).sum(-1)
    return entropy


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


def batch_pad_sequences(
    sequences: list[list[int | float]],
    target_length: int,
    padding_value: int | float = 0,
    padding_side: Literal["left", "right"] = "right",
    dtype: torch.dtype = torch.long,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    XLA-optimized batch padding of variable-length sequences.

    Unlike per-sequence padding with list comprehensions, this function:
    1. Pre-allocates output arrays using numpy (fast CPU operations)
    2. Transfers to device as a single operation (one host->device copy)
    3. Creates the mask alongside the padded sequences (no separate allocation)

    This avoids creating many small tensors and multiple device transfers that cause
    XLA graph fragmentation.

    Args:
        sequences (`list[list[int | float]]`):
            List of variable-length sequences. Each sequence is a list of token IDs (ints)
            or log probabilities (floats).
        target_length (`int`):
            Fixed target length for all sequences. Sequences longer than this will be
            truncated; shorter sequences will be padded.
        padding_value (`int | float`, *optional*, defaults to `0`):
            Value to use for padding positions.
        padding_side (`Literal["left", "right"]`, *optional*, defaults to `"right"`):
            Side on which to add padding. Also determines truncation behavior:
            - `"left"`: Pads on left, truncates from left (keeps last tokens)
            - `"right"`: Pads on right, truncates from right (keeps first tokens)
        dtype (`torch.dtype`, *optional*, defaults to `torch.long`):
            Output tensor dtype for the padded sequences.
        device (`torch.device | None`, *optional*, defaults to `None`):
            Target device for the output tensors. If `None`, tensors remain on CPU.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`:
            A tuple of `(padded_sequences, mask)` where:
            - `padded_sequences` has shape `(batch_size, target_length)` and dtype `dtype`
            - `mask` has shape `(batch_size, target_length)` and dtype `torch.long`, with
              `1` for real tokens and `0` for padding positions
    """
    batch_size = len(sequences)

    # Determine numpy dtype for intermediate computation
    if dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
        np_dtype = np.float32
    else:
        np_dtype = np.int64

    # Pre-allocate numpy arrays (fast CPU operations)
    padded = np.full((batch_size, target_length), padding_value, dtype=np_dtype)
    mask = np.zeros((batch_size, target_length), dtype=np.int64)

    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        if seq_len == 0:
            continue

        if seq_len >= target_length:
            # Truncation needed
            if padding_side == "left":
                # Keep last target_length tokens
                padded[i] = seq[seq_len - target_length :]
            else:
                # Keep first target_length tokens
                padded[i] = seq[:target_length]
            mask[i] = 1
        else:
            # Padding needed
            if padding_side == "left":
                start_idx = target_length - seq_len
                padded[i, start_idx:] = seq
                mask[i, start_idx:] = 1
            else:
                padded[i, :seq_len] = seq
                mask[i, :seq_len] = 1

    # Single conversion and transfer to device
    padded_tensor = torch.from_numpy(padded).to(dtype=dtype)
    mask_tensor = torch.from_numpy(mask).to(dtype=torch.long)

    if device is not None:
        padded_tensor = padded_tensor.to(device)
        mask_tensor = mask_tensor.to(device)

    return padded_tensor, mask_tensor


def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """
    XLA-compatible version of nanmin that doesn't use dynamic indexing.
    Compute the minimum value of a tensor, ignoring NaNs.
    """
    mask = torch.isnan(tensor)
    filled = torch.where(mask, torch.tensor(float("inf"), device=tensor.device), tensor)
    min_value = torch.min(filled)
    return min_value


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    XLA-compatible version of nanmax that doesn't use dynamic indexing.
    Compute the maximum value of a tensor, ignoring NaNs.
    """
    mask = torch.isnan(tensor)
    filled = torch.where(mask, torch.tensor(float("-inf"), device=tensor.device), tensor)
    max_value = torch.max(filled)
    return max_value


def nanstd(tensor: torch.Tensor, unbiased: bool = False) -> torch.Tensor:
    """
    XLA-compatible version of nanstd.
    Compute the standard deviation of a tensor, ignoring NaNs.
    """
    mask = ~torch.isnan(tensor)
    count = mask.sum()

    clean = torch.where(mask, tensor, torch.zeros_like(tensor))
    mean = clean.sum() / count

    diff_squared = torch.where(mask, (clean - mean) ** 2, torch.zeros_like(tensor))

    if unbiased:
        variance = diff_squared.sum() / (count - 1).clamp(min=1)
    else:
        variance = diff_squared.sum() / count

    return variance.sqrt()


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
