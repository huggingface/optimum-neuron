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

import numpy as np
import torch
from optimum.utils import logging

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
    # Use torch's built-in nanmean and compute variance with Bessel's correction
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)
    count = torch.sum(~torch.isnan(tensor))
    variance *= count / (count - 1).clamp(min=1.0)  # Bessel's correction, avoid division by zero
    return torch.sqrt(variance)
