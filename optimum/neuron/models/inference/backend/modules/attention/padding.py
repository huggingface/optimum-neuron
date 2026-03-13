# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import torch
from torch.nn import functional as F


def _maybe_cast_fp8_for_padding(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.dtype | None]:
    recast_dtype = None
    if tensor.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        recast_dtype = tensor.dtype
        tensor = tensor.to(torch.bfloat16)
    return tensor, recast_dtype


def _restore_fp8_after_padding(tensor: torch.Tensor, recast_dtype: torch.dtype | None) -> torch.Tensor:
    if recast_dtype is not None:
        tensor = tensor.to(recast_dtype)
    return tensor


def maybe_pad_interleaved(tensor, pad_dim: int, source_heads: int, target_heads: int, source_group_size: int):
    if tensor is None:
        return tensor

    tensor, recast_dtype = _maybe_cast_fp8_for_padding(tensor)

    shape = (
        tensor.shape[:pad_dim] + (source_heads, tensor.shape[pad_dim] // source_heads) + tensor.shape[pad_dim + 1 :]
    )
    tensor = tensor.view(shape)

    splits = torch.split(tensor, source_group_size, dim=pad_dim)

    pad_size = list(splits[0].size())
    pad_size[pad_dim] = (target_heads - source_heads) // (source_heads // source_group_size)
    pads = [torch.zeros(pad_size, dtype=tensor.dtype, device=tensor.device)] * len(splits)

    interleaved = [t for pair in zip(splits, pads) for t in pair]
    tensor = torch.cat(interleaved, dim=pad_dim)

    shape = tensor.shape[:pad_dim] + (tensor.shape[pad_dim] * tensor.shape[pad_dim + 1],) + tensor.shape[pad_dim + 2 :]
    tensor = _restore_fp8_after_padding(tensor, recast_dtype)
    return tensor.view(shape)


def maybe_pad_tail(tensor, source_heads: int, target_heads: int, pad_dim: int):
    if tensor is None:
        return tensor
    size_to_pad = int((tensor.shape[pad_dim] // source_heads) * target_heads - tensor.shape[pad_dim])

    dims_after_pad_dim = len(tensor.size()) - pad_dim
    pad_length = dims_after_pad_dim * 2
    pad = (0,) * (pad_length - 1) + (size_to_pad,)

    return F.pad(tensor, pad)


def replicate_kv(tensor, source_heads: int, repeats: int, head_dim=0):
    if tensor is None:
        return tensor
    shape = (
        tensor.shape[:head_dim] + (source_heads, tensor.shape[head_dim] // source_heads) + tensor.shape[head_dim + 1 :]
    )
    tensor = tensor.view(shape)
    tensor = torch.repeat_interleave(tensor, repeats=repeats, dim=head_dim)
    shape = (
        tensor.shape[:head_dim] + (tensor.shape[head_dim] * tensor.shape[head_dim + 1],) + tensor.shape[head_dim + 2 :]
    )
    return tensor.view(shape)
