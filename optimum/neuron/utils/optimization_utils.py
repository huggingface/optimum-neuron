# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Optimization utilities."""

import torch


def get_attention_scores_sd15(self, query, key, attention_mask) -> torch.Tensor:
    """Optimized attention for Stable Diffusion 1.5 UNET."""
    dtype = query.dtype

    if self.upcast_attention:
        query = query.float()
        key = key.float()

    baddbmm_input = torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device)
    beta = 0

    attention_scores = torch.baddbmm(
        baddbmm_input,
        query,
        key.transpose(-1, -2),
        beta=beta,
        alpha=self.scale,
    )
    del baddbmm_input

    # TODO: following line is supposed to give the same result and reduce unnecessary overhead(no attention mask)
    # however the compiled model output is far off from the one on cpu, need to further investigate.
    # attention_scores = self.scale * torch.bmm(query, key.transpose(-1, -2))  # -> bad perf, max diff: 5.696073055267334 (atol: 0.001)

    if self.upcast_softmax:
        attention_scores = attention_scores.float()

    attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
    del attention_scores

    attention_probs = attention_probs.to(dtype)

    return attention_probs


def get_attention_scores_sd2(self, query, key, attn_mask):
    """Optimized attention for Stable Diffusion 2 UNET."""
    dtype = query.dtype

    if self.upcast_attention:
        query = query.float()
        key = key.float()

    def _custom_badbmm(a, b):
        bmm = torch.bmm(a, b)
        scaled = bmm * 0.125
        return scaled

    # Check for square matmuls
    if query.size() == key.size():
        attention_scores = _custom_badbmm(key, query.transpose(-1, -2))

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=1).permute(0, 2, 1)
        attention_probs = attention_probs.to(dtype)

    else:
        attention_scores = _custom_badbmm(query, key.transpose(-1, -2))

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(dtype)

    return attention_probs


def get_attention_scores_sdxl(self, query, key, attn_mask):
    """Optimized attention for SDXL UNET."""

    def _custom_badbmm(a, b, scale):
        bmm = torch.bmm(a, b)
        scaled = bmm * scale
        return scaled

    if query.size() == key.size():
        attention_scores = _custom_badbmm(key, query.transpose(-1, -2), self.scale)
        attention_probs = attention_scores.softmax(dim=1).permute(0, 2, 1)

    else:
        attention_scores = _custom_badbmm(query, key.transpose(-1, -2), self.scale)
        attention_probs = attention_scores.softmax(dim=-1)

    return attention_probs
