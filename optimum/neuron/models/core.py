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
"""Core functionalities and tools for rewriting modules for Neuron."""

import math
from typing import Optional

import torch
import torch.nn as nn


class NeuronAttention:
    # TODO: add dosctring
    @property
    def sequence_parallel_enabled(self) -> bool:
        return getattr(self, "_sequence_parallel_enabled", False)

    @sequence_parallel_enabled.setter
    def sequence_parallel_enabled(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("sequence_parallel_enabled must be a boolean value.")
        self._sequence_parallel_enabled = value

    @property
    def flash_attention_enabled(self) -> bool:
        return getattr(self, "_flash_attention_enabled", False)

    @flash_attention_enabled.setter
    def flash_attention_enabled(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("flash_attention_enabled must be a boolean value.")
        self._flash_attention_enabled = value


class CoreAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.tensor],
        attention_dropout: float = 0.0,
    ) -> torch.Tensor:
        bsz, num_heads, q_len, head_dim = query_states.shape
        kv_seq_len = key_states.shape[-2]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

        if attn_weights.size() != (bsz, num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        causal_mask = torch.triu(torch.ones((1, 1, q_len, kv_seq_len), device="xla"), diagonal=1).bool()
        # TODO: change -10000.0 with a better value (dtype.min)
        attn_weights = attn_weights.masked_fill_(causal_mask, -10000.0)

        # TODO: enable that.
        # if attention_mask is not None:  # no matter the length, we just slice it
        #     causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        #     attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.double).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        return attn_output
