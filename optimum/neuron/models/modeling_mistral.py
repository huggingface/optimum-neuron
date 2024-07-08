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
"""Modules for the Mistral architecture."""

import warnings
from typing import Optional, Tuple

import torch
from transformers import MistralConfig
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_mistral import (
    MistralAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)

from ..utils.require_utils import requires_neuronx_distributed
from .core import CoreAttention, NeuronAttention


class NeuronMistralAttention(MistralAttention, NeuronAttention):
    def __init__(self, config: MistralConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx=layer_idx)
        self.core_attn = CoreAttention()

    @classmethod
    def from_original(cls, orig_module: torch.nn.Module, **options) -> "NeuronMistralAttention":
        orig_module.core_attn = CoreAttention()
        orig_module.__class__ = cls
        return orig_module

    @requires_neuronx_distributed
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        from neuronx_distributed.kernels.flash_attn import nki_flash_attn_func

        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and removed since `transformers` v4.37. Please make sure to "
                "use `attention_mask` instead.`"
            )
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.sequence_parallel_enabled:
            q_len, bsz, _ = query_states.size()
        else:
            bsz, q_len, _ = query_states.size()

        if self.sequence_parallel_enabled:
            # [S, B, hidden_dim] -> [S, B, num_heads, head_dim] -> [B, num_heads, S, head_dim]
            query_states = query_states.view(q_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
            key_states = key_states.view(q_len, bsz, self.num_key_value_heads, self.head_dim).permute(1, 2, 0, 3)
            value_states = value_states.view(q_len, bsz, self.num_key_value_heads, self.head_dim).permute(1, 2, 0, 3)
        else:
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    "The cache structure has changed since `transformers` v4.36. If you are using "
                    f"{self.__class__.__name__} for auto-regressive decoding with k/v caching, please make sure to "
                    "initialize the attention class with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # if attention_mask is not None:
        #     if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
        #         raise ValueError(
        #             f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
        #         )

        #     attn_weights = attn_weights + attention_mask

        attn_output = (
            nki_flash_attn_func(query_states, key_states, value_states, droupout_p=self.attention_dropout)
            if self.flash_attention_enabled
            else self.core_attn(query_states, key_states, value_states, attention_dropout=self.attention_dropout)
        )

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        if self.sequence_parallel_enabled:
            # [B, num_heads, S, head_dim] -> [S, B, num_heads, head_dim]
            attn_output = attn_output.permute(2, 0, 1, 3)
            attn_output = attn_output.reshape(q_len, bsz, -1)
        else:
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
