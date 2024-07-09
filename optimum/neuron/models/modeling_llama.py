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
"""Modules for the Llama architecture."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import LlamaConfig
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaModel,
    apply_rotary_pos_emb,
    repeat_kv,
)

from ..utils.require_utils import requires_neuronx_distributed
from .core import CoreAttention, NeuronAttention, PatchedModule


class NeuronLlamaAttention(LlamaAttention, NeuronAttention):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx=layer_idx)
        self.core_attn = CoreAttention()

    @classmethod
    def from_original(cls, orig_module: torch.nn.Module, **options) -> "NeuronLlamaAttention":
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
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        from neuronx_distributed.kernels.flash_attn import nki_flash_attn_func

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
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

        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

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

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class NeuronLlamaModel(LlamaModel, PatchedModule):
    @classmethod
    def from_original(cls, orig_module: torch.nn.Module, **options) -> "NeuronLlamaModel":
        orig_module.__class__ = cls
        return orig_module

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # TODO: work on the validity of that.
        if self.training:
            return None
        return super()._update_causal_mask(
            attention_mask, input_tensor, cache_position, past_key_values, output_attentions
        )
