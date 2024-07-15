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
"""Modules for the GPT Neo X architecture."""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import GPTNeoXConfig
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_mistral import (
    GPTNeoXAttention,
    GPTNeoXModel,
    apply_rotary_pos_emb,
)

from ....utils import logging
from ..utils.require_utils import requires_neuronx_distributed
from .core import CoreAttention, NeuronAttention


logger = logging.get_logger(__name__)


class GPTNeoXCoreAttention(CoreAttention):
    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_dropout: float = 0.0,
        attention_mask: Optional[torch.tensor] = None,
        head_mask: Optional[torch.tensor] = None,
    ) -> torch.Tensor:
        bsz, num_heads, q_len, head_dim = query_states.shape
        kv_seq_len = key_states.shape[-2]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

        if attn_weights.size() != (bsz, num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:  # no matter the length, we just slice it
            # This is the Transformers way of applying the mask.
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        else:
            # This is the recommended way for Neuron. This way the attention is not passed as an argument
            # avoiding communication that is not needed.
            causal_mask = torch.triu(torch.ones((1, 1, q_len, kv_seq_len), device="xla"), diagonal=1).bool()
            mask_value = torch.finfo(attn_weights.dtype).min
            attn_weights = attn_weights.masked_fill_(causal_mask, mask_value)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.double).to(query_states.dtype)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_weights = nn.functional.dropout(attn_weights, p=attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        return attn_output


class NeuronGPTNeoXAttention(GPTNeoXAttention, NeuronAttention):
    def __init__(self, config: GPTNeoXConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx=layer_idx)
        self.core_attn = GPTNeoXCoreAttention()

    @classmethod
    def from_original(cls, orig_module: torch.nn.Module, **options) -> "NeuronGPTNeoXAttention":
        orig_module.core_attn = GPTNeoXCoreAttention()
        orig_module.__class__ = cls
        return orig_module

    def _attn_projections_and_rope(
        self,
        hidden_states: torch.FloatTensor,
        position_ids: torch.LongTensor,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
    ):
        has_layer_past = layer_past is not None

        # Compute QKV
        # Attention heads if sequence_parallel_enabled:
        #   --> [seq_len, batch, (num_heads * 3 * head_size)]
        # Else:
        #   --> [batch, seq_len, (num_heads * 3 * head_size)]
        qkv = self.query_key_value(hidden_states)

        # If sequence_parallel_enabled:
        #   --> [seq_len, batch, num_heads, 3 * head_size]
        # Else:
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)

        if self.sequence_parallel_enabled:
            # [seq_len, batch, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
            query = qkv[..., : self.head_size].permute(1, 2, 0, 3)
            key = qkv[..., self.head_size : 2 * self.head_size].permute(1, 2, 0, 3)
            value = qkv[..., 2 * self.head_size :].permute(1, 2, 0, 3)
        else:
            # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
            query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
            key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
            value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims :]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims :]

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = key.shape[-2]
        if has_layer_past:
            seq_len += layer_past[0].shape[-2]
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        # Cache QKV values
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None

        return query, key, value, present

    @requires_neuronx_distributed
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        from neuronx_distributed.kernels.flash_attn import nki_flash_attn_func

        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # compute causal mask from causal mask buffer
        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        key_length = key.size(-2)

        # dynamically increase the causal mask with the key length, if needed.
        if key_length > self.bias.shape[-1]:
            self._init_bias(key_length, device=key.device)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

        attention_mask = None
        if 0.0 in causal_mask:
            attention_mask = causal_mask

        if self.flash_attention_enabled:
            if head_mask is not None:
                raise ValueError("It is not possible to specify a head mask with flash attention.")
            if attention_mask is not None:
                raise ValueError(
                    "Only a causal mask can be used with flash attention, but you provided an attention mask here."
                )

            attn_output = nki_flash_attn_func(query, key, value, droupout_p=self.attention_dropout.p)
        else:
            attn_output = self.core_attn(
                query,
                key,
                value,
                attention_dropout=self.attention_dropout,
                attention_mask=attention_mask,
                head_mask=head_mask,
            )

        return attn_output, None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        # Apply attention-specific projections and rope
        query, key, value, present = self._attn_projections_and_rope(
            hidden_states=hidden_states, position_ids=position_ids, layer_past=layer_past, use_cache=use_cache
        )

        # Compute attention
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # Reshape outputs
        if self.sequence_parallel_enabled:
            # [batch, num_attention_heads, seq_len, head_size] -> [seq_len, batch, hidden_size]
            attn_output = attn_output.permute(2, 0, 1, 3).contiguous()
            attn_output = attn_output.view(*attn_output.shape[:2], -1)
        else:
            attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)

        attn_output = self.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class NeuronGPTNeoXModel(GPTNeoXModel):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * self.config.num_hidden_layers)
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, seq_length + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_in(input_ids)

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            if self._attn_implementation == "flash_attention_2":
                attention_mask = attention_mask if 0 in attention_mask else None
            elif self._attn_implementation == "sdpa" and not output_attentions and head_mask is None:
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask=attention_mask,
                    input_shape=(batch_size, seq_length),
                    inputs_embeds=inputs_embeds,
                    past_key_values_length=past_length,
                )
            else:
                if 0 in attention_mask:
                    attention_mask = None
                else:
                    attention_mask = _prepare_4d_causal_attention_mask(
                        attention_mask=attention_mask,
                        input_shape=(batch_size, seq_length),
                        inputs_embeds=inputs_embeds,
                        past_key_values_length=past_length,
                    )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        hidden_states = self.emb_dropout(inputs_embeds)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (layer, layer_past) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    head_mask[i],
                    use_cache,
                    None,
                    output_attentions,
                )
            else:
                outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    head_mask=head_mask[i],
                    layer_past=layer_past,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_attentions = all_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.final_layer_norm(hidden_states)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
