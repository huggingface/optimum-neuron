# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Qwen3 model, partly based on Llama model and on Transformers implementation."""

from functools import partial
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.processing_utils import Unpack
from transformers.utils import logging

from ....utils import is_neuronx_distributed_available
from ..config import TrainingNeuronConfig
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
)
from ..modeling_utils import ALL_ATTENTION_FUNCTIONS


if is_neuronx_distributed_available():
    from neuronx_distributed.parallel_layers.layers import (
        ParallelEmbedding,
    )
    from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_size

logger = logging.get_logger(__name__)


def _init_normal(std, w):
    return nn.init.normal_(w, mean=0.0, std=std)


class Qwen3Attention(LlamaAttention):
    def __init__(self, config: Qwen3Config, trn_config: TrainingNeuronConfig, layer_idx: int):
        super().__init__(config, trn_config, layer_idx)
        self.q_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.trn_config.sequence_parallel_enabled:
            q_len, bsz, _ = hidden_states.size()
            q_len = q_len * get_tensor_model_parallel_size()
        else:
            bsz, q_len, _ = hidden_states.size()

        if self.qkv_linear:
            query_states, key_states, value_states = self.qkv_proj(hidden_states)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        if self.trn_config.sequence_parallel_enabled:
            query_states = query_states.view(q_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
            key_states = key_states.view(q_len, bsz, self.num_key_value_heads, self.head_dim).permute(1, 2, 0, 3)
            value_states = value_states.view(q_len, bsz, self.num_key_value_heads, self.head_dim).permute(1, 2, 0, 3)
        else:
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Main difference from LlamaAttention is that Qwen3 applies a norm on query and key after the projection
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.config._attn_implementation == "flash_attention_2":
            if self.training and self.attention_dropout > 0.0:
                raise RuntimeError(
                    "Attention dropout produces NaN with flash_attention_2. Please set it to 0.0 until this bug is "
                    "resolved by the Neuron SDK."
                )
            attention_interface = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
            attn_output = attention_interface(
                query_states,
                repeat_kv(key_states, self.num_key_value_groups),
                repeat_kv(value_states, self.num_key_value_groups),
                dropout_p=0.0 if not self.training else self.attention_dropout,
                softmax_scale=self.scaling,
                causal=True,
                mixed_precision=True,
            )
            attn_weights = None
        else:
            attn_output, attn_weights = eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                causal=attention_mask is None,
                **kwargs,
            )

        if self.trn_config.sequence_parallel_enabled:
            attn_output = attn_output.permute(2, 0, 1, 3)
            attn_output = attn_output.reshape(q_len, bsz, self.num_heads * self.head_dim)
        else:
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class Qwen3DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: Qwen3Config, trn_config: TrainingNeuronConfig, layer_idx: int):
        super().__init__(config, trn_config, layer_idx)
        self.self_attn = Qwen3Attention(config=config, trn_config=trn_config, layer_idx=layer_idx)


class Qwen3Model(LlamaModel):
    def __init__(self, config: Qwen3Config, trn_config: TrainingNeuronConfig):
        LlamaPreTrainedModel.__init__(self, config)
        # In this Neuron implementation of Qwen3, we do not support sliding window.
        if config.get_text_config().sliding_window is not None:
            raise ValueError(
                "Sliding window attention is not supported for Qwen3 on Neuron. Please disable it in the model config."
            )

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.trn_config = trn_config

        init_method = partial(_init_normal, config.initializer_range)
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            init_method=init_method,
            sequence_parallel_enabled=trn_config.sequence_parallel_enabled,
            dtype=config.torch_dtype,
        )
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, trn_config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            diagonal_attend_mask = torch.arange(target_length, device=cache_position.device) > cache_position.reshape(
                -1, 1
            )
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


class Qwen3ForCausalLM(LlamaForCausalLM):
    config_class = Qwen3Config

    def __init__(self, config, trn_config: TrainingNeuronConfig):
        super().__init__(config, trn_config)
        self.model = Qwen3Model(config, trn_config)
