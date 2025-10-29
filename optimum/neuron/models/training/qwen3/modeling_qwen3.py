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

import torch
from neuronx_distributed.kernels.flash_attn import nki_flash_attn_func
from neuronx_distributed.parallel_layers.layers import ParallelEmbedding
from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_size
from torch import nn
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, logging

from ..config import TrainingNeuronConfig
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
)


logger = logging.get_logger(__name__)


def _init_normal(std, w):
    return nn.init.normal_(w, mean=0.0, std=std)


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, reduction_dim=-1, eps=1e-6, sequence_parallel_enabled=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size))
        setattr(self.weight, "sequence_parallel_enabled", sequence_parallel_enabled)
        self.variance_epsilon = eps
        self.reduction_dim = reduction_dim

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(self.reduction_dim, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.reduction_dim not in [-1, hidden_states.dim() - 1]:
            # If reduction_dim is not the last dimension, we cannot broadcast the weight directly.
            weight_shape = [1] * hidden_states.dim()
            weight_shape[self.reduction_dim] = self.hidden_size
            output = self.weight.view(weight_shape) * hidden_states.to(input_dtype)
        else:
            # In this case, we can broadcast the weight directly.
            output = self.weight * hidden_states.to(input_dtype)
        return output

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3Attention(LlamaAttention):
    def __init__(self, config: Qwen3Config, trn_config: TrainingNeuronConfig, layer_idx: int):
        super().__init__(config, trn_config, layer_idx)
        if self.use_flash_attention_v2 and trn_config.transpose_nki_inputs:
            reduction_dim = -2
        else:
            reduction_dim = -1
        self.q_norm = Qwen3RMSNorm(
            self.head_dim, reduction_dim=reduction_dim, eps=config.rms_norm_eps
        )  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(
            self.head_dim, reduction_dim=reduction_dim, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.trn_config.sequence_parallel_enabled:
            q_len, bsz, _ = hidden_states.size()
            q_len = q_len * get_tensor_model_parallel_size()
        else:
            bsz, q_len, _ = hidden_states.size()

        if self.trn_config.fuse_qkv and self.num_heads == self.num_key_value_heads and self.kv_size_multiplier == 1:
            qkv_states = self.qkv_proj(hidden_states)
            query_states, key_states, value_states = qkv_states.split(self.split_size, dim=2)
        elif self.qkv_linear:
            query_states, key_states, value_states = self.qkv_proj(hidden_states)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states, key_states, value_states = self.permute_qkv_for_attn(
            query_states, key_states, value_states, bsz, q_len, self.num_heads, self.num_key_value_heads, self.head_dim
        )

        # Main difference from LlamaAttention is that Qwen3 applies a norm on query and key after the projection
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            flash_attn=self.use_flash_attention_v2,
            transpose_nki_inputs=self.trn_config.transpose_nki_inputs,
        )

        if self.use_flash_attention_v2:
            attn_output = nki_flash_attn_func(
                query_states,
                repeat_kv(key_states, self.num_key_value_groups),
                repeat_kv(value_states, self.num_key_value_groups),
                dropout_p=0.0,  # We never apply dropout in the flash attention path because it produces NaNs.
                softmax_scale=self.scaling,
                causal=True,
                mixed_precision=True,
                transpose_nki_inputs=self.trn_config.transpose_nki_inputs,
            )
            attn_output = nn.functional.dropout(attn_output, p=0.0 if not self.training else self.attention_dropout)
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
    config = Qwen3Config
    _no_split_modules = ["Qwen3DecoderLayer"]

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
            dtype=config.dtype,
        )
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, trn_config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = self.trn_config.gradient_checkpointing

        # Initialize weights and apply final processing
        self.post_init()


class Qwen3ForCausalLM(LlamaForCausalLM):
    config = Qwen3Config
    _no_split_modules = ["Qwen3DecoderLayer"]

    # Pipeline parallelism support
    SUPPORTS_PIPELINE_PARALLELISM = True
    PIPELINE_TRANSFORMER_LAYER_CLS = Qwen3DecoderLayer
    PIPELINE_INPUT_NAMES = ["input_ids", "attention_mask", "labels"]
    PIPELINE_LEAF_MODULE_CLASSE_NAMES = ["Qwen3RMSNorm", "LlamaRotaryEmbedding"]

    def __init__(self, config, trn_config: TrainingNeuronConfig):
        super().__init__(config, trn_config)
        self.model = Qwen3Model(config, trn_config)
