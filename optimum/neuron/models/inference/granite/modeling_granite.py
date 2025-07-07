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
# Adapted from https://github.com/aws-neuron/neuronx-distributed-inference/blob/9993358ce052fd7a1bb4a7497a6318aac36ed95c/src/neuronx_distributed_inference/models/llama/modeling_llama.py
"""PyTorch Granite model for NXD inference."""

import logging
from typing import Any

import torch
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from torch import nn
from transformers.models.granite.configuration_granite import GraniteConfig

from ..backend.config import NxDNeuronConfig
from ..backend.modules.custom_calls import CustomRMSNorm
from ..backend.modules.decoder import NxDDecoderModel
from ..llama.modeling_llama import LlamaNxDModelForCausalLM, NeuronLlamaAttention, NeuronLlamaMLP


logger = logging.getLogger("Neuron")


class NeuronGraniteDecoderLayer(nn.Module):
    """A Granite specific decoder layer with:
    - custom scaling factor applied to the qk product in attention,
    - custom scaling factors applied to attention and mlp outputs
    """

    def __init__(self, config: GraniteConfig, neuron_config: NxDNeuronConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronLlamaAttention(config, neuron_config, qk_scale=config.attention_multiplier)
        self.mlp = NeuronLlamaMLP(config, neuron_config)
        logger.debug(
            f"Instantiating RMSNorm modules with hidden size {config.hidden_size} and EPS {config.rms_norm_eps}"
        )
        self.input_layernorm = CustomRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = CustomRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.qkv_kernel_enabled = neuron_config.qkv_kernel_enabled
        self.mlp_kernel_enabled = neuron_config.mlp_kernel_enabled
        self.mlp_kernel_fuse_residual_add = neuron_config.mlp_kernel_fuse_residual_add
        self.sequence_parallel_enabled = neuron_config.sequence_parallel_enabled
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: tuple[torch.Tensor | None] = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor | None]]:
        residual = hidden_states

        # RMSNorm (fused with QKV kernel when SP is disabled)
        if (not self.qkv_kernel_enabled or self.sequence_parallel_enabled) and self.input_layernorm:
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            rmsnorm=self.input_layernorm,
            **kwargs,
        )

        # Granite specific: attention output is multiplied by residual multiplier
        hidden_states = hidden_states * self.config.residual_multiplier

        if self.mlp_kernel_enabled and self.mlp_kernel_fuse_residual_add:
            assert not self.sequence_parallel_enabled, (
                "mlp_kernel_fuse_residual_add should be off when sequence parallelism is enabled"
            )
            # First residual add handled in the MLP kernel
            hidden_states, residual = self.mlp(
                hidden_states,
                rmsnorm=self.post_attention_layernorm,
                residual=residual,
            )
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            # RMSNorm (fused with QKV kernel when SP is disabled)
            if not self.mlp_kernel_enabled or self.sequence_parallel_enabled:
                hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states, _ = self.mlp(
                hidden_states,
                rmsnorm=self.post_attention_layernorm,
            )

        # Granite specific: MLP output is multiplied by residual_multiplier
        hidden_states = hidden_states * self.config.residual_multiplier

        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache)
        return outputs


class NxDGraniteEmbedding(ParallelEmbedding):
    """A custom neuron parallel embedding layer with scaled outputs"""

    def __init__(self, config: GraniteConfig, neuron_config: NxDNeuronConfig):
        super().__init__(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            dtype=neuron_config.torch_dtype,
            shard_across_embedding=not neuron_config.vocab_parallel,
            sequence_parallel_enabled=False,
            pad=True,
            use_spmd_rank=neuron_config.vocab_parallel,
        )
        self.config = config

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # Granite specific: embeddings are multiplied by custom scale factor
        embeddings = super().forward(input_)
        return embeddings * self.config.embedding_multiplier


class NxDGraniteHead(ColumnParallelLinear):
    """A custom lm head neuron column parallel linear layer with scaled logits"""

    def __init__(self, config: GraniteConfig, neuron_config: NxDNeuronConfig):
        super().__init__(
            config.hidden_size,
            config.vocab_size,
            gather_output=not neuron_config.on_device_sampling,
            bias=False,
            pad=True,
        )
        self.config = config

    def forward(self, input: torch.Tensor, *_: Any) -> torch.Tensor:
        logits = super().forward(input)
        # Granite specific: divide logits by custom scaling factor
        return logits / self.config.logits_scaling


class NxDGraniteModel(NxDDecoderModel):
    """
    The differences with the standard neuron decoder are:
    - the use of scaled embeddings,
    - the used of the custom granite layers with scaled attention and mlp,
    - the use of scaled head logits.
    """

    def __init__(self, config: GraniteConfig, neuron_config: NxDNeuronConfig):
        super().__init__(config, neuron_config)

        self.embed_tokens = NxDGraniteEmbedding(config, neuron_config)

        self.lm_head = NxDGraniteHead(config, neuron_config)

        self.layers = nn.ModuleList(
            [NeuronGraniteDecoderLayer(config, neuron_config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = CustomRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class GraniteNxDModelForCausalLM(LlamaNxDModelForCausalLM):
    """
    This class extends LlamaForCausalLM create traceable
    blocks for Neuron.

    Args:
        LlamaForCausalLM (_type_): _description_
    """

    _model_cls = NxDGraniteModel
