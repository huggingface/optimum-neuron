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
# Adapted from https://github.com/aws-neuron/neuronx-distributed-inference/blob/9993358ce052fd7a1bb4a7497a6318aac36ed95c/src/neuronx_distributed_inference/models/Qwen3/modeling_Qwen3.py
"""PyTorch Qwen3 model for NXD inference."""

import logging

import torch
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from torch import nn
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from ..backend.config import NxDNeuronConfig
from ..backend.modules.attention.attention_base import NeuronAttentionBase
from ..backend.modules.custom_calls import CustomRMSNorm
from ..backend.modules.decoder import NxDDecoderModel
from ..llama.modeling_llama import (
    LlamaNxDModelForCausalLM,
    LlamaRotaryEmbedding,
    NeuronLlamaDecoderLayer,
    convert_state_dict_to_fused_qkv,
)


logger = logging.getLogger("Neuron")


class NeuronQwen3Attention(NeuronAttentionBase):
    """
    Compared with NeuronLLamaAttention, this class uses CustomRMSNorm after the the query and key projections.
    """

    def __init__(self, config: Qwen3Config, neuron_config: NxDNeuronConfig):
        super().__init__(config, neuron_config)
        self.q_layernorm = CustomRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_layernorm = CustomRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config)


class NeuronQwen3DecoderLayer(NeuronLlamaDecoderLayer):
    """
    Just use the NeuronQwen3Attention instead of the NeuronLlamaAttention
    """

    def __init__(self, config: Qwen3Config, neuron_config: NxDNeuronConfig):
        super().__init__(config, neuron_config)
        self.self_attn = NeuronQwen3Attention(config, neuron_config)


class NxDQwen3Model(NxDDecoderModel):
    """
    The neuron version of the Qwen3Model
    """

    def __init__(self, config: Qwen3Config, neuron_config: NxDNeuronConfig):
        super().__init__(config, neuron_config)

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            dtype=neuron_config.torch_dtype,
            shard_across_embedding=not neuron_config.vocab_parallel,
            sequence_parallel_enabled=False,
            pad=True,
            use_spmd_rank=neuron_config.vocab_parallel,
        )

        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=not neuron_config.on_device_sampling,
            bias=False,
            pad=True,
        )

        self.layers = nn.ModuleList(
            [NeuronQwen3DecoderLayer(config, neuron_config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = CustomRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen3NxDModelForCausalLM(LlamaNxDModelForCausalLM):
    """
    Qwen3 model for NxD inference.
    This class is a wrapper around the NxDQwen3Model, which uses NeuronQwen3DecoderLayer.
    """

    _model_cls = NxDQwen3Model

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: Qwen3Config, neuron_config: NxDNeuronConfig) -> dict:
        # Rename the QK projection layernorms to match the NeuronAttentionBase expectations
        for l in range(config.num_hidden_layers):
            attn_prefix = f"layers.{l}.self_attn"
            state_dict[f"{attn_prefix}.k_layernorm.weight"] = state_dict[f"{attn_prefix}.k_norm.weight"]
            state_dict.pop(f"{attn_prefix}.k_norm.weight")
            state_dict[f"{attn_prefix}.q_layernorm.weight"] = state_dict[f"{attn_prefix}.q_norm.weight"]
            state_dict.pop(f"{attn_prefix}.q_norm.weight")

        if neuron_config.fused_qkv:
            state_dict = convert_state_dict_to_fused_qkv(state_dict, config)

        if neuron_config.vocab_parallel:
            # TODO: this hack can be removed after replication_id is ready to use
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(0, neuron_config.local_ranks_size)

        # to facilitate rank usage in attention
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        # to facilitate rank usage in base model
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        return state_dict
