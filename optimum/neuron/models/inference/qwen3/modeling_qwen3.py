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
import warnings

import torch
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from torch import nn
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from ..backend.config import NxDNeuronConfig
from ..backend.modules.attention.attention_base import NeuronAttentionBase
from ..backend.modules.decoder import NxDDecoderModel
from ..backend.modules.rms_norm import NeuronRMSNorm
from ..llama.modeling_llama import (
    LlamaNxDModelForCausalLM,
    LlamaRotaryEmbedding,
    NeuronLlamaDecoderLayer,
    convert_state_dict_to_fused_qkv,
)


logger = logging.getLogger("Neuron")


class NeuronQwen3Attention(NeuronAttentionBase):
    """
    Compared with NeuronLLamaAttention, this class uses NeuronRMSNorm after the the query and key projections.
    """

    def __init__(self, config: Qwen3Config, neuron_config: NxDNeuronConfig):
        super().__init__(config, neuron_config)
        self.rotary_emb = LlamaRotaryEmbedding(config)
        # Qwen3 specific: set q_layernorm and k_layernorm
        self.q_layernorm = NeuronRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_layernorm = NeuronRMSNorm(self.head_dim, eps=config.rms_norm_eps)


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
            shard_across_embedding=True,
            pad=True,
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
        self.norm = NeuronRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


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

        # to facilitate rank usage in attention
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        # to facilitate rank usage in base model
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        return state_dict

    @classmethod
    def _get_neuron_config(
        cls,
        checkpoint_id: str,
        checkpoint_revision: str,
        instance_type: str,
        batch_size: int,
        sequence_length: int,
        tensor_parallel_size: int,
        dtype: torch.dtype,
    ):
        continuous_batching = (batch_size > 1) if batch_size else False
        on_device_sampling = True
        if continuous_batching and tensor_parallel_size == 2:
            # Neuron SDK 2.24 bug: the model will produce garbage output when continuous_batching is enabled
            # if the tensor parallel size is 2 and on_device_sampling is enabled.
            warnings.warn(
                "Activating continuous batching but disabling on-device sampling because of a neuron runtime bug when tensor parallel size is 2."
            )
            on_device_sampling = False
        return NxDNeuronConfig(
            checkpoint_id=checkpoint_id,
            checkpoint_revision=checkpoint_revision,
            batch_size=batch_size,
            sequence_length=sequence_length,
            tp_degree=tensor_parallel_size,
            torch_dtype=dtype,
            target=instance_type,
            on_device_sampling=on_device_sampling,
            fused_qkv=True,
            continuous_batching=continuous_batching,
        )
