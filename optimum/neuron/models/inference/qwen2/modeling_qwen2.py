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
"""PyTorch Qwen2 model for NXD inference."""

import logging

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2Config

from ..backend.config import NxDNeuronConfig
from ..backend.modules.custom_calls import CustomRMSNorm
from ..backend.modules.decoder import NxDDecoderModel
from ..llama.modeling_llama import (
    LlamaNxDModelForCausalLM,
    NeuronLlamaAttention,
    NeuronLlamaDecoderLayer,
    NeuronLlamaMLP,
)


logger = logging.getLogger("Neuron")


class NeuronQwen2DecoderLayer(NeuronLlamaDecoderLayer):
    """
    The only difference with the NeuronLlamaDecoderLayer is the addition of the QKV projection biases in the attention
    """

    def __init__(self, config: Qwen2Config, neuron_config: NxDNeuronConfig):
        super().__init__(config, neuron_config)
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronLlamaAttention(config, neuron_config, qkv_proj_bias=True)
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


class NxDQwen2Model(NxDDecoderModel):
    """
    Just use the NeuronQwen2DecoderLayer instead of the NeuronLlamaDecoderLayer
    """

    def __init__(self, config: Qwen2Config, neuron_config: NxDNeuronConfig):
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
            [NeuronQwen2DecoderLayer(config, neuron_config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = CustomRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen2NxDModelForCausalLM(LlamaNxDModelForCausalLM):
    """
    Qwen2 model for NXD inference.
    This class is a wrapper around the NxDQwen2Model, which uses NeuronQwen2DecoderLayer.
    """

    _model_cls = NxDQwen2Model

    @classmethod
    def _get_neuron_config(
        cls,
        checkpoint_id: str,
        checkpoint_revision: str,
        batch_size: int,
        sequence_length: int,
        tensor_parallel_size: int,
        auto_cast_type: str,
    ):
        neuron_config = super()._get_neuron_config(
            checkpoint_id=checkpoint_id,
            checkpoint_revision=checkpoint_revision,
            batch_size=batch_size,
            sequence_length=sequence_length,
            tensor_parallel_size=tensor_parallel_size,
            auto_cast_type=auto_cast_type,
        )
        # Do not use fused QKV for Qwen2 models because of the QKV biases
        neuron_config.fused_qkv = False
        return neuron_config
