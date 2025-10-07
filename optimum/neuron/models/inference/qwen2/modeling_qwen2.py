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
import warnings

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2Config

from ..backend.config import NxDNeuronConfig
from ..backend.modules.decoder import NxDDecoderModel
from ..backend.modules.rms_norm import NeuronRMSNorm
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
        self.input_layernorm = NeuronRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = NeuronRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
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
            [NeuronQwen2DecoderLayer(config, neuron_config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = NeuronRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


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
        instance_type: str,
        batch_size: int,
        sequence_length: int,
        tensor_parallel_size: int,
        auto_cast_type: str,
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
            torch_dtype=auto_cast_type,
            target=instance_type,
            on_device_sampling=on_device_sampling,
            fused_qkv=False,
            continuous_batching=continuous_batching,
        )
