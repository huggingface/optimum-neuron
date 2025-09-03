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
"""PyTorch SmolLM3 model for NXD inference."""

import logging

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from torch import nn
from transformers.models.smollm3.configuration_smollm3 import SmolLM3Config

from ..backend.config import NxDNeuronConfig  # noqa: E402
from ..backend.modules.attention.attention_base import NeuronAttentionBase
from ..backend.modules.attention.utils import RotaryEmbedding
from ..backend.modules.decoder import NxDDecoderModel
from ..backend.modules.rms_norm import NeuronRMSNorm
from ..llama.modeling_llama import (
    LlamaNxDModelForCausalLM,
    NeuronLlamaDecoderLayer,
)


logger = logging.getLogger("Neuron")


class NeuronSmolLM3Attention(NeuronAttentionBase):
    """
    The only difference with the NeuronAttentionBase is the definition of the SmolLM3 rotary embedding
    """

    def __init__(
        self,
        config: SmolLM3Config,
        neuron_config: NxDNeuronConfig,
        layer_idx: int,
        qkv_proj_bias: bool | None = False,
        o_proj_bias: bool | None = False,
        qk_scale: float | None = None,
    ):
        if config.use_sliding_window:
            raise ValueError("SmolLM3 for Neuron does not support sliding window attention.")
        if getattr(config, "rope_scaling", None) is not None:
            raise ValueError("SmolLM3 for Neuron does not support rope scaling.")
        super().__init__(
            config, neuron_config, qkv_proj_bias=qkv_proj_bias, o_proj_bias=o_proj_bias, qk_scale=qk_scale
        )
        if config.no_rope_layers[layer_idx]:
            # Yes, the condition is slightly counter-intuitive, but that is the transformers convention
            head_dim = config.hidden_size // config.num_attention_heads
            self.rotary_emb = RotaryEmbedding(
                head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
            )
        else:
            self.rotary_emb = None


class NeuronSmolLM3DecoderLayer(NeuronLlamaDecoderLayer):
    def __init__(self, config: SmolLM3Config, neuron_config: NxDNeuronConfig, layer_idx: int):
        super().__init__(config, neuron_config)
        self.self_attn = NeuronSmolLM3Attention(config, neuron_config, layer_idx)


class NxDSmolLM3Model(NxDDecoderModel):
    """
    The neuron version of the SmolLM3Model
    """

    def __init__(self, config: SmolLM3Config, neuron_config: NxDNeuronConfig):
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
            [
                NeuronSmolLM3DecoderLayer(config, neuron_config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = NeuronRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class SmolLM3NxDModelForCausalLM(LlamaNxDModelForCausalLM):
    _model_cls = NxDSmolLM3Model
