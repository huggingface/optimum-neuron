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
"""PyTorch LLaMA model for NXD inference."""

import logging

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from torch import nn
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaRMSNorm, LlamaRotaryEmbedding

from ..backend.config import NxDNeuronConfig  # noqa: E402
from ..backend.modules.attention.attention_base import NeuronAttentionBase
from ..backend.modules.attention.utils import (
    RotaryEmbedding,
)
from ..backend.modules.decoder import NxDDecoderModel
from ..llama.modeling_llama import (
    LlamaNxDModelForCausalLM,
    NeuronLlamaDecoderLayer,
    NeuronLlamaMLP,
)


logger = logging.getLogger("Neuron")


class NeuronQwen2Attention(NeuronAttentionBase):
    """
    Compared with LlamaAttention, this class just
    1. replaces the q_proj, k_proj, v_proj with column parallel layer
    2. replaces the o_proj with row parallel layer
    3. update self.num_head to be self.num_head / tp_degree
    4. update self.num_key_value_heads to be self.num_key_value_heads / tp_degree
    5. update forward() method to adjust to changes from self.num_head
    """

    def __init__(self, config: LlamaConfig, neuron_config: NxDNeuronConfig):
        super().__init__(config, neuron_config, qkv_bias=True)
        head_dim = config.hidden_size // config.num_attention_heads
        if not hasattr(config, "rope_scaling") or config.rope_scaling is None:
            self.rotary_emb = RotaryEmbedding(
                head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
            )
        else:
            rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type", None))
            if rope_type == "llama3":
                self.rotary_emb = Llama3RotaryEmbedding(
                    dim=head_dim,
                    max_position_embeddings=config.max_position_embeddings,
                    base=config.rope_theta,
                    factor=config.rope_scaling["factor"],
                    low_freq_factor=config.rope_scaling["low_freq_factor"],
                    high_freq_factor=config.rope_scaling["high_freq_factor"],
                    original_max_position_embeddings=config.rope_scaling["original_max_position_embeddings"],
                )
            else:
                # LlamaRotaryEmbedding automatically chooses the correct scaling type from config.
                # Warning: The HF implementation may have precision issues when run on Neuron.
                # We include it here for compatibility with other scaling types.
                self.rotary_emb = LlamaRotaryEmbedding(config)


class NeuronQwen2DecoderLayer(NeuronLlamaDecoderLayer):
    """
    Just replace the attention with the NXD version, and MLP with the NXD version
    """

    def __init__(self, config: LlamaConfig, neuron_config: NxDNeuronConfig):
        super().__init__(config, neuron_config)
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronQwen2Attention(config, neuron_config)
        self.mlp = NeuronLlamaMLP(config, neuron_config)
        logger.debug(
            f"Instantiating RMSNorm modules with hidden size {config.hidden_size} and EPS {config.rms_norm_eps}"
        )
        self.input_layernorm = LlamaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = LlamaRMSNorm(
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
    The neuron version of the LlamaModel
    """

    def __init__(self, config: LlamaConfig, neuron_config: NxDNeuronConfig):
        super().__init__(config, neuron_config)

        if parallel_state.model_parallel_is_initialized():
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
        else:
            self.embed_tokens = nn.Embedding(
                config.vocab_size,
                config.hidden_size,
                config.pad_token_id,
            )
            self.lm_head = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
            )

        self.layers = nn.ModuleList(
            [NeuronQwen2DecoderLayer(config, neuron_config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen2NxDModelForCausalLM(LlamaNxDModelForCausalLM):

    _model_cls = NxDQwen2Model
