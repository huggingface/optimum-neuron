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
    ParallelEmbedding,
)
from torch import nn
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from ..backend.config import NxDNeuronConfig
from ..backend.modules.decoder import NxDDecoderModelForEmbedding, NxDModelForEmbeddingLM
from ..backend.modules.rms_norm import NeuronRMSNorm
from ..llama.modeling_llama import (
    convert_state_dict_to_fused_qkv,
)
from ..qwen3.modeling_qwen3 import (
    NeuronQwen3DecoderLayer,
)


logger = logging.getLogger("Neuron")


class NxDQwen3EmbeddingModel(NxDDecoderModelForEmbedding):
    """
    The neuron version of the Qwen3Model with output_hidden_states support
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

        self.layers = nn.ModuleList(
            [NeuronQwen3DecoderLayer(config, neuron_config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = NeuronRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen3NxDModelForCausalLMEmbedding(NxDModelForEmbeddingLM):
    _model_cls = NxDQwen3EmbeddingModel

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: Qwen3Config, neuron_config: NxDNeuronConfig) -> dict:
        for l in range(config.num_hidden_layers):
            attn_prefix = f"layers.{l}.self_attn"
            state_dict[f"{attn_prefix}.k_layernorm.weight"] = state_dict[f"{attn_prefix}.k_norm.weight"]
            state_dict.pop(f"{attn_prefix}.k_norm.weight")
            state_dict[f"{attn_prefix}.q_layernorm.weight"] = state_dict[f"{attn_prefix}.q_norm.weight"]
            state_dict.pop(f"{attn_prefix}.q_norm.weight")

        if neuron_config.fused_qkv:
            state_dict = convert_state_dict_to_fused_qkv(state_dict, config)

        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        return state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

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
        continuous_batching = False  # Disable for embeddings
        on_device_sampling = False  # Disable for embeddings
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

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pooling_type: str = "last",
        normalize: bool = False,
    ) -> torch.Tensor:
        """
        Get embeddings using the parent's forward infrastructure.
        """
        batch_size, seq_len = input_ids.shape

        # Create position_ids
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        else:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Create dummy sampling_params (required by parent forward but not used for embeddings)
        from ..backend.modules.generation.sampling import prepare_sampling_params

        sampling_params = prepare_sampling_params(
            batch_size=batch_size,
            top_k=1,
            top_p=1.0,
            temperature=1.0,
        )

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            sampling_params=sampling_params,
        )

        hidden_states = outputs.hidden_states
        return hidden_states
