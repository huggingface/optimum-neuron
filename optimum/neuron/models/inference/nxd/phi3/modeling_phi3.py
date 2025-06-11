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
"""PyTorch Phi3 model for NXD inference."""

import logging

import torch
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.mappings import (
    gather_from_sequence_parallel_region,
)
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.phi3.modeling_phi3 import Phi3Config

from ..backend.config import NxDNeuronConfig  # noqa: E402
from ..backend.modules.custom_calls import CustomRMSNorm
from ..backend.modules.decoder import NxDDecoderModel
from ..llama.modeling_llama import (
    LlamaNxDModelForCausalLM,
    NeuronLlamaAttention,
    NeuronLlamaDecoderLayer,
)


logger = logging.getLogger("Neuron")


class NeuronPhi3MLP(nn.Module):
    """
    This class just replace the linear layers (gate_up_proj and down_proj) with column and row parallel layers
    """

    def __init__(self, config: Phi3Config, neuron_config: NxDNeuronConfig):
        super().__init__()
        self.tp_degree = neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]

        self.sequence_parallel_enabled = getattr(neuron_config, "sequence_parallel_enabled", False)
        self.sequence_dimension = 1 if self.sequence_parallel_enabled else None
        self.rms_norm_eps = config.rms_norm_eps
        self.logical_nc_config = neuron_config.logical_nc_config
        self.gate_up_proj = ColumnParallelLinear(
            self.hidden_size,
            2 * self.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=neuron_config.torch_dtype,
            pad=True,
            sequence_parallel_enabled=False,
            sequence_dimension=None,
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=neuron_config.torch_dtype,
            pad=True,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            sequence_dimension=self.sequence_dimension,
            reduce_dtype=neuron_config.rpl_reduce_dtype,
        )

    def forward(self, x, rmsnorm=None):
        # all-gather is done here instead of CPL layers to
        # avoid 2 all-gathers from up and gate projections
        if self.sequence_parallel_enabled:
            x = gather_from_sequence_parallel_region(x, self.sequence_dimension)

        gate_up_proj_output = self.gate_up_proj(x)
        gate_proj_output, up_proj_output = torch.tensor_split(gate_up_proj_output, 2, dim=-1)
        down_proj_input = self.act_fn(gate_proj_output) * up_proj_output
        output = self.down_proj(down_proj_input)
        return output, None


class NeuronPhi3DecoderLayer(NeuronLlamaDecoderLayer):
    """
    The only difference with the NeuronLlamaDecoderLayer is the use of NeuronPhi3MLP instead of NeuronLlamaMLP
    """

    def __init__(self, config: Phi3Config, neuron_config: NxDNeuronConfig):
        if not neuron_config.fused_qkv:
            raise ValueError("Phi3 models must be exported with fused_qkv set to True")
        super().__init__(config, neuron_config)
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronLlamaAttention(config, neuron_config)
        self.mlp = NeuronPhi3MLP(config, neuron_config)
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


class NxDPhi3Model(NxDDecoderModel):
    """
    Just use the NeuronPhi3DecoderLayer instead of the NeuronLlamaDecoderLayer
    """

    def __init__(self, config: Phi3Config, neuron_config: NxDNeuronConfig):
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
            [NeuronPhi3DecoderLayer(config, neuron_config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = CustomRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Phi3NxDModelForCausalLM(LlamaNxDModelForCausalLM):
    """
    Phi3 model for NxD inference.
    This class inherits from the Llama model class but uses the NxDPhi3Model that uses fused MLP projections.
     It also changes the state_dict loading method accordingly.
    """

    _model_cls = NxDPhi3Model

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: Phi3Config, neuron_config: NxDNeuronConfig) -> dict:
        # Rename the fused qkv projections as expected by the NeuronAttentionBase
        for l in range(config.num_hidden_layers):  # noqa: E741
            state_dict[f"layers.{l}.self_attn.Wqkv.weight"] = state_dict[f"layers.{l}.self_attn.qkv_proj.weight"]
            state_dict.pop(f"layers.{l}.self_attn.qkv_proj.weight")

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
