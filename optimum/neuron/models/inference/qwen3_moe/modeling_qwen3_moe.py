# coding=utf-8
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
# Adapted from https://github.com/aws-neuron/neuronx-distributed-inference/blob/main/src/neuronx_distributed_inference/models/qwen3_moe/modeling_qwen3_moe.py
"""Qwen3 MOE model for NXD inference."""

import gc

import torch
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, ParallelEmbedding
from torch import nn
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from ..backend.config import NxDNeuronConfig
from ..backend.modules.attention.attention_base import NeuronAttentionBase
from ..backend.modules.attention.utils import RotaryEmbedding
from ..backend.modules.decoder import NxDDecoderModel, NxDModelForCausalLM
from ..backend.modules.moe import initialize_moe_module
from ..backend.modules.rms_norm import NeuronRMSNorm
from ..llama.modeling_llama import NeuronLlamaMLP
from ..mixtral.modeling_mixtral import NeuronMixtralDecoderLayer


def convert_qwen3_moe_hf_to_neuron_state_dict(neuron_state_dict, config, neuron_config):
    """
    Helper function which converts the huggingface checkpoints to state dictionary compatible with the stucture of the neuron MoE model.
    """
    assert neuron_config.glu_mlp is True, "Only GLU MLP is supported"

    # to facilitate rank usage in base model
    neuron_state_dict["rank_util.rank"] = torch.arange(0, neuron_config.tp_degree, dtype=torch.int32)

    for l in range(config.num_hidden_layers):  # noqa: E741
        # Rename the q_norm, k_norm names
        neuron_state_dict[f"layers.{l}.self_attn.k_layernorm.weight"] = (
            neuron_state_dict[f"layers.{l}.self_attn.k_norm.weight"].detach().clone()
        )
        del neuron_state_dict[f"layers.{l}.self_attn.k_norm.weight"]

        # Rename the q_norm, k_norm names
        neuron_state_dict[f"layers.{l}.self_attn.q_layernorm.weight"] = (
            neuron_state_dict[f"layers.{l}.self_attn.q_norm.weight"].detach().clone()
        )
        del neuron_state_dict[f"layers.{l}.self_attn.q_norm.weight"]

        if l not in config.mlp_only_layers and (config.num_experts > 0 and (l + 1) % config.decoder_sparse_step == 0):
            # MoE layer

            # Copy the router weights
            neuron_state_dict[f"layers.{l}.mlp.router.linear_router.weight"] = (
                neuron_state_dict[f"layers.{l}.mlp.gate.weight"].detach().clone()
            )
            del neuron_state_dict[f"layers.{l}.mlp.gate.weight"]

            intermediate_size, hidden_size = neuron_state_dict[f"layers.{l}.mlp.experts.0.gate_proj.weight"].shape
            device = neuron_state_dict[f"layers.{l}.mlp.experts.0.gate_proj.weight"].device
            dtype = neuron_state_dict[f"layers.{l}.mlp.experts.0.gate_proj.weight"].dtype

            # copy the MLP parameters
            gate_up_proj = torch.empty(
                config.num_experts,
                hidden_size,
                2 * intermediate_size,
                dtype=dtype,
                device=device,
            )
            for e in range(config.num_experts):
                # Copy gate_proj and up_proj after concatenation
                gate_proj_weights = (
                    neuron_state_dict[f"layers.{l}.mlp.experts.{e}.gate_proj.weight"].T.detach().clone()
                )
                up_proj_weights = neuron_state_dict[f"layers.{l}.mlp.experts.{e}.up_proj.weight"].T.detach().clone()

                gate_up_proj_slice = torch.narrow(gate_up_proj, 0, e, 1)
                gate_proj_slice = torch.narrow(gate_up_proj_slice, 2, 0, intermediate_size)
                gate_proj_slice.copy_(gate_proj_weights)
                up_proj_slice = torch.narrow(gate_up_proj_slice, 2, intermediate_size, intermediate_size)
                up_proj_slice.copy_(up_proj_weights)

                del neuron_state_dict[f"layers.{l}.mlp.experts.{e}.gate_proj.weight"]
                del neuron_state_dict[f"layers.{l}.mlp.experts.{e}.up_proj.weight"]
            neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj

            down_proj = torch.empty(
                config.num_experts,
                intermediate_size,
                hidden_size,
                dtype=dtype,
                device=device,
            )
            for e in range(config.num_experts):
                # Copy down_proj
                down_proj_weights = (
                    neuron_state_dict[f"layers.{l}.mlp.experts.{e}.down_proj.weight"].T.detach().clone()
                )
                down_proj_slice = torch.narrow(down_proj, 0, e, 1)
                down_proj_slice.copy_(down_proj_weights)
                del neuron_state_dict[f"layers.{l}.mlp.experts.{e}.down_proj.weight"]
            neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.down_proj.weight"] = down_proj

        gc.collect()

    return neuron_state_dict


class NeuronQwen3MoEAttention(NeuronAttentionBase):
    def __init__(self, config: Qwen3MoeConfig, neuron_config: NxDNeuronConfig):
        super().__init__(config, neuron_config)
        self.tp_degree = parallel_state.get_tensor_model_parallel_size()
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        # Qwen3Moe specific: set q_layernorm and k_layernorm
        self.q_layernorm = NeuronRMSNorm(self.head_dim, self.rms_norm_eps)
        self.k_layernorm = NeuronRMSNorm(self.head_dim, self.rms_norm_eps)


class NeuronQwen3MoeDecoderLayer(NeuronMixtralDecoderLayer):
    """
    The only difference with the NeuronMixtralDecoderLayer is the use
    of the NeuronQwen3MoEAttention.
    """

    def __init__(self, config: Qwen3MoeConfig, neuron_config: NxDNeuronConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronQwen3MoEAttention(config, neuron_config)

        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            # MoE layer
            self.mlp = initialize_moe_module(
                neuron_config=neuron_config,
                num_experts=config.num_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                hidden_act=config.hidden_act,
            )
        else:
            # Dense layer
            self.mlp = NeuronLlamaMLP(config, neuron_config)

        self.input_layernorm = NeuronRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = NeuronRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )


class NxDQwen3MoeModel(NxDDecoderModel):
    """
    NxDQwen3MoeModel extends the Qwen3MoeModel to be traceable.
    The forward function of this class is traced.
    """

    def __init__(self, config: Qwen3MoeConfig, neuron_config: NxDNeuronConfig):
        super().__init__(config, neuron_config)

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            dtype=neuron_config.dtype,
            shard_across_embedding=True,
        )
        self.layers = nn.ModuleList(
            [
                NeuronQwen3MoeDecoderLayer(config, neuron_config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = NeuronRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=not neuron_config.on_device_sampling,
            bias=False,
        )


class Qwen3MoeNxDModelForCausalLM(NxDModelForCausalLM):
    """
    This class can be used as Qwen3MoeForCausalLM
    """

    _model_cls = NxDQwen3MoeModel

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: Qwen3MoeConfig, neuron_config: NxDNeuronConfig
    ) -> dict:
        return convert_qwen3_moe_hf_to_neuron_state_dict(state_dict, config, neuron_config)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_compiler_args(cls, neuron_config: NxDNeuronConfig):
        compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation --model-type transformer -O1"
        # Add flags for cc-overlap
        compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
        compiler_args += " --auto-cast=none"
        # Enable vector-offset DGE
        compiler_args += " --internal-enable-dge-levels vector_dynamic_offsets"
        compiler_args += " --internal-hlo2tensorizer-options='--verify-hlo=true'"
        return compiler_args

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
        return NxDNeuronConfig(
            checkpoint_id=checkpoint_id,
            checkpoint_revision=checkpoint_revision,
            batch_size=batch_size,
            sequence_length=sequence_length,
            tp_degree=tensor_parallel_size,
            torch_dtype=dtype,
            target=instance_type,
            on_device_sampling=True,
            continuous_batching=continuous_batching,
        )
