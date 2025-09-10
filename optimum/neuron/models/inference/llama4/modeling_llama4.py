# coding=utf-8
# Copyright 2025 The LLAMA4 and HuggingFace Inc. team. All rights reserved.
#
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

import gc
import logging
from typing import Optional, Tuple

import torch
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from torch import nn
from transformers.models.llama4.configuration_llama4 import Llama4TextConfig

from ..backend.config import NxDNeuronConfig
from ..backend.modules.attention.attention_base import NeuronAttentionBase
from ..backend.modules.attention.gqa import BaseGroupQueryAttention
from ..backend.modules.attention.rope import apply_rotary_polar_compatible, precompute_freqs_cis
from ..backend.modules.decoder import NxDDecoderModel, NxDModelForCausalLM
from ..backend.modules.moe_v2 import initialize_moe_module
from ..backend.modules.rms_norm import NeuronRMSNorm
from ..llama.modeling_llama import NeuronLlamaMLP


logger = logging.getLogger("Neuron")


def preshard_hook_fn(module: torch.nn.Module, model_state_dict: dict, prefix: str) -> bool:
    if isinstance(module, (BaseGroupQueryAttention,)):
        return module.preshard_hook(model_state_dict, prefix)

    return False


def _helper_concat_and_delete_qkv(llama_state_dict, layer_num, attr):
    """
    Helper function to concatenate and delete QKV attributes for fusedqkv (weight or scale).
    Args:
        llama_state_dict: The state dictionary containing model weights
        layer_num: The index of the layer to process
        attr: The attribute to process ('weight' or 'scale')
    """
    llama_state_dict[f"layers.{layer_num}.self_attn.Wqkv.{attr}"] = torch.cat(
        [
            llama_state_dict[f"layers.{layer_num}.self_attn.q_proj.{attr}"],
            llama_state_dict[f"layers.{layer_num}.self_attn.k_proj.{attr}"],
            llama_state_dict[f"layers.{layer_num}.self_attn.v_proj.{attr}"],
        ],
    )

    del llama_state_dict[f"layers.{layer_num}.self_attn.q_proj.{attr}"]
    del llama_state_dict[f"layers.{layer_num}.self_attn.k_proj.{attr}"]
    del llama_state_dict[f"layers.{layer_num}.self_attn.v_proj.{attr}"]


def convert_state_dict_to_fused_qkv(llama_state_dict, cfg: Llama4TextConfig):
    """
    This function concats the qkv weights and scales to a Wqkv weight and scale for fusedqkv, and deletes the qkv weights.
    """

    for l in range(cfg.num_hidden_layers):  # noqa: E741
        _helper_concat_and_delete_qkv(llama_state_dict, l, "weight")

    gc.collect()

    return llama_state_dict


class L2Norm(torch.nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dtype = torch.float32

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x)


class NeuronLlama4Attention(NeuronAttentionBase):
    def __init__(
        self,
        config: Llama4TextConfig,
        neuron_config: NxDNeuronConfig,
        layer_idx: int,
    ):
        super().__init__(config, neuron_config)
        # Llama4Text specific: rotary_emb is not set as it uses a different rope implementation
        self.rotary_freqs = precompute_freqs_cis(config, neuron_config)
        self.qk_norm = None
        if config.use_qk_norm and config.no_rope_layers[layer_idx]:
            # Llama4Text specific: set q_layernorm and k_layernorm for some layers
            self.q_layernorm = L2Norm(config.rms_norm_eps)
            self.k_layernorm = L2Norm(config.rms_norm_eps)

    def rotate_qkv_tensors(
        self,
        position_ids: torch.LongTensor,
        Q: torch.Tensor,
        K: torch.Tensor,
        cos_cache=None,
        sin_cache=None,
    ):
        rotary_freqs = self.rotary_freqs.to(position_ids.device)[position_ids]
        Q, K = apply_rotary_polar_compatible(Q, K, rotary_freqs)
        Q, K = Q.transpose(1, 2), K.transpose(1, 2)
        # If layernorm is used for q and k, apply it AFTER the rotation
        if self.q_layernorm is not None:
            Q = self.q_layernorm(Q)
        if self.k_layernorm is not None:
            K = self.k_layernorm(K)
        return Q, K, cos_cache, sin_cache


class NeuronLlama4TextMoEDecoderLayer(nn.Module):
    def __init__(self, config: Llama4TextConfig, neuron_config: NxDNeuronConfig):
        super().__init__()
        self.moe = initialize_moe_module(
            config=config,
            neuron_config=neuron_config,
            n_shared_experts=1,
            router_dtype=torch.float32,
            router_act_fn="sigmoid",
            fused_shared_experts=True,
            early_expert_affinity_modulation=True,
        )

    def forward(self, hidden_states):
        """Forward pass for the MOE module"""
        return self.moe(hidden_states)[0]


class NeuronLlama4TextDecoderLayer(nn.Module):
    def __init__(self, config: Llama4TextConfig, neuron_config: NxDNeuronConfig, layer_idx: int):
        super().__init__()

        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        self.self_attn = NeuronLlama4Attention(config, neuron_config, layer_idx)

        self.is_moe_layer = layer_idx in config.moe_layers
        if self.is_moe_layer:
            self.feed_forward = NeuronLlama4TextMoEDecoderLayer(config, neuron_config)
        else:
            self.feed_forward = NeuronLlamaMLP(config, neuron_config)

        self.input_layernorm = NeuronRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = NeuronRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states, present_key_value, cos_cache, sin_cache


class NeuronLlama4TextModel(NxDDecoderModel):
    def __init__(self, config: Llama4TextConfig, neuron_config: NxDNeuronConfig):
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
            [
                NeuronLlama4TextDecoderLayer(config, neuron_config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = NeuronRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=not neuron_config.on_device_sampling,
            bias=False,
            pad=True,
            dtype=neuron_config.torch_dtype,
        )


class Llama4NxDModelForCausalLM(NxDModelForCausalLM):
    """
    This class extends LlamaForCausalLM create traceable
    blocks for Neuron.

    Args:
        LlamaForCausalLM (_type_): _description_
    """

    _model_cls = NeuronLlama4TextModel

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: Llama4TextConfig, neuron_config: NxDNeuronConfig
    ) -> dict:
        if "language_model.lm_head.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["language_model.lm_head.weight"]
            del state_dict["language_model.lm_head.weight"]

        dict_keys = list(state_dict.keys())
        for key in dict_keys:
            if key.startswith("language_model.model."):
                new_key = key.replace("language_model.model.", "")
                state_dict[new_key] = state_dict.pop(key)

        key_map = {
            "self_attn.qkv_proj.weight": "self_attn.Wqkv.weight",
            # router
            "feed_forward.router.weight": "feed_forward.moe.router.linear_router.weight",
            # experts
            "feed_forward.experts.gate_up_proj": "feed_forward.moe.expert_mlps.mlp_op.gate_up_proj.weight",
            "feed_forward.experts.down_proj": "feed_forward.moe.expert_mlps.mlp_op.down_proj.weight",
            # shared experts
            "feed_forward.shared_expert.down_proj.weight": "feed_forward.moe.shared_experts.down_proj.weight",
            # scales
            "feed_forward.experts.gate_up_proj.scale": "feed_forward.moe.expert_mlps.mlp_op.gate_up_proj.scale",
            "feed_forward.experts.down_proj.scale": "feed_forward.moe.expert_mlps.mlp_op.down_proj.scale",
        }

        moe_intermediate_size = config.intermediate_size
        state_dict_keys = set(state_dict.keys())
        num_experts = config.num_local_experts
        for layer_n in range(config.num_hidden_layers):
            prefix = f"layers.{layer_n}."
            if prefix + "feed_forward.shared_expert.up_proj.weight" in state_dict_keys:
                exp_down_key = prefix + "feed_forward.experts.down_proj"
                state_dict[exp_down_key] = state_dict[exp_down_key].view(
                    num_experts, moe_intermediate_size, config.hidden_size
                )

                shared_new_key = prefix + "feed_forward.moe.shared_experts.gate_up_proj.weight"
                shared_swig_key = prefix + "feed_forward.shared_expert.up_proj.weight"
                shared_in_key = prefix + "feed_forward.shared_expert.gate_proj.weight"

                state_dict[shared_new_key] = torch.cat([state_dict[shared_in_key], state_dict[shared_swig_key]], dim=0)
                state_dict_keys.add(shared_new_key)

                del state_dict[shared_swig_key]
                del state_dict[shared_in_key]

            for old_key, new_key in key_map.items():
                if prefix + old_key in state_dict_keys:
                    state_dict[prefix + new_key] = state_dict[prefix + old_key]
                    del state_dict[prefix + old_key]

            gc.collect()

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

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

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
        continuous_batching = (batch_size > 1) if batch_size else False
        return NxDNeuronConfig(
            checkpoint_id=checkpoint_id,
            checkpoint_revision=checkpoint_revision,
            batch_size=batch_size,
            sequence_length=sequence_length,
            tp_degree=tensor_parallel_size,
            torch_dtype=auto_cast_type,
            continuous_batching=continuous_batching,
        )
