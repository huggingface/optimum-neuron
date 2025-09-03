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

import gc
import logging

import torch
from transformers.models.phi3.modeling_phi3 import Phi3Config

from ..backend.config import NxDNeuronConfig  # noqa: E402
from ..llama.modeling_llama import (
    LlamaNxDModelForCausalLM,
    NxDLlamaModel,
)


logger = logging.getLogger("Neuron")


class Phi3NxDModelForCausalLM(LlamaNxDModelForCausalLM):
    """
    Phi3 model for NxD inference.
    This class inherits from the Neuron Llama model class since the Phi3 modeling is just a
    Llama modeling with fused qkv and mlp projections.
    The state_dict loading method is modified to unfuse the weights at loading time.
    """

    _model_cls = NxDLlamaModel

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: Phi3Config, neuron_config: NxDNeuronConfig) -> dict:
        for l in range(config.num_hidden_layers):  # noqa: E741
            if neuron_config.fused_qkv:
                # Just rename the qkv projection to the expected name
                state_dict[f"layers.{l}.self_attn.Wqkv.weight"] = state_dict.pop(
                    f"layers.{l}.self_attn.qkv_proj.weight"
                )
            else:
                # Unfuse the qkv projections as expected by the NeuronAttentionBase
                fused_qkv = state_dict[f"layers.{l}.self_attn.qkv_proj.weight"]
                state_dict[f"layers.{l}.self_attn.q_proj.weight"] = fused_qkv[: config.hidden_size, :].clone().detach()
                k_weight, v_weight = torch.chunk(fused_qkv[config.hidden_size :, :], 2, dim=0)
                state_dict[f"layers.{l}.self_attn.k_proj.weight"] = k_weight.clone().detach()
                state_dict[f"layers.{l}.self_attn.v_proj.weight"] = v_weight.clone().detach()
                state_dict.pop(f"layers.{l}.self_attn.qkv_proj.weight")
                gc.collect()
            # Unfuse the mlp projections
            gate_weight, up_weight = torch.chunk(state_dict[f"layers.{l}.mlp.gate_up_proj.weight"], 2, dim=0)
            state_dict[f"layers.{l}.mlp.gate_proj.weight"] = gate_weight.clone().detach()
            state_dict[f"layers.{l}.mlp.up_proj.weight"] = up_weight.clone().detach()
            state_dict.pop(f"layers.{l}.mlp.gate_up_proj.weight")
            gc.collect()

        # to facilitate rank usage in attention
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        # to facilitate rank usage in base model
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        return state_dict
