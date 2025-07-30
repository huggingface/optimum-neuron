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
# Adapted from https://github.com/aws-neuron/neuronx-distributed-inference/blob/9993358ce052fd7a1bb4a7497a6318aac36ed95c/src/neuronx_distributed_inference/modules/moe.py
from neuronx_distributed.modules.moe.expert_mlps import ExpertMLPs
from neuronx_distributed.modules.moe.model import MoE
from neuronx_distributed.modules.moe.routing import RouterTopK


def initialize_moe_module(
    neuron_config,
    num_experts,
    top_k,
    hidden_size,
    intermediate_size,
    hidden_act,
    normalize_top_k_affinities=True,
):
    """
    Initializes and returns an MoE module corresponding to the given configuration.
    """
    router = RouterTopK(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
    )
    expert_mlps = ExpertMLPs(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
        capacity_factor=neuron_config.capacity_factor,
        glu_mlp=neuron_config.glu_mlp,
        normalize_top_k_affinities=normalize_top_k_affinities,
    )
    moe = MoE(
        router=router,
        expert_mlps=expert_mlps,
    )
    # Set MoE module in eval mode
    moe.eval()
    return moe
