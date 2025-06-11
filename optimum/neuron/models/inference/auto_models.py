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
"""
The symbols declared in this file are imported by the auto model lookup method.

Add each neuron model class supported for inference below to allow it to be instantiated
automatically by the neuron factory classes such as NeuronModelForCausalLM.
"""

import os

from ..auto_model import register_neuron_model
from .hlo.granite.model import GraniteHloModelForCausalLM
from .hlo.phi3.model import Phi3HloModelForCausalLM
from .nxd.llama.modeling_llama import LlamaNxDModelForCausalLM
from .nxd.mixtral.modeling_mixtral import MixtralNxDModelForCausalLM
from .nxd.qwen2.modeling_qwen2 import Qwen2NxDModelForCausalLM


prioritize_hlo_backend = os.environ.get("OPTIMUM_NEURON_PRIORITIZE_HLO_BACKEND", "0") == "1"


def register_neuron_model_for_inference(model_type: str, task: str):
    """
    Register a model for inference.
    """
    return register_neuron_model(model_type, task, "inference")


@register_neuron_model_for_inference("granite", "text-generation")
class GraniteModelForCausalLM(GraniteHloModelForCausalLM):
    """
    Granite model with HLO backend for inference on AWS Neuron.
    """

    pass


@register_neuron_model_for_inference("llama", "text-generation")
class LLamaModelForCausalLM(LlamaNxDModelForCausalLM):
    """
    Llama model with NxD backend for inference on AWS Neuron.
    """

    pass


@register_neuron_model_for_inference("phi3", "text-generation")
class Phi3ModelForCausalLM(Phi3HloModelForCausalLM):
    """
    Phi3 model with HLO backend for inference on AWS Neuron.
    """

    pass


@register_neuron_model_for_inference("qwen2", "text-generation")
class Qwen2ModelForCausalLM(Qwen2NxDModelForCausalLM):
    """
    Qwen2 model with NxD backend for inference on AWS Neuron.
    """

    pass


@register_neuron_model_for_inference("mixtral", "text-generation")
class MixtralNeuronModelForCausalLM(MixtralNxDModelForCausalLM):
    """
    Mixtral model with NxD backend for inference on AWS Neuron.
    """

    pass
