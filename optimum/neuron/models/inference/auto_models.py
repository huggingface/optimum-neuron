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
from .gpt_oss.modeling_gpt_oss import GptOssNxDModelForCausalLM
from .granite.modeling_granite import GraniteNxDModelForCausalLM
from .llama.modeling_llama import LlamaNxDModelForCausalLM
from .llama4.modeling_llama4 import Llama4NxDModelForCausalLM
from .mixtral.modeling_mixtral import MixtralNxDModelForCausalLM
from .phi3.modeling_phi3 import Phi3NxDModelForCausalLM
from .qwen2.modeling_qwen2 import Qwen2NxDModelForCausalLM
from .qwen3.modeling_qwen3 import Qwen3NxDModelForCausalLM, Qwen3NxDModelForEmbedding
from .qwen3_moe.modeling_qwen3_moe import Qwen3MoeNxDModelForCausalLM
from .smollm3.modeling_smollm3 import SmolLM3NxDModelForCausalLM


prioritize_hlo_backend = os.environ.get("OPTIMUM_NEURON_PRIORITIZE_HLO_BACKEND", "0") == "1"


def register_neuron_model_for_inference(model_type: str, task: str):
    """
    Register a model for inference.
    """
    return register_neuron_model(model_type, task, "inference")


@register_neuron_model_for_inference("granite", "text-generation")
class GraniteNeuronModelForCausalLM(GraniteNxDModelForCausalLM):
    """
    Granite model with NxD backend for inference on AWS Neuron.
    """

    pass


@register_neuron_model_for_inference("gpt_oss", "text-generation")
class GptOssNeuronModelForCausalLM(GptOssNxDModelForCausalLM):
    """
    GPT-OSS model with NxD backend for inference on AWS Neuron.
    """

    pass


@register_neuron_model_for_inference("llama", "text-generation")
class LlamaNeuronModelForCausalLM(LlamaNxDModelForCausalLM):
    """
    Llama model with NxD backend for inference on AWS Neuron.
    """

    pass


@register_neuron_model_for_inference("llama4_text", "text-generation")
class Llama4NeuronModelForCausalLM(Llama4NxDModelForCausalLM):
    """
    Llama4 model with NxD backend for inference on AWS Neuron.
    """

    pass


@register_neuron_model_for_inference("mixtral", "text-generation")
class MixtralNeuronModelForCausalLM(MixtralNxDModelForCausalLM):
    """
    Mixtral model with NxD backend for inference on AWS Neuron.
    """

    pass


@register_neuron_model_for_inference("phi3", "text-generation")
class Phi3NeuronModelForCausalLM(Phi3NxDModelForCausalLM):
    """
    Phi3 model with NxD backend for inference on AWS Neuron.
    """

    pass


@register_neuron_model_for_inference("qwen2", "text-generation")
class Qwen2NeuronModelForCausalLM(Qwen2NxDModelForCausalLM):
    """
    Qwen2 model with NxD backend for inference on AWS Neuron.
    """

    pass


@register_neuron_model_for_inference("qwen3", "text-generation")
class Qwen3NeuronModelForCausalLM(Qwen3NxDModelForCausalLM):
    """
    Qwen3 model with NxD backend for inference on AWS Neuron.
    """

    pass


@register_neuron_model_for_inference("qwen3", "feature-extraction")
class Qwen3NeuronModelForEmbedding(Qwen3NxDModelForEmbedding):
    """
    Qwen3 model with NxD backend for inference on AWS Neuron.
    """

    pass


@register_neuron_model_for_inference("qwen3_moe", "text-generation")
class Qwen3MoeNeuronModelForCausalLM(Qwen3MoeNxDModelForCausalLM):
    """
    Qwen3Moe model with NxD backend for inference on AWS Neuron.
    """

    pass


@register_neuron_model_for_inference("smollm3", "text-generation")
class SmolLM3NeuronModelForCausalLM(SmolLM3NxDModelForCausalLM):
    """
    SomlLM3 model with NxD backend for inference on AWS Neuron.
    """

    pass
