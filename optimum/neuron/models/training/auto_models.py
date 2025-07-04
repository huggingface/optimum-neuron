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

from ..auto_model import register_neuron_model
from .granite.modeling_granite import GraniteForCausalLM, GraniteModel
from .llama.modeling_llama import LlamaForCausalLM, LlamaModel
from .qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3Model


def register_neuron_model_for_training(model_type: str, task: str):
    """
    Register a model for training.
    """
    return register_neuron_model(model_type, task, "training")


# Register all training models for "model" task (e.g. model without any head)
register_neuron_model_for_training("llama", "model")(LlamaModel)
register_neuron_model_for_training("granite", "model")(GraniteModel)
register_neuron_model_for_training("qwen3", "model")(Qwen3Model)

# Register all training models for CausalLM task
register_neuron_model_for_training("llama", "text-generation")(LlamaForCausalLM)
register_neuron_model_for_training("granite", "text-generation")(GraniteForCausalLM)
register_neuron_model_for_training("qwen3", "text-generation")(Qwen3ForCausalLM)
