# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
"""Neuron export configurations for models using transformers_neuronx."""


from optimum.exporters.tasks import TasksManager

from ..config import TextNeuronDecoderConfig


register_in_tasks_manager = TasksManager.create_register("neuron")


@register_in_tasks_manager("gpt2", "text-generation")
class GPT2NeuronConfig(TextNeuronDecoderConfig):
    NEURONX_CLASS = "gpt2.model.GPT2ForSampling"


@register_in_tasks_manager("llama", "text-generation")
class LLamaNeuronConfig(TextNeuronDecoderConfig):
    NEURONX_CLASS = "llama.model.LlamaForSampling"
    CONTINUOUS_BATCHING = True
    ATTENTION_lAYOUT = "BSH"


@register_in_tasks_manager("opt", "text-generation")
class OPTNeuronConfig(TextNeuronDecoderConfig):
    NEURONX_CLASS = "opt.model.OPTForSampling"


@register_in_tasks_manager("bloom", "text-generation")
class BloomNeuronConfig(TextNeuronDecoderConfig):
    NEURONX_CLASS = "bloom.model.BloomForSampling"


@register_in_tasks_manager("mistral", "text-generation")
class MistralNeuronConfig(TextNeuronDecoderConfig):
    NEURONX_CLASS = "mistral.model.MistralForSampling"
    CONTINUOUS_BATCHING = True


@register_in_tasks_manager("mixtral", "text-generation")
class MixtralNeuronConfig(TextNeuronDecoderConfig):
    NEURONX_CLASS = "mixtral.model.MixtralForSampling"
    CONTINUOUS_BATCHING = False
