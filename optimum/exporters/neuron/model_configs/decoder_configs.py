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
"""Neuron export configurations for decoder models."""

from optimum.exporters.tasks import TasksManager

from ....neuron.models.inference.hlo.granite.model import GraniteHloModelForCausalLM
from ....neuron.models.inference.hlo.llama.model import LlamaHloModelForCausalLM
from ....neuron.models.inference.hlo.phi3.model import Phi3HloModelForCausalLM
from ....neuron.models.inference.hlo.qwen2.model import Qwen2HloModelForCausalLM
from ..base import NeuronExportConfig


register_in_tasks_manager = TasksManager.create_register("neuron")


class NeuronDecoderExportConfig(NeuronExportConfig):
    """
    Base class for configuring the export of Neuron Decoder models

    Class attributes:

    - NEURONX_CLASS (`type`) -- the class to use to instantiate the model.

    The NEURONX_CLASS must always be defined in each model configuration.
    """

    NEURONX_CLASS = None

    def __init__(self, task: str):
        pass

    @property
    def neuronx_class(self):
        return self.NEURONX_CLASS


@register_in_tasks_manager("llama", "text-generation")
class LLamaNeuronConfig(NeuronDecoderExportConfig):
    NEURONX_CLASS = LlamaHloModelForCausalLM


@register_in_tasks_manager("qwen2", "text-generation")
class Qwen2NeuronConfig(NeuronDecoderExportConfig):
    NEURONX_CLASS = Qwen2HloModelForCausalLM


@register_in_tasks_manager("granite", "text-generation")
class GraniteNeuronConfig(NeuronDecoderExportConfig):
    NEURONX_CLASS = GraniteHloModelForCausalLM


@register_in_tasks_manager("phi3", "text-generation")
class Phi3NeuronConfig(NeuronDecoderExportConfig):
    NEURONX_CLASS = Phi3HloModelForCausalLM
