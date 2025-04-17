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

from ....neuron.backends.hlo.config import HloNeuronConfig
from ....neuron.backends.hlo.decoder import NeuronHloDecoderModel
from ....neuron.models.inference.hlo.granite.model import GraniteHloModel
from ....neuron.models.inference.hlo.llama.model import LlamaHloModel
from ....neuron.models.inference.hlo.phi3.model import Phi3HloModel
from ....neuron.models.inference.hlo.qwen2.model import Qwen2HloModel
from ..base import NeuronExportConfig


register_in_tasks_manager = TasksManager.create_register("neuron")


class NeuronDecoderExportConfig(NeuronExportConfig):
    """
    Base class for configuring the export of Neuron Decoder models

    Class attributes:

    - INPUT_ARGS (`Tuple[Union[str, Tuple[Union[str, Tuple[str]]]]]`) -- A tuple where each element is either:
        - An argument  name, for instance "batch_size" or "sequence_length", that indicates that the argument can
        be passed to export the model,
    - NEURONX_CLASS (`type`) -- the class to use to instantiate the model.
    - CONTINUOUS_BATCHING (`bool`, defaults to `False`) -- Whether the model supports continuous batching or not.
    - ATTENTION_LAYOUT (`str`, defaults to `HSB`) -- Layout to be used for attention computation.

    The NEURONX_CLASS must always be defined in each model configuration.

    Args:
        task (`str`): The task the model should be exported for.
    """

    INPUT_ARGS = ("batch_size", "sequence_length")
    NEURONX_CLASS = None
    ALLOW_FLASH_ATTENTION = False
    CONTINUOUS_BATCHING = False
    ATTENTION_lAYOUT = "HSB"
    FUSE_QKV = True

    def __init__(self, task: str):
        pass

    @property
    def neuronx_class(self):
        return self.NEURONX_CLASS

    @property
    def allow_flash_attention(self):
        return self.ALLOW_FLASH_ATTENTION

    @property
    def continuous_batching(self):
        return self.CONTINUOUS_BATCHING

    @property
    def attention_layout(self):
        return self.ATTENTION_lAYOUT

    @property
    def fuse_qkv(self):
        return self.FUSE_QKV

    def get_neuron_config(
        self,
        checkpoint_id: str,
        checkpoint_revision: str,
        batch_size: int,
        sequence_length: int,
        auto_cast_type: str,
        tensor_parallel_size: int,
    ):
        if issubclass(self.neuronx_class, NeuronHloDecoderModel):
            return HloNeuronConfig(
                checkpoint_id=checkpoint_id,
                checkpoint_revision=checkpoint_revision,
                batch_size=batch_size,
                sequence_length=sequence_length,
                tp_degree=tensor_parallel_size,
                auto_cast_type=auto_cast_type,
                attention_layout=self.attention_layout,
                fuse_qkv=self.fuse_qkv,
                continuous_batching=(batch_size > 1 and self.continuous_batching),
                allow_flash_attention=self.allow_flash_attention,
            )
        return None


@register_in_tasks_manager("llama", "text-generation")
class LLamaNeuronConfig(NeuronDecoderExportConfig):
    NEURONX_CLASS = LlamaHloModel
    ALLOW_FLASH_ATTENTION = True
    CONTINUOUS_BATCHING = True
    ATTENTION_lAYOUT = "BSH"


@register_in_tasks_manager("qwen2", "text-generation")
class Qwen2NeuronConfig(NeuronDecoderExportConfig):
    NEURONX_CLASS = Qwen2HloModel
    ALLOW_FLASH_ATTENTION = True
    CONTINUOUS_BATCHING = True
    FUSE_QKV = False


@register_in_tasks_manager("granite", "text-generation")
class GraniteNeuronConfig(NeuronDecoderExportConfig):
    NEURONX_CLASS = GraniteHloModel
    ALLOW_FLASH_ATTENTION = True
    CONTINUOUS_BATCHING = True


@register_in_tasks_manager("phi3", "text-generation")
class Phi3NeuronConfig(NeuronDecoderExportConfig):
    NEURONX_CLASS = Phi3HloModel
    CONTINUOUS_BATCHING = True
