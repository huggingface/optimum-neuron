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

import importlib

from transformers_neuronx import ContinuousBatchingConfig
from transformers_neuronx import NeuronConfig as TnxNeuronConfig

from optimum.exporters.tasks import TasksManager

from ....neuron.backends.hlo.config import NeuronConfig
from ....neuron.backends.hlo.decoder import NeuronHloDecoderModel
from ....neuron.models.granite.model import GraniteForSampling
from ....neuron.models.llama.model import LlamaHloModel
from ....neuron.models.phi3.model import Phi3HloModel
from ....neuron.models.qwen2.model import Qwen2ForSampling
from ..base import NeuronExportConfig


register_in_tasks_manager = TasksManager.create_register("neuron")


class NeuronDecoderExportConfig(NeuronExportConfig):
    """
    Base class for configuring the export of Neuron Decoder models

    Class attributes:

    - INPUT_ARGS (`Tuple[Union[str, Tuple[Union[str, Tuple[str]]]]]`) -- A tuple where each element is either:
        - An argument  name, for instance "batch_size" or "sequence_length", that indicates that the argument can
        be passed to export the model,
    - NEURONX_CLASS (`str`) -- the name of the transformers-neuronx class to instantiate for the model.
    It is a full class name defined relatively to the transformers-neuronx module, e.g. `gpt2.model.GPT2ForSampling`
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
        if isinstance(self.NEURONX_CLASS, type):
            self._neuronx_class = self.NEURONX_CLASS
        else:
            module_name, class_name = self.NEURONX_CLASS.rsplit(".", maxsplit=1)
            module = importlib.import_module(f"transformers_neuronx.{module_name}")
            self._neuronx_class = getattr(module, class_name, None)
            if self._neuronx_class is None:
                raise ImportError(
                    f"{class_name} not found in {module_name}. Please check transformers-neuronx version."
                )

    @property
    def neuronx_class(self):
        return self._neuronx_class

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

    def get_export_kwargs(self, batch_size: int, sequence_length: int, auto_cast_type: str, tensor_parallel_size: int):
        base_kwargs = {
            "batch_size": batch_size,
            "n_positions": sequence_length,
            "tp_degree": tensor_parallel_size,
            # transformers-neuronx uses f32/f16 instead of fp32/fp16
            "amp": auto_cast_type.replace("p", ""),
        }
        neuron_kwargs = {
            "attention_layout": self.attention_layout,
            "fuse_qkv": self.fuse_qkv,
        }
        # Continuous batching is always enabled for models that support it because static batching
        # is broken for these models:  see https://github.com/aws-neuron/transformers-neuronx/issues/79
        continuous_batching = batch_size > 1 and self.continuous_batching
        export_kwargs = {}
        if issubclass(self.neuronx_class, NeuronHloDecoderModel):
            # For new models, all export kwargs are integrated into NeuronConfig
            neuron_kwargs.update(base_kwargs)
            neuron_kwargs["allow_flash_attention"] = self.allow_flash_attention
            neuron_kwargs["continuous_batching"] = continuous_batching
            export_kwargs["neuron_config"] = NeuronConfig(**neuron_kwargs)
        else:
            # For legacy models, base kwargs are passed individually
            export_kwargs.update(base_kwargs)
            if continuous_batching:
                neuron_kwargs["continuous_batching"] = ContinuousBatchingConfig(
                    batch_size_for_shared_caches=batch_size
                )
                export_kwargs["n_positions"] = [sequence_length]
                export_kwargs["context_length_estimate"] = [sequence_length]
            export_kwargs["neuron_config"] = TnxNeuronConfig(**neuron_kwargs)

        return export_kwargs


@register_in_tasks_manager("gpt2", "text-generation")
class GPT2NeuronConfig(NeuronDecoderExportConfig):
    NEURONX_CLASS = "gpt2.model.GPT2ForSampling"


@register_in_tasks_manager("llama", "text-generation")
class LLamaNeuronConfig(NeuronDecoderExportConfig):
    NEURONX_CLASS = LlamaHloModel
    ALLOW_FLASH_ATTENTION = True
    CONTINUOUS_BATCHING = True
    ATTENTION_lAYOUT = "BSH"


@register_in_tasks_manager("opt", "text-generation")
class OPTNeuronConfig(NeuronDecoderExportConfig):
    NEURONX_CLASS = "opt.model.OPTForSampling"


@register_in_tasks_manager("bloom", "text-generation")
class BloomNeuronConfig(NeuronDecoderExportConfig):
    NEURONX_CLASS = "bloom.model.BloomForSampling"


@register_in_tasks_manager("mistral", "text-generation")
class MistralNeuronConfig(NeuronDecoderExportConfig):
    NEURONX_CLASS = "mistral.model.MistralForSampling"
    CONTINUOUS_BATCHING = True


@register_in_tasks_manager("mixtral", "text-generation")
class MixtralNeuronConfig(NeuronDecoderExportConfig):
    NEURONX_CLASS = "mixtral.model.MixtralForSampling"
    CONTINUOUS_BATCHING = False


@register_in_tasks_manager("qwen2", "text-generation")
class Qwen2NeuronConfig(NeuronDecoderExportConfig):
    NEURONX_CLASS = Qwen2ForSampling
    ALLOW_FLASH_ATTENTION = True
    CONTINUOUS_BATCHING = True
    FUSE_QKV = False


@register_in_tasks_manager("granite", "text-generation")
class GraniteNeuronConfig(NeuronDecoderExportConfig):
    NEURONX_CLASS = GraniteForSampling
    ALLOW_FLASH_ATTENTION = True
    CONTINUOUS_BATCHING = True


@register_in_tasks_manager("phi3", "text-generation")
class Phi3NeuronConfig(NeuronDecoderExportConfig):
    NEURONX_CLASS = Phi3HloModel
    CONTINUOUS_BATCHING = True
