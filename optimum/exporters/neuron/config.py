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
"""
Common Neuron configuration classes that handle most of the features for building model specific
configurations.
"""
import importlib

from ...exporters.base import ExportConfig
from ...neuron.utils import is_transformers_neuronx_available
from ...utils import (
    DummyBboxInputGenerator,
    DummyTextInputGenerator,
    DummyVisionInputGenerator,
    logging,
)
from .base import NeuronConfig


logger = logging.get_logger(__name__)


class TextEncoderNeuronConfig(NeuronConfig):
    """
    Handles encoder-based text architectures.
    """

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator,)
    MANDATORY_AXES = ("batch_size", "sequence_length", ("multiple-choice", "num_choices"))


class VisionNeuronConfig(NeuronConfig):
    """
    Handles vision architectures.
    """

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVisionInputGenerator,)
    MANDATORY_AXES = ("batch_size", "num_channels", "width", "height")


class TextAndVisionNeuronConfig(NeuronConfig):
    """
    Handles multi-modal text and vision architectures.
    """

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, DummyVisionInputGenerator, DummyBboxInputGenerator)


class DecoderNeuronConfig(ExportConfig):
    """
    Handles decoder architectures.
    """

    NEURONX_ARGS = {"batch_size": 1, "n_positions": 128, "tp_degree": 2, "amp": "f32"}
    NEURONX_MODULE = None
    NEURONX_CLASS = None

    def __init__(self, task, **kwargs):
        if not is_transformers_neuronx_available():
            raise ModuleNotFoundError("The transformers-neuronx package is required.")
        module = importlib.import_module(self.NEURONX_MODULE)
        self._neuronx_class = getattr(module, self.NEURONX_CLASS, None)
        if self._neuronx_class is None:
            raise ImportError(f"{self.NEURONX_CLASS} not found in {self.NEURONX_MODULE}. Please check versions.")

    def split_kwargs(self, **kwargs):
        """Split between kwargs that need to be passed when loading the transformers model
        and those that need to be passed to the neuron optimizer.
        """
        model_kwargs = kwargs
        neuron_kwargs = {}
        for arg, default in self.NEURONX_ARGS.items():
            if arg in model_kwargs:
                neuron_kwargs[arg] = model_kwargs[arg]
                model_kwargs.pop(arg)
            else:
                neuron_kwargs[arg] = default
        return model_kwargs, neuron_kwargs

    @property
    def neuronx_class(self):
        return self._neuronx_class
