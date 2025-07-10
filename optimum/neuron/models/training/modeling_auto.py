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

from typing import Union

from transformers import AutoConfig

from ..auto_model import get_neuron_model_class
from .config import TrainingNeuronConfig


class _BaseNeuronModelClass:
    """
    Base neuron auto model class for training with Neuron custom implementations.

    This class provides a unified interface to load custom Neuron training models
    similar to transformers.AutoModel but specifically for training with Neuron devices.
    """

    _task_name: str

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, bytes], trn_config: TrainingNeuronConfig, *model_args, **kwargs
    ):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model_type = config.model_type
        neuron_model_class = get_neuron_model_class(model_type, cls._task_name, "training")
        return neuron_model_class.from_pretrained(pretrained_model_name_or_path, trn_config, *model_args, **kwargs)

    @classmethod
    def from_config(cls, config, trn_config: TrainingNeuronConfig):
        model_type = config.model_type
        neuron_model_class = get_neuron_model_class(model_type, cls._task_name, "training")
        return neuron_model_class(config, trn_config)


class NeuronModel(_BaseNeuronModelClass):
    _task_name = "model"


class NeuronModelForCausalLM(_BaseNeuronModelClass):
    _task_name = "text-generation"
