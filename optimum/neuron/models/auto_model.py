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
"""Used to register and retrieve Neuron model classes for different tasks and modes."""

from typing import Literal


_REGISTERED_NEURON_MODELS = {}


def register_neuron_model(model_type: str, task: str, mode: Literal["inference", "training"]):
    def wrapper(cls):
        _REGISTERED_NEURON_MODELS[(model_type, task, mode)] = cls
        return cls

    return wrapper


def has_neuron_model_class(model_type: str, task: str, mode: Literal["inference", "training"]) -> bool:
    """Check if a neuron modeling class is registered for the specified model configuration."""
    from .inference import auto_models  # noqa F401

    return (model_type, task, mode) in _REGISTERED_NEURON_MODELS


def get_neuron_model_class(model_type: str, task: str, mode: Literal["inference", "training"]):
    """Get the neuron modeling class for the specified model configuration."""
    if not has_neuron_model_class(model_type, task, mode):
        model_types = [tp for tp, tsk, md in _REGISTERED_NEURON_MODELS.keys() if tsk == task and md == mode]
        raise ValueError(
            f"Model type {model_type} is not supported for task {task} in neuron in {mode} mode. Supported types are: {model_types}."
        )
    return _REGISTERED_NEURON_MODELS[(model_type, task, mode)]
