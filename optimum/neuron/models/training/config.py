# coding=utf-8
# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
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
"""Multi processing configuration extended for Neuron"""

from optimum.neuron.configuration_utils import NeuronConfig, register_neuron_config


@register_neuron_config()
class TrainingNeuronConfig(NeuronConfig):
    r"""
    Neuron configurations for extra features and performance optimizations.

    Args:
        sequence_parallel_enabled (`bool`, *optional*, defaults to `False`):
            Whether to enable sequence parallelism. If `True`, the model will be trained with sequence parallelism.
        tensor_parallel_size (`int`, *optional*, defaults to `1`):
            The size of the tensor parallelism. This is the number of devices used for tensor parallelism.
        kv_size_multiplier (`int`, *optional*, defaults to `1`):
            The size multiplier for key-value pairs. This is used to control the size of the key-value pairs in the
            model.
        gradient_checkpointing (`bool`, *optional*, defaults to `False`):
            Whether to enable gradient checkpointing. If `True`, the model will use gradient checkpointing to save
            memory.
    """

    def __init__(
        self,
        sequence_parallel_enabled: bool = False,
        tensor_parallel_size: int = 1,
        kv_size_multiplier: int = 1,
        gradient_checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sequence_parallel_enabled = sequence_parallel_enabled
        self.tensor_parallel_size = tensor_parallel_size
        self.kv_size_multiplier = kv_size_multiplier
        self.gradient_checkpointing = gradient_checkpointing
