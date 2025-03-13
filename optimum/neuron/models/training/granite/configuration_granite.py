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
"""Granite model configuration extended for Neuron"""

from transformers.models.granite import GraniteConfig


class NeuronGraniteConfig(GraniteConfig):
    r"""
    This is an extension to the [`GraniteConfig`] class to add Neuron specific parameters regarding parallelism.
    """

    def __init__(
        self,
        sequence_parallel_enabled: bool = False,
        tensor_parallel_size: int = 1,
        gradient_checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sequence_parallel_enabled = sequence_parallel_enabled
        self.tensor_parallel_size = tensor_parallel_size
        self.gradient_checkpointing = gradient_checkpointing
