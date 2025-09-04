# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from abc import ABC
from dataclasses import dataclass

import torch


class PreTrainedModel:
    # A fake PreTrainedModel class to be inserted in NeuronModel class hierarchy
    # to fool transformers pipeline framework identification algorithm
    pass


@dataclass
class NeuronModel(PreTrainedModel, ABC):
    """Base class for all Neuron models."""

    device: torch.device = torch.device("cpu")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def to(self, device: str | torch.device):
        if not isinstance(device, torch.device):
            device = torch.device(device)
        if device.type != self.device.type:
            raise ValueError(f"Neuron models cannot be moved to {device.type}.")

    def forward(self, *args, **kwargs):
        raise NotImplementedError("NeuronModel is an abstract class. Please use a subclass.")
