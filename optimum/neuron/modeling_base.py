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

from typing import TYPE_CHECKING

import torch

from optimum.modeling_base import OptimizedModel


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel


class NeuronModel(OptimizedModel):
    def __init__(self, model: "PreTrainedModel", config: "PretrainedConfig"):
        super().__init__(model, config)
        if hasattr(model, "device"):
            self.device = model.device
        else:
            self.device = torch.device("cpu")

    def to(self, device: str | torch.device):
        if not isinstance(device, torch.device):
            device = torch.device(device)
        if device.type != self.device.type:
            raise ValueError(f"Neuron models cannot be moved to {device.type}.")
