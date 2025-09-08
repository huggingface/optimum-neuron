# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Custom dataclasses for Neuron."""

import os
from dataclasses import dataclass, field
from typing import Literal
import enum

from ..utils.torch_xla_and_neuronx_initialization import set_neuron_cc_flag


class NeuronDistributedType(str, enum.Enum):
    """
    Represents a type of distributed environment specific to Neuron.

    Values:
        - **MODEL_PARALLELISM** -- Tensor and Pipeline Parallelisms using `torch_xla` and `neuronx_distributed`.
    """

    MODEL_PARALLELISM = "MODEL_PARALLELISM"


class MixedPrecisionMode(str, enum.Enum):
    NO = "NO"
    FULL_BF16 = "FULL_BF16"
    AUTOCAST_BF16 = "AUTOCAST_BF16"
    STANDARD = "STANDARD"

@dataclass
class MixedPrecisionConfig:
    mode: MixedPrecisionMode | str
    stochastic_rounding: bool | Literal["auto"] = "auto"
    optimizer_use_master_weights: bool | Literal["auto"] = "auto"
    optimizer_use_fp32_grad_acc: bool | Literal["auto"] = "auto"

    def __post_init__(self):
        if isinstance(self.mode, str):
            self.mode = MixedPrecisionMode(self.mode)

        if self.mode is MixedPrecisionMode.FULL_BF16:
            if self.stochastic_rounding == "auto":
                self.stochastic_rounding = True

            if self.optimizer_use_master_weights == "auto":
                self.optimizer_use_master_weights = self.stochastic_rounding is False

            if self.optimizer_use_master_weights:
                # In full bf16 mode, stochastic rounding must be enabled or we need to use master weights for the 
                # optimizer. It is not supported for now.
                raise ValueError("In full bf16 mode, using master weights is not supported.")
        elif self.mode is MixedPrecisionMode.AUTOCAST_BF16:
            set_neuron_cc_flag("--auto-cast", "none")
            
            if self.stochastic_rounding == "auto":
                self.stochastic_rounding = False

            if self.optimizer_use_master_weights == "auto":
                self.optimizer_use_master_weights = False

            if self.optimizer_use_master_weights: 
                raise ValueError("Using master weights is not supported in autocast bf16 mode.")
        elif self.mode is MixedPrecisionMode.STANDARD: 
            if self.stochastic_rounding == "auto":
                self.stochastic_rounding = False

            if self.optimizer_use_master_weights == "auto":
                self.optimizer_use_master_weights = True

            if self.optimizer_use_fp32_grad_acc == "auto":
                self.optimizer_use_fp32_grad_acc = True
        else: # NO
            self.stochastic_rounding = False
            self.optimizer_use_master_weights = False
            self.optimizer_use_fp32_grad_acc = False

        if self.stochastic_rounding:
            os.environ["NEURON_RT_STOCHASTIC_ROUNDING_EN"] = "1"
        else:
            os.environ["NEURON_RT_STOCHASTIC_ROUNDING_EN"] = "0"


    




