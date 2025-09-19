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

import enum
import os
from dataclasses import dataclass

from ....utils import logging
from ...utils.torch_xla_and_neuronx_initialization import set_neuron_cc_flag


logger = logging.get_logger(__name__)


class MixedPrecisionMode(str, enum.Enum):
    NO = "NO"
    FULL_BF16 = "FULL_BF16"
    AUTOCAST_BF16 = "AUTOCAST_BF16"


@dataclass
class MixedPrecisionConfig:
    mode: MixedPrecisionMode | str
    stochastic_rounding: bool = True
    optimizer_use_master_weights: bool = False
    optimizer_use_fp32_grad_acc: bool = False
    optimizer_save_master_weights_in_ckpt: bool = False

    def __post_init__(self):
        if isinstance(self.mode, str):
            self.mode = MixedPrecisionMode(self.mode)

        if self.mode is MixedPrecisionMode.FULL_BF16:
            if not self.stochastic_rounding and not self.optimizer_use_master_weights:
                logger.warning(
                    "In full bf16 mode, it is recommended to enable stochastic rounding or use master weights for the "
                    "optimizer."
                )
        elif self.mode is MixedPrecisionMode.AUTOCAST_BF16:
            set_neuron_cc_flag("--auto-cast", "none")
        else:
            self.stochastic_rounding = False
            self.optimizer_use_master_weights = False
            self.optimizer_use_fp32_grad_acc = False
            self.optimizer_save_master_weights_in_ckpt = False

        if not self.optimizer_use_master_weights and self.optimizer_use_fp32_grad_acc:
            raise ValueError("optimizer_use_fp32_grad_acc requires optimizer_use_master_weights to be True.")

        if not self.optimizer_use_master_weights and self.optimizer_save_master_weights_in_ckpt:
            raise ValueError("optimizer_save_master_weights_in_ckpt requires optimizer_use_master_weights to be True.")

        if self.stochastic_rounding:
            os.environ["NEURON_RT_STOCHASTIC_ROUNDING_EN"] = "1"
        else:
            os.environ["NEURON_RT_STOCHASTIC_ROUNDING_EN"] = "0"
