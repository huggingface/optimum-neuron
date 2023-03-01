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
"""Model specific Neuron configurations."""


from typing import List

from ...utils.normalized_config import NormalizedConfigManager
from .config import TextEncoderNeuronConfig


class BertNeuronConfig(TextEncoderNeuronConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfigManager.get_normalized_config_class("bert")
    ATOL_FOR_VALIDATION = 1e-2

    @property
    def inputs(self) -> List[str]:
        return ["input_ids", "attention_mask", "token_type_ids"]


class AlbertNeuronConfig(BertNeuronConfig):
    pass


# class ConvBertNeuronConfig(BertNeuronConfig):
#     pass


class ElectraNeuronConfig(BertNeuronConfig):
    pass


class FlaubertNeuronConfig(BertNeuronConfig):
    ATOL_FOR_VALIDATION = 1e-1


class MobileBertNeuronConfig(BertNeuronConfig):
    pass


class RoFormerNeuronConfig(BertNeuronConfig):
    pass


class XLMNeuronConfig(BertNeuronConfig):
    ATOL_FOR_VALIDATION = 1e-1


class DistilBertNeuronConfig(BertNeuronConfig):
    ATOL_FOR_VALIDATION = 1e-1

    @property
    def inputs(self) -> List[str]:
        return ["input_ids", "attention_mask"]


class CamembertNeuronConfig(DistilBertNeuronConfig):
    pass


class MPNetNeuronConfig(DistilBertNeuronConfig):
    pass


class RobertaNeuronConfig(DistilBertNeuronConfig):
    pass


class XLMRobertaNeuronConfig(DistilBertNeuronConfig):
    pass


class DebertaNeuronConfig(BertNeuronConfig):
    @property
    def inputs(self) -> List[str]:
        common_inputs = super().inputs
        if self._config.type_vocab_size == 0:
            # We remove token type ids.
            common_inputs.pop(-1)
        return common_inputs


class DebertaV2NeuronConfig(DebertaNeuronConfig):
    pass
