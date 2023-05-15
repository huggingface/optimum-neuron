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
from ..tasks import TasksManager
from .config import TextEncoderNeuronConfig


COMMON_TEXT_TASKS = [
    "feature-extraction",
    "fill-mask",
    "multiple-choice",
    "question-answering",
    "text-classification",
    "token-classification",
]
register_in_tasks_manager = TasksManager.create_register("neuron")


@register_in_tasks_manager("bert", *COMMON_TEXT_TASKS)
class BertNeuronConfig(TextEncoderNeuronConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfigManager.get_normalized_config_class("bert")
    ATOL_FOR_VALIDATION = 1e-4

    @property
    def inputs(self) -> List[str]:
        return ["input_ids", "attention_mask", "token_type_ids"]


@register_in_tasks_manager("albert", *COMMON_TEXT_TASKS)
class AlbertNeuronConfig(BertNeuronConfig):
    pass


@register_in_tasks_manager("convbert", *COMMON_TEXT_TASKS)
class ConvBertNeuronConfig(BertNeuronConfig):
    @property
    def outputs(self) -> List[str]:
        if self.task == "feature-extraction":
            return ["last_hidden_state"]
        return self._TASK_TO_COMMON_OUTPUTS[self.task]


@register_in_tasks_manager("electra", *COMMON_TEXT_TASKS)
class ElectraNeuronConfig(ConvBertNeuronConfig):
    pass


@register_in_tasks_manager("flaubert", *COMMON_TEXT_TASKS)
class FlaubertNeuronConfig(ConvBertNeuronConfig):
    pass


@register_in_tasks_manager("mobilebert", *COMMON_TEXT_TASKS)
class MobileBertNeuronConfig(BertNeuronConfig):
    pass


@register_in_tasks_manager("roformer", *COMMON_TEXT_TASKS)
class RoFormerNeuronConfig(ConvBertNeuronConfig):
    pass


@register_in_tasks_manager("xlm", *COMMON_TEXT_TASKS)
class XLMNeuronConfig(ConvBertNeuronConfig):
    pass


@register_in_tasks_manager("distilbert", *COMMON_TEXT_TASKS)
class DistilBertNeuronConfig(BertNeuronConfig):
    ATOL_FOR_VALIDATION = 1e-4

    @property
    def inputs(self) -> List[str]:
        return ["input_ids", "attention_mask"]

    @property
    def outputs(self) -> List[str]:
        if self.task == "feature-extraction":
            return ["last_hidden_state"]
        return self._TASK_TO_COMMON_OUTPUTS[self.task]


@register_in_tasks_manager("camembert", *COMMON_TEXT_TASKS)
class CamembertNeuronConfig(BertNeuronConfig):
    ATOL_FOR_VALIDATION = 1e-4

    @property
    def inputs(self) -> List[str]:
        return ["input_ids", "attention_mask"]


@register_in_tasks_manager("mpnet", *COMMON_TEXT_TASKS)
class MPNetNeuronConfig(CamembertNeuronConfig):
    pass


@register_in_tasks_manager("roberta", *COMMON_TEXT_TASKS)
class RobertaNeuronConfig(CamembertNeuronConfig):
    pass


@register_in_tasks_manager("xlm-roberta", *COMMON_TEXT_TASKS)
class XLMRobertaNeuronConfig(CamembertNeuronConfig):
    pass


# https://github.com/aws-neuron/aws-neuron-sdk/issues/642
# Failed only for INF1: 'XSoftmax'
@register_in_tasks_manager("deberta", *COMMON_TEXT_TASKS)
class DebertaNeuronConfig(BertNeuronConfig):
    @property
    def inputs(self) -> List[str]:
        common_inputs = super().inputs
        if self._config.type_vocab_size == 0:
            # We remove token type ids.
            common_inputs.pop(-1)
        return common_inputs


# https://github.com/aws-neuron/aws-neuron-sdk/issues/642
# Failed only for INF1: 'XSoftmax'
@register_in_tasks_manager("deberta-v2", *COMMON_TEXT_TASKS)
class DebertaV2NeuronConfig(DebertaNeuronConfig):
    pass
