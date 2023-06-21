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
"""Classes related to `neuronx-distributed` to perform parallelism."""

from typing import TYPE_CHECKING, Optional, Type

from transformers.models.bert.modeling_bert import BertSelfAttention, BertSelfOutput

from ...utils import NormalizedConfigManager
from ..utils import is_neuronx_distributed_available
from .base import Parallelizer
from .utils import linear_to_parallel_linear
from .parallel_layers import ParallelSelfAttention, ParallelSelfOutput


if is_neuronx_distributed_available():
    from neuronx_distributed.parallel_layers import parallel_state

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel



class BertParallelSelfAttention(ParallelSelfAttention, BertSelfAttention):
    pass


class BertParallelSelfOutput(ParallelSelfOutput, BertSelfOutput):
    pass


class BertParallelizer(Parallelizer):
    @classmethod
    def parallelize(cls, model: "PreTrainedModel") -> "PreTrainedModel":
        for layer in model.bert.encoder.layer:
            layer.attention.self = BertParallelSelfAttention(model.config)
            layer.attention.output = BertParallelSelfOutput(model.config)
        return model

class RobertaParallelSelfAttention(BertParallelSelfAttention):
    pass


class RobertaParallelSelfOutput(BertParallelSelfOutput):
    pass

class RobertaParallelizer(Parallelizer):
    @classmethod
    def parallelize(cls, model: "PreTrainedModel") -> "PreTrainedModel":
        for layer in model.roberta.encoder.layer:
            layer.attention.self = RobertaParallelSelfAttention(model.config)
            layer.attention.output = RobertaParallelSelfOutput(model.config)
        return model
