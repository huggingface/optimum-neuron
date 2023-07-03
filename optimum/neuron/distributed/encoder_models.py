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

from typing import TYPE_CHECKING, Dict, Optional

from ..utils import is_neuronx_distributed_available
from .base import Parallelizer
from .parallel_layers import ParallelSelfAttention, ParallelSelfOutput
from .utils import embedding_to_parallel_embedding


if is_neuronx_distributed_available():
    pass

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel


class BertParallelSelfAttention(ParallelSelfAttention):
    ALL_HEAD_SIZE_NAME = "all_head_size"


class BertParallelSelfOutput(ParallelSelfOutput):
    pass


class BertParallelizer(Parallelizer):
    @classmethod
    def parallelize(
        cls, model: "PreTrainedModel", orig_to_parallel: Optional[Dict[int, "torch.nn.Parameter"]] = None
    ) -> "PreTrainedModel":
        model.bert.embeddings.word_embeddings = embedding_to_parallel_embedding(model.bert.embeddings.word_embeddings)
        for layer in model.bert.encoder.layer:
            layer.attention.self = BertParallelSelfAttention.transform(
                layer.attention.self, model.config, orig_to_parallel=orig_to_parallel
            )
            layer.attention.output = BertParallelSelfOutput.transform(
                layer.attention.output, model.config, orig_to_parallel=orig_to_parallel
            )
        return model


class RobertaParallelSelfAttention(BertParallelSelfAttention):
    pass


class RobertaParallelSelfOutput(BertParallelSelfOutput):
    pass


class RobertaParallelizer(Parallelizer):
    @classmethod
    def parallelize(
        cls, model: "PreTrainedModel", orig_to_parallel: Optional[Dict[int, "torch.nn.Parameter"]] = None
    ) -> "PreTrainedModel":
        for layer in model.roberta.encoder.layer:
            layer.attention.self = RobertaParallelSelfAttention.transform(
                layer.attention.self, model.config, orig_to_parallel=orig_to_parallel
            )
            layer.attention.output = RobertaParallelSelfOutput.transform(
                layer.attention.output, model.config, orig_to_parallel=orig_to_parallel
            )
        return model
