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

from ..utils.require_utils import requires_neuronx_distributed
from .base import Parallelizer
from .parallel_layers import ParallelCrossEntropy, ParallelEmbedding, ParallelSelfAttention, ParallelSelfOutput


if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel


class BertParallelEmbedding(ParallelEmbedding):
    EMBEDDING_NAME = "bert.embeddings.word_embeddings"
    LM_HEAD_NAME = {
        "BertForPreTraining": "cls.predictions.decoder",
        "BertLMHeadModel": "cls.predictions.decoder",
        "BertForMaskedLM": "cls.predictions.decoder",
    }

    @classmethod
    @requires_neuronx_distributed
    def transform(
        cls,
        model: "PreTrainedModel",
        layer: "torch.nn.Module",
        orig_to_parallel: Optional[Dict[int, "torch.nn.Parameter"]] = None,
        device: Optional["torch.device"] = None,
    ) -> "torch.nn.Module":
        layer = super().transform(model, layer, orig_to_parallel=orig_to_parallel, device=device)
        from transformers.models.bert.modeling_bert import BertLMPredictionHead

        for mod in layer.modules():
            if isinstance(mod, BertLMPredictionHead):
                mod.bias = mod.decoder.bias
        return layer


class BertParallelSelfAttention(ParallelSelfAttention):
    ALL_HEAD_SIZE_NAME = "all_head_size"


class BertParallelSelfOutput(ParallelSelfOutput):
    pass


class BertParallelCrossEntropy(ParallelCrossEntropy):
    LAST_LINEAR_PROJECTION_NAME = {
        "BertForPreTraining": "cls.predictions.decoder",
        "BertLMHeadModel": "cls.predictions.decoder",
        "BertForMaskedLM": "cls.predictions.decoder",
    }


class BertParallelizer(Parallelizer):
    @classmethod
    def _parallelize(
        cls,
        model: "PreTrainedModel",
        orig_to_parallel: Optional[Dict[int, "torch.nn.Parameter"]],
        device: Optional["torch.device"] = None,
        parallelize_embeddings: bool = True,
    ) -> "PreTrainedModel":
        if parallelize_embeddings:
            model = BertParallelEmbedding.transform(model, model, device=device)
        for layer in model.bert.encoder.layer:
            layer.attention.self = BertParallelSelfAttention.transform(
                model,
                layer.attention.self,
                orig_to_parallel=orig_to_parallel,
                device=device,
            )
            layer.attention.output = BertParallelSelfOutput.transform(
                model,
                layer.attention.output,
                orig_to_parallel=orig_to_parallel,
                device=device,
            )
        # Valid because we currently parallelize the cross-entropy loss only for language-modeling tasks where the
        # embeddings and the LM head are tied.
        if parallelize_embeddings:
            model = BertParallelCrossEntropy.transform(model, model, device=device)
        return model


class RobertaParallelEmbedding(ParallelEmbedding):
    EMBEDDING_NAME = "roberta.embeddings.word_embeddings"
    LM_HEAD_NAME = {
        "RobertaForCausalLM": "lm_head.decoder",
        "RobertaForMaskedLM": "lm_head.decoder",
    }


class RobertaParallelSelfAttention(BertParallelSelfAttention):
    pass


class RobertaParallelSelfOutput(BertParallelSelfOutput):
    pass


class RobertaParallelCrossEntropy(ParallelCrossEntropy):
    LAST_LINEAR_PROJECTION_NAME = {
        "RobertaForCausalLM": "lm_head.decoder",
        "RobertaForMaskedLM": "lm_head.decoder",
    }


class RobertaParallelizer(Parallelizer):
    @classmethod
    def _parallelize(
        cls,
        model: "PreTrainedModel",
        orig_to_parallel: Optional[Dict[int, "torch.nn.Parameter"]],
        device: Optional["torch.device"] = None,
        parallelize_embeddings: bool = True,
    ) -> "PreTrainedModel":
        if parallelize_embeddings:
            model = RobertaParallelEmbedding.transform(model, model, device=device)
        for layer in model.roberta.encoder.layer:
            layer.attention.self = RobertaParallelSelfAttention.transform(
                model,
                layer.attention.self,
                orig_to_parallel=orig_to_parallel,
                device=device,
            )
            layer.attention.output = RobertaParallelSelfOutput.transform(
                model,
                layer.attention.output,
                orig_to_parallel=orig_to_parallel,
                device=device,
            )
        # Valid because we currently parallelize the cross-entropy loss only for language-modeling tasks where the
        # embeddings and the LM head are tied.
        if parallelize_embeddings:
            model = RobertaParallelCrossEntropy.transform(model, model, device=device)
        return model
