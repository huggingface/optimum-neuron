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

from typing import TYPE_CHECKING, Optional

import torch

from ..utils.require_utils import requires_neuronx_distributed
from .base import Parallelizer, SequenceParallelismSpecs
from .parallel_layers import (
    ParallelCrossEntropy,
    ParallelEmbedding,
    ParallelSelfAttention,
    ParallelSelfOutput,
    SequenceCollectiveOpInfo,
)


if TYPE_CHECKING:
    from transformers import PreTrainedModel


def create_sequence_parallel_attention_forward(attention_forward, sequence_parallel_enabled: bool):
    import functools

    @functools.wraps(attention_forward)
    def sequence_parallel_attention_forward(self, *args, **kwargs):
        outputs = attention_forward(*args, **kwargs)
        context_layer = outputs[0]
        if sequence_parallel_enabled:
            # [B, S, hidden_dim] -> [S, B, hidden_dim]
            context_layer = context_layer.transpose(0, 1)
        return (context_layer,) + outputs[1:]

    return sequence_parallel_attention_forward


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
        sequence_parallel_enabled: bool = False,
        device: Optional["torch.device"] = None,
    ) -> "torch.nn.Module":
        layer = super().transform(model, layer, sequence_parallel_enabled=sequence_parallel_enabled, device=device)
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


class BertSequenceParallelismSpecs(SequenceParallelismSpecs):
    SEQUENCE_PARALLEL_LAYERNORM_PATTERNS = [
        "bert.embeddings.LayerNorm",
        "bert.encoder.layer.[0-9]+.attention.output.LayerNorm",
        "bert.encoder.layer.[0-9]+.output.LayerNorm",
    ]
    SEQUENCE_COLLECTIVE_OPS_INFOS = [
        SequenceCollectiveOpInfo("scatter", torch.nn.LayerNorm, "input", "first"),
        SequenceCollectiveOpInfo("gather", "bert.encoder.layer.[0-9]+.output.LayerNorm", "output", "last"),
    ]

    @classmethod
    def patch_for_sequence_parallelism(cls, model: "PreTrainedModel", sequence_parallel_enabled: bool):
        if not sequence_parallel_enabled:
            return

        from transformers.models.bert.modeling_bert import BertSelfAttention

        def transpose_for_scores(self, x: "torch.Tensor") -> "torch.Tensor":
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            x = x.view(new_x_shape)
            if sequence_parallel_enabled:
                # [S, B, num_heads, head_dim] -> [B, num_heads, S, head_dim]
                return x.permute(1, 2, 0, 3)
            return x.permute(0, 2, 1, 3)

        for module in model.modules():
            if isinstance(module, BertSelfAttention):
                module.transpose_for_scores = transpose_for_scores.__get__(module)
                module.forward = create_sequence_parallel_attention_forward(
                    module.forward, sequence_parallel_enabled
                ).__get__(module)


class BertParallelizer(Parallelizer):
    SEQUENCE_PARALLELSIM_SPECS_CLS = BertSequenceParallelismSpecs

    @classmethod
    def _parallelize(
        cls,
        model: "PreTrainedModel",
        device: Optional["torch.device"] = None,
        parallelize_embeddings: bool = True,
        sequence_parallel_enabled: bool = False,
    ) -> "PreTrainedModel":
        if parallelize_embeddings:
            model = BertParallelEmbedding.transform(
                model, model, sequence_parallel_enabled=sequence_parallel_enabled, device=device
            )
        for layer in model.bert.encoder.layer:
            layer.attention.self = BertParallelSelfAttention.transform(
                model,
                layer.attention.self,
                sequence_parallel_enabled=sequence_parallel_enabled,
                device=device,
            )
            layer.attention.output = BertParallelSelfOutput.transform(
                model,
                layer.attention.output,
                sequence_parallel_enabled=sequence_parallel_enabled,
                device=device,
            )
        # Valid because we currently parallelize the cross-entropy loss only for language-modeling tasks where the
        # embeddings and the LM head are tied.
        if parallelize_embeddings:
            BertParallelEmbedding.overwrite_vocab_size_value_for_cross_entropy_computation(model)
            model = BertParallelCrossEntropy.transform(
                model, model, sequence_parallel_enabled=sequence_parallel_enabled, device=device
            )
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


class RobertaSequenceParallelismSpecs(SequenceParallelismSpecs):
    SEQUENCE_PARALLEL_LAYERNORM_PATTERNS = [
        "roberta.embeddings.LayerNorm",
        "roberta.encoder.layer.[0-9]+.attention.output.LayerNorm",
        "roberta.encoder.layer.[0-9]+.output.LayerNorm",
    ]
    SEQUENCE_COLLECTIVE_OPS_INFOS = [
        SequenceCollectiveOpInfo("scatter", torch.nn.LayerNorm, "input", "first"),
        SequenceCollectiveOpInfo("gather", "roberta.encoder.layer.[0-9]+.output.LayerNorm", "output", "last"),
    ]

    @classmethod
    def patch_for_sequence_parallelism(cls, model: "PreTrainedModel", sequence_parallel_enabled: bool):
        if not sequence_parallel_enabled:
            return

        from transformers.models.roberta.modeling_roberta import RobertaSelfAttention

        def transpose_for_scores(self, x: "torch.Tensor") -> "torch.Tensor":
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            x = x.view(new_x_shape)
            if sequence_parallel_enabled:
                # [S, B, num_heads, head_dim] -> [B, num_heads, S, head_dim]
                return x.permute(1, 2, 0, 3)
            return x.permute(0, 2, 1, 3)

        for module in model.modules():
            if isinstance(module, RobertaSelfAttention):
                module.transpose_for_scores = transpose_for_scores.__get__(module)
                module.forward = create_sequence_parallel_attention_forward(
                    module.forward, sequence_parallel_enabled
                ).__get__(module)


class RobertaParallelizer(Parallelizer):
    SEQUENCE_PARALLELSIM_SPECS_CLS = RobertaSequenceParallelismSpecs

    @classmethod
    def _parallelize(
        cls,
        model: "PreTrainedModel",
        device: Optional["torch.device"] = None,
        parallelize_embeddings: bool = True,
        sequence_parallel_enabled: bool = False,
    ) -> "PreTrainedModel":
        if parallelize_embeddings:
            model = RobertaParallelEmbedding.transform(
                model, model, sequence_parallel_enabled=sequence_parallel_enabled, device=device
            )
        for layer in model.roberta.encoder.layer:
            layer.attention.self = RobertaParallelSelfAttention.transform(
                model,
                layer.attention.self,
                sequence_parallel_enabled=sequence_parallel_enabled,
                device=device,
            )
            layer.attention.output = RobertaParallelSelfOutput.transform(
                model,
                layer.attention.output,
                sequence_parallel_enabled=sequence_parallel_enabled,
                device=device,
            )
        # Valid because we currently parallelize the cross-entropy loss only for language-modeling tasks where the
        # embeddings and the LM head are tied.
        if parallelize_embeddings:
            RobertaParallelEmbedding.overwrite_vocab_size_value_for_cross_entropy_computation(model)
            model = RobertaParallelCrossEntropy.transform(
                model, model, sequence_parallel_enabled=sequence_parallel_enabled, device=device
            )
        return model
