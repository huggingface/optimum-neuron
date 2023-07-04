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

from typing import TYPE_CHECKING

from .base import Parallelizer
from .parallel_layers import ParallelSelfAttention
from .utils import embedding_to_parallel_embedding


if TYPE_CHECKING:
    from transformers import PreTrainedModel


class GPTNeoParallelSelfAttention(ParallelSelfAttention):
    QUERIES_NAME = "q_proj"
    KEYS_NAME = "k_proj"
    VALUES_NAME = "v_proj"
    OUTPUT_PROJECTION_NAME = "out_proj"
    ALL_HEAD_SIZE_NAME = "embed_dim"


class GPTNeoParallelizer(Parallelizer):
    @classmethod
    def parallelize(cls, model: "PreTrainedModel") -> "PreTrainedModel":
        model.transformer.wte = embedding_to_parallel_embedding(model.transformer.wte)
        for block in model.transformer.h:
            block.attn.attention = GPTNeoParallelSelfAttention.transform(block.attn.attention, model.config)
        return model


class LlamaParallelSelfAttention(ParallelSelfAttention):
    QUERIES_NAME = "q_proj"
    KEYS_NAME = "k_proj"
    VALUES_NAME = "v_proj"
    OUTPUT_PROJECTION_NAME = "o_proj"
    NUM_ATTENTION_HEADS_NAME = "num_heads"
    ALL_HEAD_SIZE_NAME = "hidden_size"


class LlamaParallelizer(Parallelizer):
    @classmethod
    def parallelize(cls, model: "PreTrainedModel") -> "PreTrainedModel":
        model.model.embed_tokens, model.lm_head = embedding_to_parallel_embedding(
            model.model.embed_tokens, lm_head_layer=model.lm_head
        )
        for layer in model.model.layers:
            layer.self_attn = LlamaParallelSelfAttention.transform(layer.self_attn, model.config)
        return model
