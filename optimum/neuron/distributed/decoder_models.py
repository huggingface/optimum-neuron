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

from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoSelfAttention

from .parallel_layers import ParallelSelfAttention
from .base import Parallelizer

if TYPE_CHECKING:
    from transformers import PreTrainedModel


class GPTNeoParallelSelfAttention(ParallelSelfAttention, GPTNeoSelfAttention):
    QUERIES_NAME = "q_proj"
    KEYS_NAME = "k_proj"
    VALUES_NAME = "v_proj"
    OUTPUT_PROJECTION_NAME = "out_proj"
    ALL_HEAD_SIZE_NAME = "embed_dim"



class GPTNeoParallelizer(Parallelizer):
    @classmethod
    def parallelize(cls, model: "PreTrainedModel") -> "PreTrainedModel":
        for block in model.transformer.h:
            block.attn.attention = GPTNeoParallelSelfAttention(model.config)
        return model

