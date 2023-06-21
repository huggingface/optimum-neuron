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
"""Classes related to parallel versions of common blocks in Transformers models."""

from typing import TYPE_CHECKING, Optional, Type

from transformers.models.bert.modeling_bert import BertSelfAttention, BertSelfOutput

from ...utils import NormalizedConfigManager
from ..utils import is_neuronx_distributed_available
from .base import Parallelizer
from .utils import linear_to_parallel_linear


if is_neuronx_distributed_available():
    from neuronx_distributed.parallel_layers import parallel_state

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel


class ParallelSelfAttention:
    QUERIES_NAME = "query"
    KEYS_NAME = "key"
    VALUES_NAME = "value"
    OUTPUT_PROJECTION_NAME: Optional[str] = None
    # TODO: add this in NormalizedConfig
    ALL_HEAD_SIZE_NAME = "all_head_size"

    def __init__(self, config: "PretrainedConfig", position_embedding_type: Optional[Type] = None):
        super().__init__(config, position_embedding_type)
        self.normalized_config = NormalizedConfigManager.get_normalized_config_class(config.model_type)(config)
        all_head_size = getattr(self, self.ALL_HEAD_SIZE_NAME)
        for name in [self.QUERIES_NAME, self.KEYS_NAME, self.VALUES_NAME]:
            setattr(
                self,
                name,
                linear_to_parallel_linear(getattr(self, name), "column", gather_output=False),
            )
        if self.OUTPUT_PROJECTION_NAME is not None:
            setattr(
                self,
                self.OUTPUT_PROJECTION_NAME,
                linear_to_parallel_linear(getattr(self, self.OUTPUT_PROJECTION_NAME), "row", input_is_parallel=True),
            )
        num_attention_heads_name = self.normalized_config.NUM_ATTENTION_HEADS
        setattr(
            self,
            num_attention_heads_name,
            self.normalized_config.num_attention_heads // parallel_state.get_tensor_model_parallel_size(),
        )
        setattr(
            self,
            self.ALL_HEAD_SIZE_NAME,
            all_head_size // parallel_state.get_tensor_model_parallel_size(),
        )


class ParallelSelfOutput:
    OUTPUT_PROJECTION_NAME = "dense"

    def __init__(self, config: "PretrainedConfig"):
        super().__init__(config)
        self.normalized_config = NormalizedConfigManager.get_normalized_config_class(config.model_type)(config)
        setattr(
            self,
            self.OUTPUT_PROJECTION_NAME,
            linear_to_parallel_linear(getattr(self, self.OUTPUT_PROJECTION_NAME), "row", input_is_parallel=True),
        )

