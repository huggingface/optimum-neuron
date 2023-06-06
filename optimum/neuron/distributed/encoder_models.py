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


if is_neuronx_distributed_available():
    from neuronx_distributed.parallel_layers import parallel_state

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel


class ParallelSelfAttention:
    QUERY_NAME = "query"
    KEY_NAME = "key"
    VALUE_NAME = "value"
    ALL_HEAD_SIZE_NAME = "all_head_size"

    def __init__(self, config: "PretrainedConfig", position_embedding_type: Optional[Type] = None):
        super().__init__(config, position_embedding_type)
        self.normalized_config = NormalizedConfigManager.get_normalized_config_class(config.model_type)(config)
        all_head_size = getattr(self, self.ALL_HEAD_SIZE_NAME)
        for name in [self.QUERY_NAME, self.KEY_NAME, self.VALUE_NAME]:
            setattr(
                self,
                name,
                linear_to_parallel_linear(getattr(self, name), "column", gather_output=False),
                # layers.ColumnParallelLinear(self.normalized_config.hidden_size, all_head_size, gather_output=False),
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
            all_head_size / parallel_state.get_tensor_model_parallel_size(),
        )


class ParallelSelfOutput:
    DENSE_NAME = "dense"

    def __init__(self, config: "PretrainedConfig"):
        super().__init__(config)
        self.normalized_config = NormalizedConfigManager.get_normalized_config_class(config.model_type)(config)
        setattr(
            self,
            self.DENSE_NAME,
            linear_to_parallel_linear(getattr(self, self.DENSE_NAME), "row", input_is_parallel=True),
            # layers.RowParallelLinear(config.hidden_size, config.hidden_size, input_is_parallel=True),
        )


class BertParallelSelfAttention(ParallelSelfAttention, BertSelfAttention):
    pass


class BertParallelSelfOutput(ParallelSelfOutput, BertSelfOutput):
    pass


class BertParallelizer(Parallelizer):
    @classmethod
    def parallelize(cls, model: "PreTrainedModel") -> "PreTrainedModel":
        device = next(model.parameters()).device
        for layer in model.bert.encoder.layer:
            layer.attention.self = BertParallelSelfAttention(model.config)
            layer.attention.output = BertParallelSelfOutput(model.config)
        model.to(device)
        # neuronx_distributed.parallel_layers.load(model_path.as_posix(), model, sharded=False)

        devices = [f"{name} => {p.device}" for name, p in model.named_parameters()]
        print("\n".join(devices))
        return model
