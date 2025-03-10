# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# Copyright 2025 The HuggingFace Team. All rights reserved.
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
# ==============================================================================
import enum
from typing import Optional

import torch


class Layout(enum.Enum):
    HSB = "HSB"
    BSH = "BSH"
    SBH = "SBH"

    def __eq__(self, value):
        return super().__eq__(Layout(value))


# Group query attention sharding configurations
class GQA(enum.Enum):
    # [Default] Sharding over the heads splits entire (complete) K/V heads
    # onto the NeuronCores where the corresponding Q heads reside. This is
    # similar to traditional MHA except that the Q and K/V heads do not need
    # to be equal.
    #
    # This cannot be enabled when number of K/V heads cannot be evenly split
    # across the NeuronCores according to the tensor parallelism degree.
    SHARD_OVER_HEADS = "shard-over-heads"

    # This transforms a GQA attention mechanism into a traditional MHA mechanism
    # by replicating the K/V heads to evenly match the corresponding Q heads.
    # This consumes more memory than would otherwise be used with other sharding
    # mechanisms but avoids collective communications overheads.
    REPLICATED_HEADS = "replicated-heads"


valid_dtypes = [
    "float32",
    "float16",
    "bfloat16",
]


class NeuronConfig:
    """
    Neuron configurations for extra features and performance optimizations.

    Arguments:
        n_positions: the model maximum number of input tokens
        batch_size: the model input batch_size
        amp: the neuron dtype (one of fp32, fp16, bf16)
        tp_degree: the tensor parallelism degree
        continuous_batching: Enables the model to be used with continuous
            batching using the given configurations.
        attention_layout: Layout to be used for attention computation.
            To be selected from `["HSB", "BSH"]`.
        collectives_layout: Layout to be used for collectives within attention.
            To be selected from `["HSB", "BSH"]`.
        group_query_attention: The sharding configuration to use when the number
            of query attention heads is not equal to the number of key/value
            heads. Neuron attempts to select the best configuration by default.
        all_reduce_dtype: The data type that is used for AllReduce collectives.
            To be selected from `["float32", "float16", "bfloat16"]`.
        fuse_qkv: Fuses the QKV projection into a single matrix multiplication.
        log_softmax_scores: Return log-softmax scores along with logits.
        output_all_logits: Return all logits from each model invocation.
        attn_output_transposed: Transposes the attention output projection weight tensor.
        allow_flash_attention: if possible, use flash attention.
    """

    def __init__(
        self,
        n_positions: int = 1024,
        batch_size: int = 1,
        amp: str = "bf16",
        tp_degree: int = 2,
        *,
        continuous_batching: Optional[bool] = False,
        attention_layout: Layout = Layout.HSB,
        collectives_layout: Layout = Layout.HSB,
        group_query_attention: Optional[GQA] = None,
        all_reduce_dtype: Optional[str] = None,
        fuse_qkv: bool = False,
        log_softmax_scores: bool = False,
        output_all_logits: bool = False,
        attn_output_transposed: bool = False,
        allow_flash_attention: bool = True,
    ):
        self.n_positions = n_positions
        self.batch_size = batch_size
        self.amp = amp
        self.tp_degree = tp_degree
        self.all_reduce_dtype = all_reduce_dtype
        self.fuse_qkv = fuse_qkv
        self.continuous_batching = continuous_batching
        self.attention_layout = attention_layout
        self.collectives_layout = collectives_layout
        self.log_softmax_scores = log_softmax_scores
        self.group_query_attention = group_query_attention
        if self.group_query_attention is not None:
            self.group_query_attention = GQA(self.group_query_attention)
        self.output_all_logits = output_all_logits
        self.attn_output_transposed = attn_output_transposed
        self.allow_flash_attention = allow_flash_attention

    @property
    def vectorize_last_token_id(self):
        return self.continuous_batching

    def to_json(self):
        json_serializable_types = (str, int, float, bool)

        def _to_json(obj):
            if obj is None or isinstance(obj, json_serializable_types):
                return obj
            elif isinstance(obj, enum.Enum):
                return obj.value
            elif isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, list):
                return [_to_json(e) for e in obj]
            elif isinstance(obj, dict):
                return {_to_json(k): _to_json(v) for k, v in obj.items()}
            elif isinstance(obj, tuple):
                return str(tuple(_to_json(e) for e in obj))
            else:
                as_dict = obj.__dict__
                return _to_json(as_dict)

        return _to_json(self)
