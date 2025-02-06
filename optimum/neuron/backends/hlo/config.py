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
import warnings
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
        padding_side: The expected tokenizer batch padding side. See:
            https://huggingface.co/docs/transformers/v4.39.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.padding_side
            The default padding side is "left", however using "right"
            padding enables variable length sequences to be used. This is
            enabled when using features such as continuous batching.
        group_query_attention: The sharding configuration to use when the number
            of query attention heads is not equal to the number of key/value
            heads. Neuron attempts to select the best configuration by default.
        bf16_rms_norm: Uses BF16 weights and hidden states input for RMS norm operations.
            By default, the RMS norm operates on FP32 dtype of inputs.
        all_reduce_dtype: The data type that is used for AllReduce collectives.
            To be selected from `["float32", "float16", "bfloat16"]`.
        cast_logits_dtype: The data type to cast logits to in the forward
            pass. To be selected from `["float32", "float16", "bfloat16"]`.
        fuse_qkv: Fuses the QKV projection into a single matrix multiplication.
        log_softmax_scores: Return log-softmax scores along with logits.
        output_all_logits: Return all logits from each model invocation.
        attn_output_transposed: Transposes the attention output projection weight tensor.
        compilation_worker_count: Count of concurrent compilation workers.
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
        padding_side: str = "left",
        group_query_attention: Optional[GQA] = None,
        bf16_rms_norm: bool = False,
        all_reduce_dtype: Optional[str] = None,
        cast_logits_dtype: str = "float32",
        fuse_qkv: bool = False,
        log_softmax_scores: bool = False,
        output_all_logits: bool = False,
        attn_output_transposed: bool = False,
        compilation_worker_count: Optional[int] = None,
        **kwargs,
    ):
        self.n_positions = n_positions
        self.batch_size = batch_size
        self.amp = amp
        self.tp_degree = tp_degree
        self.all_reduce_dtype = all_reduce_dtype
        self.cast_logits_dtype = cast_logits_dtype
        assert cast_logits_dtype in valid_dtypes, (
            f"The `cast_logits_dtype={cast_logits_dtype}` argument must be one of {valid_dtypes}"
        )
        self.fuse_qkv = fuse_qkv
        self.continuous_batching = continuous_batching
        self.padding_side = padding_side
        assert padding_side in [
            "left",
            "right",
        ], f"The `padding_side={padding_side}` argument must be either 'left' or 'right'"

        self.lhs_aligned = padding_side == "right"
        if "use_2d_cache_ids" in kwargs:
            warnings.warn(
                "NeuronConfig `use_2d_cache_ids` argument is deprecated. Please specify `padding_side = 'right'`."
            )
            self.lhs_aligned = kwargs.pop("use_2d_cache_ids", False)
        if "lhs_aligned" in kwargs:
            warnings.warn(
                "NeuronConfig `lhs_aligned` argument is deprecated. Please specify `padding_side = 'right'`."
            )
            self.lhs_aligned = kwargs.pop("lhs_aligned", False)
        if self.continuous_batching:
            # Force left alignment for continuous batching.
            self.lhs_aligned = True
            self.padding_side = "right"
        self.attention_layout = attention_layout
        self.collectives_layout = collectives_layout
        self.log_softmax_scores = log_softmax_scores
        self.group_query_attention = group_query_attention
        if self.group_query_attention is not None:
            self.group_query_attention = GQA(self.group_query_attention)
        self.bf16_rms_norm = bf16_rms_norm
        self.output_all_logits = output_all_logits

        assert len(kwargs) == 0, f"Unexpected NeuronConfig keyword arguments: {kwargs}"

        self.dist = None

        self.layer_partition = {}

        self.attn_output_transposed = attn_output_transposed

        self.compilation_worker_count = compilation_worker_count

    @property
    def use_2d_cache_ids(self):
        return self.lhs_aligned

    @property
    def vectorize_last_token_id(self):
        return self.lhs_aligned

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
