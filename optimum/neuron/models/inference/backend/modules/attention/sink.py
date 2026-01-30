# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/aws-neuron/neuronx-distributed-inference/blob/main/src/neuronx_distributed_inference/modules/attention/sink.py
"""Learned sink module for attention."""

from typing import List, Optional

import torch
from neuronx_distributed.parallel_layers.layers import BaseParallelLinear
from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_size
from neuronx_distributed.parallel_layers.utils import divide, set_tensor_model_parallel_attributes


class LearnedSink(BaseParallelLinear):
    """Learned sink for attention computation.

    This module implements learned sinks as described in the GPT-OSS architecture.
    The sink is a learnable parameter that is added to the attention scores before softmax.

    Args:
        learned_sinks_size: Size of the learned sinks (currently only 1 is supported)
        num_attention_heads: Number of attention heads
        torch_dtype: Data type for the sink tensor
        tensor_model_parallel_size: Tensor parallel size (defaults to world size)
        rank_ordering: Optional ordering of ranks for tensor parallel
    """

    def __init__(
        self,
        learned_sinks_size: int,
        num_attention_heads: int,
        torch_dtype: torch.dtype,
        tensor_model_parallel_size: Optional[int] = None,
        rank_ordering: Optional[List[int]] = None,
    ):
        super().__init__()
        assert learned_sinks_size == 1, f"Learned sinks only supports learned_sinks_size == 1 ({learned_sinks_size})"

        self.tensor_model_parallel_size = (
            tensor_model_parallel_size if tensor_model_parallel_size is not None else get_tensor_model_parallel_size()
        )

        sink_size_per_partition = divide(num_attention_heads, self.tensor_model_parallel_size)
        self.sink = torch.nn.Parameter(torch.zeros(sink_size_per_partition, dtype=torch_dtype), requires_grad=False)

        set_tensor_model_parallel_attributes(
            self.sink,
            is_parallel=True,
            dim=0,
            stride=1,
            num_partitions=self.tensor_model_parallel_size,
            rank_ordering=rank_ordering,
        )

    def get_sink(self) -> torch.Tensor:
        """Return the sink tensor."""
        return self.sink
