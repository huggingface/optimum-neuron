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
"""Utilities for performing parallelism with `neuronx_distributed`"""

from typing import Literal, Union

import torch

from ..utils import is_neuronx_distributed_available


if is_neuronx_distributed_available():
    from neuronx_distributed.parallel_layers import layers
    from neuronx_distributed.parallel_layers.parallel_state import (
        get_tensor_model_parallel_rank,
    )

TENSOR_PARALLEL_SHARDS_DIR_NAME = "tensor_parallel_shards"


def linear_to_parallel_linear(
    linear_layer: "torch.nn.Linear",
    axis: Union[Literal["row"], Literal["column"]],
    input_is_parallel: bool = False,
    gather_output: bool = True,
):
    if axis not in ["row", "column"]:
        raise ValueError(f'axis must either be "row" or "column", but {axis} was given here.')

    kwargs = {}
    if axis == "row":
        parallel_linear_class = layers.RowParallelLinear
        kwargs["input_is_parallel"] = input_is_parallel
    else:
        parallel_linear_class = layers.ColumnParallelLinear
        kwargs["gather_output"] = gather_output

    parallel_linear_layer = parallel_linear_class(linear_layer.in_features, linear_layer.out_features, **kwargs)

    tp_rank = get_tensor_model_parallel_rank()
    row_size, col_size = parallel_linear_layer.weight.shape

    with torch.no_grad():
        # TODO: what about the bias?
        if axis == "row":
            parallel_linear_layer.weight.copy_(linear_layer.weight[:, tp_rank * col_size : (tp_rank + 1) * col_size])
        else:
            parallel_linear_layer.weight.copy_(linear_layer.weight[tp_rank * row_size : (tp_rank + 1) * row_size, :])

    parallel_linear_layer.to(linear_layer.weight.device)
    return parallel_linear_layer
