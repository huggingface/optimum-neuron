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

from typing import Dict, Literal, Optional, Tuple, Union

import torch

from ..utils import is_neuronx_distributed_available


if is_neuronx_distributed_available():
    from neuronx_distributed.parallel_layers import layers
    from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_rank

TENSOR_PARALLEL_SHARDS_DIR_NAME = "tensor_parallel_shards"


def embedding_to_parallel_embedding(
    embedding_layer: "torch.nn.Embedding",
    lm_head_layer: Optional["torch.nn.Linear"] = None,
    orig_to_parallel: Optional[Dict[int, "torch.nn.Parameter"]] = None,
) -> Union["layers.ParallelEmbedding", Tuple["layers.ParallelEmbedding", "layers.ColumnParallelLinear"]]:
    parallel_embedding_layer = layers.ParallelEmbedding(
        embedding_layer.num_embeddings,
        embedding_layer.embedding_dim,
        dtype=embedding_layer.weight.dtype,
    )

    tp_rank = get_tensor_model_parallel_rank()
    row_size, _ = parallel_embedding_layer.weight.shape

    is_tied = False
    if lm_head_layer is not None:
        is_tied = id(embedding_layer.weight.data) == id(lm_head_layer.weight.data)

    with torch.no_grad():
        parallel_embedding_layer.weight.copy_(embedding_layer.weight[tp_rank * row_size : (tp_rank + 1) * row_size, :])
        if lm_head_layer is not None:
            parallel_lm_head_layer = linear_to_parallel_linear(
                lm_head_layer,
                "column",
                gather_output=True,
                orig_to_parallel=orig_to_parallel if not is_tied else None,
            )
            if is_tied:
                parallel_lm_head_layer.weight = parallel_embedding_layer.weight

    if orig_to_parallel:
        orig_to_parallel[id(embedding_layer.weight)] = parallel_embedding_layer.weight

    del embedding_layer.weight

    if lm_head_layer is None:
        return parallel_embedding_layer
    return parallel_embedding_layer, parallel_lm_head_layer


def linear_to_parallel_linear(
    linear_layer: "torch.nn.Linear",
    axis: Union[Literal["row"], Literal["column"]],
    input_is_parallel: bool = False,
    gather_output: bool = True,
    orig_to_parallel: Optional[Dict[int, "torch.nn.Parameter"]] = None,
) -> Union["layers.RowParallelLinear", "layers.ColumnParallelLinear"]:
    if axis not in ["row", "column"]:
        raise ValueError(f'axis must either be "row" or "column", but {axis} was given here.')

    kwargs = {}
    if axis == "row":
        parallel_linear_class = layers.RowParallelLinear
        kwargs["input_is_parallel"] = input_is_parallel
    else:
        parallel_linear_class = layers.ColumnParallelLinear
        kwargs["gather_output"] = gather_output

    kwargs["dtype"] = linear_layer.weight.dtype
    kwargs["bias"] = linear_layer.bias is not None

    parallel_linear_layer = parallel_linear_class(linear_layer.in_features, linear_layer.out_features, **kwargs)

    tp_rank = get_tensor_model_parallel_rank()
    row_size, col_size = parallel_linear_layer.weight.shape

    with torch.no_grad():
        if axis == "row":
            parallel_linear_layer.weight.copy_(linear_layer.weight[:, tp_rank * col_size : (tp_rank + 1) * col_size])
            if linear_layer.bias is not None:
                parallel_linear_layer.bias.copy_(linear_layer.bias)
                if orig_to_parallel is not None:
                    orig_to_parallel[id(linear_layer.bias)] = parallel_linear_layer.bias
        else:
            parallel_linear_layer.weight.copy_(linear_layer.weight[tp_rank * row_size : (tp_rank + 1) * row_size, :])
            if linear_layer.bias is not None:
                parallel_linear_layer.bias.copy_(linear_layer.bias[tp_rank * row_size : (tp_rank + 1) * row_size])
                if orig_to_parallel is not None:
                    orig_to_parallel[id(linear_layer.bias)] = parallel_linear_layer.bias

    if orig_to_parallel is not None:
        orig_to_parallel[id(linear_layer.weight)] = parallel_linear_layer.weight

    return parallel_linear_layer
