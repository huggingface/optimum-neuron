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
"""Tests checking that `neuronx_distributed` parallel layers are working as expected."""

import pytest
import torch
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    create_local_weight,
)
from neuronx_distributed.parallel_layers.utils import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_size,
)
from torch import nn
from transformers import set_seed

from optimum.neuron.utils.testing_utils import is_trainium_test

from ..distributed_utils import distributed_test
from .utils import assert_close


import torch_xla.core.xla_model as xm


@is_trainium_test
@pytest.mark.parametrize(
    "row_or_column",
    ["row", "column"],
    ids=["row_parallel_linear", "column_parallel_linear"],
)
@pytest.mark.parametrize(
    "weights_dtype, inputs_dtype",
    [
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float32),
    ],
    ids=[
        "weights=float32-inputs=float32",
        "weights=bfloat16-inputs=bfloat16",
        "weights=bfloat16-inputs=float32",
    ],
)
@pytest.mark.parametrize(
    "input_size, output_size",
    [(8192, 2048), (400, 300)],
    ids=["input_size=8192-output_size=2048", "input_size=400-output_size=300"],
)
@distributed_test(world_size=2, tp_size=2, pp_size=1)
def test_parallel_linears(row_or_column, weights_dtype, inputs_dtype, input_size, output_size):
    set_seed(42)
    device = "xla"

    inputs = torch.randn(1, 12, input_size, dtype=inputs_dtype).to(device)

    # First we compute the expected output using the linear.
    linear = nn.Linear(input_size, output_size, bias=False, dtype=weights_dtype, device=device)

    outputs = linear(inputs)
    xm.mark_step()

    # Then we compute the output using the parallel linear.
    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_size()

    if row_or_column == "column":
        partition_dim = 0
        per_partition_size = linear.weight.size(partition_dim) // tp_size
        parallel_linear = ColumnParallelLinear(
            input_size,
            output_size,
            bias=False,
            dtype=weights_dtype,
            gather_output=False,
            reduce_dtype=torch.float32,
            sequence_parallel_enabled=False,
        ).to(device="xla")

        parallel_inputs = inputs
    else:
        partition_dim = 1
        per_partition_size = linear.weight.size(partition_dim) // tp_size
        parallel_linear = RowParallelLinear(
            input_size,
            output_size,
            bias=False,
            dtype=weights_dtype,
            input_is_parallel=True,
            reduce_dtype=torch.float32,
            sequence_parallel_enabled=False,
        ).to(device="xla")

        parallel_inputs = inputs[:, :, tp_rank * per_partition_size : (tp_rank + 1) * per_partition_size]

    stride = 1
    with torch.no_grad():
        parallel_linear.weight.data = create_local_weight(linear.weight, partition_dim, per_partition_size, stride).to(
            device="xla"
        )

    parallel_outputs = parallel_linear(parallel_inputs)

    if row_or_column == "column":
        parallel_outputs = xm.all_gather(parallel_outputs, dim=-1)

    xm.mark_step()

    outputs = outputs.to("cpu")
    parallel_outputs = parallel_outputs.to("cpu")

    # Finally we compare that the outputs are the same.
    assert_close(outputs, parallel_outputs, msg="Sharded linear output does not match the unsharded one.")
