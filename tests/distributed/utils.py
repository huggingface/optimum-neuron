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
"""Utilities for tests distributed."""

import torch

from optimum.neuron.utils.require_utils import requires_neuronx_distributed, requires_torch_xla


@requires_torch_xla
@requires_neuronx_distributed
def gather_along_last_dim(input_: torch.Tensor) -> torch.Tensor:
    import torch_xla.core.xla_model as xm
    from neuronx_distributed.parallel_layers.parallel_state import (
        get_tensor_model_parallel_group,
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_size,
    )

    """Gathers tensors and concatenate along the last dimension."""

    world_size = get_tensor_model_parallel_size()
    # Bypass the function if we are using only 1 device.
    if world_size == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = get_tensor_model_parallel_rank()

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=get_tensor_model_parallel_group())

    # Compilation fails when not synchronizing here.
    xm.mark_step()

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    xm.mark_step()

    return output


@requires_torch_xla
@requires_neuronx_distributed
def gather_along_first_dim(input_: torch.Tensor) -> torch.Tensor:
    import torch_xla.core.xla_model as xm
    from neuronx_distributed.parallel_layers.parallel_state import (
        get_tensor_model_parallel_group,
        get_tensor_model_parallel_size,
    )

    world_size = get_tensor_model_parallel_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    output = xm.all_gather(input_, groups=get_tensor_model_parallel_group(as_list=True), pin_layout=False)

    return output


@requires_torch_xla
@requires_neuronx_distributed
def gather_along_dim(input_: torch.Tensor, dim: int) -> torch.Tensor:
    input_ = input_.clone().contiguous()
    if dim == 0:
        return gather_along_first_dim(input_)
    elif dim in [-1, input_.dim() - 1]:
        return gather_along_last_dim(input_)
    else:
        t = input_.transpose(0, dim).contiguous()
        gathered_t = gather_along_first_dim(t)
        return gathered_t.transpose(0, dim).contiguous()
