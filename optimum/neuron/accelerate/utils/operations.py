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

import pickle
from typing import Any, Callable

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from neuronx_distributed.parallel_layers.parallel_state import (
    get_context_model_parallel_size,
    get_data_parallel_group,
    get_data_parallel_replica_groups,
    get_data_parallel_size,
    get_pipeline_model_parallel_replica_groups,
    get_tensor_model_parallel_replica_groups,
)


def broadcast_object(obj: Any, src: int = 0, groups: list[list[int]] | None = None, world_size_function: Callable[[], int] = xr.world_size, get_rank_function: Callable[[], int] = xr.global_ordinal, fixed_size: int | None = None) -> Any:
    """
    Broadcasts arbitrary objects across XLA-distributed processes.
    Returns the object from the source rank on all ranks.
    If `groups` is specified, broadcast is done separately in each group, and the `src` rank is relative to each group.
    """
    world_size = world_size_function()
    if world_size == 1:
        return obj

    rank = get_rank_function()

    if rank == src:
        bytes_ = pickle.dumps(obj)
        length = len(bytes_)
        # Ensure the serialized object fits in the fixed size if specified.
        # Otherwise we would corrupt the transferred data.
        if fixed_size is not None and length > fixed_size:
            raise ValueError(f"Serialized object size {length} exceeds the specified fixed_size {fixed_size}")
    else:
        bytes_ = b""
        length = 0

    # First, broadcast the length of the serialized object.
    max_length = xm.all_reduce("max", torch.tensor(length, dtype=torch.int64).to(xm.xla_device()))
    max_length = max_length.cpu()

    # Ensure all ranks agree on the max length.
    torch_xla.sync()

    max_length = int(max_length.item())

    if fixed_size is not None:
        target_length = fixed_size
    else:
        target_length = max_length

    if rank == src:
        np_buffer = np.frombuffer(bytes_, dtype=np.uint8)
        data_tensor = torch.from_numpy(np_buffer).to(xm.xla_device())
        padding_length = target_length - length
        if padding_length > 0:
            padding_tensor = torch.zeros(padding_length, dtype=torch.uint8, device=xm.xla_device())
            data_tensor = torch.cat([data_tensor, padding_tensor], dim=0)
    else:
        data_tensor = torch.zeros(target_length, dtype=torch.uint8, device=xm.xla_device())

    data_tensor = xm.all_reduce("sum", data_tensor, groups=groups)
    torch_xla.sync()

    # In this case we truncate the tensor to the original max length on device to minimize the data transfer from device
    # to host.
    if fixed_size is None:
        data_tensor = data_tensor[:max_length]

    data_tensor_cpu = data_tensor.cpu()
    reduced_bytes = data_tensor_cpu.numpy().tobytes()

    # Truncate to the original max length on host if fixed_size is specified to avoid changing shapes on device.
    if fixed_size is not None:
        reduced_bytes = reduced_bytes[:max_length]

    return pickle.loads(reduced_bytes)


def broadcast_object_to_data_parallel_group(obj: Any, src: int = 0, fixed_size: int | None = None) -> Any:
    """
    Broadcasts arbitrary objects across XLA-distributed data parallel group.
    Returns the object from the source rank on all ranks in the data parallel group.
    """
    groups = get_data_parallel_replica_groups()
    return broadcast_object(
        obj,
        src=src,
        groups=groups,
        world_size_function=get_data_parallel_size,
        get_rank_function=get_data_parallel_replica_groups,
        fixed_size=fixed_size,
    )

def broadcast_object_to_tensor_model_parallel_group(obj: Any, src: int = 0, fixed_size: int | None = None) -> Any:
    """
    Broadcasts arbitrary objects across XLA-distributed tensor model parallel group.
    Returns the object from the source rank on all ranks in the tensor model parallel group.
    """
    groups = get_tensor_model_parallel_replica_groups()
    return broadcast_object(
        obj,
        src=src,
        groups=groups,
        world_size_function=get_context_model_parallel_size,
        get_rank_function=get_tensor_model_parallel_replica_groups,
        fixed_size=fixed_size,
    )

def broadcast_object_to_pipeline_model_parallel_group(obj: Any, src: int = 0, fixed_size: int | None = None) -> Any:
    """
    Broadcasts arbitrary objects across XLA-distributed pipeline model parallel group.
    Returns the object from the source rank on all ranks in the pipeline model parallel group.
    """
    groups = get_pipeline_model_parallel_replica_groups()
    return broadcast_object(
        obj,
        src=src,
        groups=groups,
        world_size_function=get_context_model_parallel_size,
        get_rank_function=get_pipeline_model_parallel_replica_groups,
        fixed_size=fixed_size,
    )


def gather_object(obj: Any) -> list[Any]:
    """
    Gathers arbitrary objects across XLA-distributed processes.
    Returns list of objects from all ranks on all ranks.

    Note: Requires two all-gather operations (lengths then data).
    For small objects, this overhead may be significant.
    """
    world_size = get_data_parallel_size()

    # Early exit for single process
    if world_size == 1:
        return [obj]

    groups = get_data_parallel_group(as_list=True)

    serialized = pickle.dumps(obj)
    byte_len = len(serialized)

    byte_tensor = torch.frombuffer([serialized], dtype=torch.uint8).to(xm.xla_device())

    len_tensor = torch.tensor([byte_len], dtype=torch.int64, device=byte_tensor.device)
    # all_gather concatenates along dim=0, so [1] -> [world_size]
    gathered_lengths = xm.all_gather(len_tensor, dim=0, groups=groups, pin_layout=False)
    torch_xla.sync()

    max_len = torch.max(gathered_lengths)
    max_len = int(max_len.item())
    padded = torch.zeros(max_len, dtype=torch.uint8, device=byte_tensor.device)
    padded[:byte_len] = byte_tensor

    # all_gather concatenates, so [max_len] -> [world_size * max_len]
    gathered_data = xm.all_gather(padded, dim=0, groups=groups, pin_layout=False)
    torch_xla.sync()

    gathered_data_cpu = gathered_data.cpu()
    gathered_lengths_cpu = gathered_lengths.cpu()

    results = []
    offset = 0
    for i in range(world_size):
        actual_len = int(gathered_lengths_cpu[i].item())
        valid_bytes = gathered_data_cpu[offset:offset + max_len][:actual_len].numpy().tobytes()
        valid_bytes = valid_bytes[0]
        results.append(pickle.loads(valid_bytes))
        offset += max_len

    return results
