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
    get_data_parallel_replica_groups,
    get_data_parallel_size,
    get_pipeline_model_parallel_replica_groups,
    get_tensor_model_parallel_replica_groups,
)

from ...utils.misc import is_precompilation


def broadcast_object(
    obj: Any,
    src: int = 0,
    groups: list[list[int]] | None = None,
    world_size_function: Callable[[], int] = xr.world_size,
    get_rank_function: Callable[[], int] = xr.global_ordinal,
    fixed_size: int | None = None,
) -> Any:
    """
    Broadcasts arbitrary picklable objects across XLA-distributed processes.
    Returns the object from the source rank on all ranks.
    If `groups` is specified, broadcast is done separately in each group, and the `src` rank is relative to each group.

    Args:
        obj (Any): The object to broadcast. Must be picklable (serializable via pickle).
        src (int, defaults to `0`): The source rank within each group.
        groups (list[list[int]] | None, defaults to `None`): Optional list of process groups for separate broadcasts.
        world_size_function (Callable[[], int], defaults to `xr.world_size`): Function to get the world size.
        get_rank_function (Callable[[], int], defaults to `xr.global_ordinal`): Function to get the current rank.
        fixed_size (int | None, defaults to `None`): Optional fixed buffer size for serialization. If specified,
            the serialized object must not exceed this size.

    Returns:
        Any: The broadcast object from the source rank.

    Note:
        Objects must be picklable. This includes most Python built-in types, but excludes
        certain objects like lambdas, local functions, or objects with open file handles.
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
    max_length = xm.all_reduce("max", torch.tensor([length], dtype=torch.int64).to(xm.xla_device()))
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
        padding_length = target_length - length
        if padding_length > 0:
            padding = np.zeros([padding_length], dtype=np.uint8)
            np_buffer = np.concatenate([np_buffer, padding], axis=0)
        data_tensor = torch.from_numpy(np_buffer).to(xm.xla_device())
    else:
        data_tensor = torch.zeros(target_length, dtype=torch.uint8, device=xm.xla_device())

    data_tensor = xm.all_reduce("sum", data_tensor, groups=groups)
    torch_xla.sync()

    data_tensor_cpu = data_tensor.cpu()
    reduced_bytes = data_tensor_cpu.numpy().tobytes()

    reduced_bytes = reduced_bytes[:max_length]

    return pickle.loads(reduced_bytes)


def broadcast_object_to_data_parallel_group(obj: Any, src: int = 0, fixed_size: int | None = None) -> Any:
    """
    Broadcasts arbitrary picklable objects across XLA-distributed data parallel group.
    Returns the object from the source rank on all ranks in the data parallel group.

    Args:
        obj (Any): The object to broadcast. Must be picklable (serializable via pickle).
        src (int, defaults to `0`): The source rank within the data parallel group.
        fixed_size (int | None, defaults to `None`): Optional fixed buffer size for serialization.
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
    Broadcasts arbitrary picklable objects across XLA-distributed tensor model parallel group.
    Returns the object from the source rank on all ranks in the tensor model parallel group.

    Args:
        obj (Any): The object to broadcast. Must be picklable (serializable via pickle).
        src (int, defaults to `0`): The source rank within the tensor model parallel group.
        fixed_size (int | None, defaults to `None`): Optional fixed buffer size for serialization.
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
    Broadcasts arbitrary picklable objects across XLA-distributed pipeline model parallel group.
    Returns the object from the source rank on all ranks in the pipeline model parallel group.

    Args:
        obj (Any): The object to broadcast. Must be picklable (serializable via pickle).
        src (int, defaults to `0`): The source rank within the pipeline model parallel group.
        fixed_size (int | None, defaults to `None`): Optional fixed buffer size for serialization.
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


def gather_object(
    obj: Any,
    groups: list[list[int]] | None = None,
    world_size_function: Callable[[], int] = xr.world_size,
    fixed_size: int | None = None,
) -> list[Any]:
    """
    Gathers arbitrary picklable objects across XLA-distributed processes.
    Returns list of objects from all ranks on all ranks.
    If `groups` is specified, gather is done separately in each group.

    Args:
        obj (Any): The object to gather. Must be picklable (serializable via pickle).
        groups (list[list[int]] | None, defaults to `None`): Optional list of process groups for separate gathers.
        world_size_function (Callable[[], int], defaults to `xr.world_size`): Function to get the world size.
        fixed_size (int | None, defaults to `None`): Optional fixed buffer size for serialization. If specified,
            the serialized object must not exceed this size.

    Returns:
        list[Any]: List of objects from all ranks.

    Note:
        Objects must be picklable. This includes most Python built-in types, but excludes
        certain objects like lambdas, local functions, or objects with open file handles.
    """
    world_size = world_size_function()

    # Early exit for single process
    if world_size == 1:
        return [obj]

    serialized = pickle.dumps(obj)
    length = len(serialized)

    if fixed_size is not None and length > fixed_size:
        raise ValueError(f"Serialized object size {length} exceeds the specified fixed_size {fixed_size}")

    lengths = xm.all_gather(
        torch.tensor([length], dtype=torch.int64).to(device=xm.xla_device()),
        dim=0,
        groups=groups,
        pin_layout=False,
    )
    torch_xla.sync()
    lengths_cpu = lengths.cpu()
    max_length = lengths_cpu.max()
    max_length = int(max_length.item())

    if fixed_size is not None:
        target_length = fixed_size
    else:
        target_length = max_length

    np_buffer = np.frombuffer(serialized, dtype=np.uint8)
    padding_length = target_length - length
    if padding_length > 0:
        padding = np.zeros([padding_length], dtype=np.uint8)
        np_buffer = np.concatenate([np_buffer, padding], axis=0)
    data_tensor = torch.from_numpy(np_buffer).to(xm.xla_device())

    data_tensor = xm.all_gather(
        data_tensor,
        dim=0,
        groups=groups,
        pin_layout=False,
    )
    torch_xla.sync()

    data_tensors_cpu = data_tensor.cpu().split(target_length)
    data_bytes = [t.numpy().tobytes() for t in data_tensors_cpu]

    # During precompilation, all_gather returns tensors with uninitialized data or zeros,
    # breaking the pickle.loads step below. So we return a list of the original object instead,
    # it should not break anything since precompilation does not rely on the gathered objects.
    if is_precompilation():
        return [obj for _ in range(world_size)]

    results = []
    for i in range(world_size):
        length_i = lengths_cpu[i].item()
        bytes_i = data_bytes[i][:length_i]
        obj_i = pickle.loads(bytes_i)
        results.append(obj_i)

    return results


def gather_object_from_data_parallel_group(obj: Any, fixed_size: int | None = None) -> list[Any]:
    """
    Gathers arbitrary picklable objects across XLA-distributed data parallel group.
    Returns list of objects from all ranks in the data parallel group on all ranks.

    Args:
        obj (Any): The object to gather. Must be picklable (serializable via pickle).
        fixed_size (int | None, defaults to `None`): Optional fixed buffer size for serialization.
    """
    groups = get_data_parallel_replica_groups()
    return gather_object(
        obj,
        groups=groups,
        world_size_function=get_data_parallel_size,
        fixed_size=fixed_size,
    )


def gather_object_from_tensor_model_parallel_group(obj: Any, fixed_size: int | None = None) -> list[Any]:
    """
    Gathers arbitrary picklable objects across XLA-distributed tensor model parallel group.
    Returns list of objects from all ranks in the tensor model parallel group on all ranks.

    Args:
        obj (Any): The object to gather. Must be picklable (serializable via pickle).
        fixed_size (int | None, defaults to `None`): Optional fixed buffer size for serialization.
    """
    groups = get_tensor_model_parallel_replica_groups()
    return gather_object(
        obj,
        groups=groups,
        world_size_function=get_context_model_parallel_size,
        fixed_size=fixed_size,
    )


def gather_object_from_pipeline_model_parallel_group(obj: Any, fixed_size: int | None = None) -> list[Any]:
    """
    Gathers arbitrary picklable objects across XLA-distributed pipeline model parallel group.
    Returns list of objects from all ranks in the pipeline model parallel group on all ranks.

    Args:
        obj (Any): The object to gather. Must be picklable (serializable via pickle).
        fixed_size (int | None, defaults to `None`): Optional fixed buffer size for serialization.
    """
    groups = get_pipeline_model_parallel_replica_groups()
    return gather_object(
        obj,
        groups=groups,
        world_size_function=get_context_model_parallel_size,
        fixed_size=fixed_size,
    )
