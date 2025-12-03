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
# Adapted from https://github.com/aws-neuron/neuronx-distributed-inference/blob/9993358ce052fd7a1bb4a7497a6318aac36ed95c/src/neuronx_distributed_inference/utils/distributed.py
import os

import torch
from neuronx_distributed.parallel_layers.utils import divide


def get_init_world_size() -> int:
    """Get world size set by distributed launcher (torchrun or mpirun)"""
    for var in ["WORLD_SIZE", "OMPI_COMM_WORLD_SIZE"]:
        if var in os.environ and os.environ[var] != "":
            return int(os.environ[var])
    return -1


def get_init_rank() -> int:
    """Get rank set by distributed launcher (torchrun or mpirun)"""
    for var in ["RANK", "OMPI_COMM_WORLD_RANK"]:
        if var in os.environ and os.environ[var] != "":
            return int(os.environ[var])
    return -1


def get_dp_rank_spmd(global_rank: torch.tensor, tp_degree: int):
    dp_rank = torch.div(
        global_rank,
        tp_degree,
        rounding_mode="floor",
    ).to(torch.int32)
    return dp_rank


def split_along_dim(tensor: torch.tensor, dim: int, rank: int, num_partitions: int):
    if tensor is None:
        return None

    num_per_partition = divide(tensor.size(dim), num_partitions)
    indices = torch.arange(0, num_per_partition, device=tensor.device)
    indices = indices + (rank * num_per_partition)
    tensor = torch.index_select(tensor, dim=dim, index=indices)

    return tensor
