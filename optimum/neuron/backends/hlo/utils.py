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
import itertools
import math

import torch
import torch.nn.functional as F

from .config import GQA, NeuronConfig


def parse_dtype_replica_groups(neuron_config):
    dtype = None
    replica_groups = None

    dtype = neuron_config.all_reduce_dtype
    replica_groups = [list(range((neuron_config.tp_degree)))]

    return dtype, replica_groups


def pad_sizes(shape, dims, sizes, left=False):
    if isinstance(dims, int):
        dims = (dims,)
    if isinstance(sizes, int):
        sizes = (sizes,) * len(dims)
    lhs = [0] * len(shape)
    rhs = [0] * len(shape)
    side = lhs if left else rhs
    for dim, size in zip(dims, sizes):
        # Don't truncate tensor if the current size exceeds the "padded size"
        side[dim] = max(0, size - shape[dim])
    sizes = tuple(itertools.chain(*zip(reversed(lhs), reversed(rhs))))
    if sum(sizes) == 0:
        return None
    return sizes


def pad_interleaved(tensor, dim, size, source_len_per_group, pad_len_per_group):
    """
    Pad the selected `dim` of `tensor`, up to `size`. Put zeros interleavedly base on `source_len_per_group`
    and `pad_len_per_group`

        [
            # group i # ? ... x source_len_per_group ... ?, 0, ... x pad_len_per_group ... , 0]
        ]

    For example:
        pad_interleaved([1,2,3]], dim=0, size=9, source_len_per_group=1, pad_len_per_group=2)
        each group becomes [?, 0, 0], with number of group as 3, we have
        result: [1, 0, 0, 2, 0, 0, 3, 0, 0]
    """
    assert isinstance(dim, int), "pad_interleaved now only supports single dim"

    assert isinstance(size, int), "pad_interleaved now only supports single dim of size"

    padded_shape = list(tensor.shape)

    padded_shape[dim] = size

    padded_tensor = torch.zeros(padded_shape, dtype=tensor.dtype)

    num_groups = size // (source_len_per_group + pad_len_per_group)

    src_indices = [slice(0, None) for _ in range(len(padded_shape))]
    target_indices = [slice(0, None, None) for _ in range(len(padded_shape))]
    s_src = 0
    s_target = 0
    for _ in range(num_groups):
        target_indices[dim] = slice(s_target, s_target + source_len_per_group)
        src_indices[dim] = slice(s_src, s_src + source_len_per_group)
        padded_tensor[target_indices] = tensor[src_indices]
        s_src += source_len_per_group
        s_target += source_len_per_group + pad_len_per_group

    return padded_tensor


def maybe_pad_tensor(tensor, dims, sizes, left=False):
    if tensor is None:
        return tensor
    padding = pad_sizes(tensor.shape, dims, sizes, left=left)
    if padding is not None:
        if isinstance(tensor, torch.nn.Parameter):
            tensor = tensor.detach()
        return F.pad(tensor, padding)
    return tensor


def round_up_to_divisor(value, divisor):
    return math.ceil(value / divisor) * divisor


def get_pad_size(vocab_size, divisor):
    return ((vocab_size // divisor + 1) * divisor - vocab_size) % divisor


def interleave_qkv(q, k, v, tp_degree, dim=1):
    """
    Create a merged QKV Tensor with weights arranged for sharding.

    Args:
        q(torch.Tensor): Weights/Bias for Q
        k(torch.Tensor): Weights/Bias for K
        v(torch.Tensor): Weights/Bias for V
        tp_degree(int): tp_degree used for sharding
        dim(int): Dimension to concatenate tensors
    Returns:
        tensor: Concatenated QKV tensor with interleaved weights
    """

    def get_slice_params(tensor, dim, tp_degree):
        size = tensor.shape[dim]
        shard_size = size // tp_degree
        slices = [slice(None) for _ in tensor.shape]
        return size, shard_size, slices

    def get_shard_tensors(tensors, size, shard_size, slices, dim):
        for idx, start in enumerate(range(0, size, shard_size)):
            slices[dim] = slice(start, start + shard_size, 1)
            shard_tensors = [t[tuple(slices)].contiguous() for t in tensors]
            yield idx, shard_tensors

    if q.shape[dim] == k.shape[dim]:
        fused_qkv_ratio = 3  # Q + K + V
        size, shard_size, slices = get_slice_params(q, dim, tp_degree)
        is_single_dim = len(q.shape) == 1
        if is_single_dim:
            tensor = torch.zeros((size * fused_qkv_ratio), dtype=q.dtype)
        else:
            hidden_dim, interleave_dim = q.shape
            tensor = torch.zeros((hidden_dim, interleave_dim * fused_qkv_ratio), dtype=q.dtype)
        for idx, shard_tensors in get_shard_tensors((q, k, v), size, shard_size, slices, dim):
            shard = torch.cat(shard_tensors, dim=dim).contiguous()
            if is_single_dim:
                tensor[(idx) * shard.shape[dim] : (idx + 1) * shard.shape[dim]] = shard
            else:
                tensor[:, (idx) * shard.shape[dim] : (idx + 1) * shard.shape[dim]] = shard
    else:
        q_hidden_dim, q_interleave_dim = q.shape
        _, kv_interleave_dim = k.shape
        tensor = torch.zeros((q_hidden_dim, q_interleave_dim + kv_interleave_dim * 2), dtype=q.dtype)
        q_size, q_shard_size, q_slices = get_slice_params(q, dim, tp_degree)
        kv_size, kv_shard_size, kv_slices = get_slice_params(k, dim, tp_degree)
        for idx, ((_, q_shard_tensors), (_, kv_shard_tensors)) in enumerate(
            zip(
                get_shard_tensors((q,), q_size, q_shard_size, q_slices, dim),
                get_shard_tensors((k, v), kv_size, kv_shard_size, kv_slices, dim),
            )
        ):
            q_shard = q_shard_tensors[0]
            kv_shard = torch.concat(kv_shard_tensors, dim=dim).contiguous()
            shard = torch.cat((q_shard, kv_shard), dim=dim).contiguous()
            tensor[:, (idx) * shard.shape[1] : (idx + 1) * shard.shape[1]] = shard
    return tensor


def build_replica_groups(num_groups, group_size, interleave=False):
    """
    Construct replica_groups to handle "intra-group" reduce operations.

    Each nested list represents the ids of the cores within the same group.

    Examples:

        group_size = 2
        num_groups = 3
        replica_groups = [[0, 1], [2, 3], [4, 5]]

        group_size = 3
        num_groups = 2
        replica_groups = [[0, 1, 2], [3, 4, 5]]

    """
    if interleave:
        limit = num_groups * group_size
        ncs = list(range(limit))
        slices = [slice(i, limit, num_groups) for i in range(num_groups)]
        replica_groups = [ncs[s] for s in slices]
    else:
        replica_groups = [
            list(range(group_size * group, group_size * group + group_size)) for group in range(num_groups)
        ]
    return replica_groups


def get_qkv_padding(
    n_head: int,
    n_kv_head: int,
    neuron_config: NeuronConfig,
):
    """
    Compute the number of Q & KV heads after required padding.

    This function does not validate that the GQA configuration is valid for the
    given tensor parallel degree.

    Args:
        n_head: The original number of Q heads.
        n_kv_head: The original number of KV heads.
        neuron_config: The neuron configurations including the sharding options.

    Returns:
        n_head_padded: The total number of Q heads after padding.
        n_kv_head_padded: The total number of KV heads after padding.
    """
    n_head_padded = round_up_to_divisor(n_head, neuron_config.tp_degree)
    # Assume no KV padding by default. This applies to:
    # - SHARD_OVER_HEADS -> Split by head across NCs (Assumes validated size)
    n_kv_head_padded = n_kv_head

    gqa = GQA.SHARD_OVER_HEADS
    if neuron_config is not None:
        gqa = neuron_config.group_query_attention

    # When replicated, we need to check when we must fully/partially replicated
    if n_head != n_kv_head and gqa == GQA.REPLICATED_HEADS:
        # Full Replication - Base case: replicate up to the number of Q heads
        ratio = int(n_head / n_kv_head)

        # Partial replication - Check if we can reduce the replication of the
        # KV head only up to the tensor parallel degree so that the data copying
        # is minimal per NeuronCore.
        if (
            (n_kv_head < neuron_config.tp_degree)
            and (neuron_config.tp_degree % n_kv_head == 0)
            and (
                (n_head % neuron_config.tp_degree == 0)
                or ((n_head_padded - n_head) % n_kv_head == 0 and n_head_padded - n_head > 0)
            )
        ):
            ratio = int(neuron_config.tp_degree / n_kv_head)
        # In case the number of Q heads that share same KV head (i.e. n_head / n_kv_head) is equal to
        # the number of Q heads (after padding) assigned to each core (i.e. n_head_padded / tp_degree),
        # we only need to allocate one KV head to each core
        # (e.g. TP = 32, Q heads = 72 (96 after padding), KV heads = 24)
        elif n_head_padded / neuron_config.tp_degree == n_head / n_kv_head:
            ratio = 1

        n_kv_head_padded = n_kv_head * ratio

        # Ensure KV heads are at least padded to TP degree
        if n_kv_head_padded < neuron_config.tp_degree:
            n_kv_head_padded = neuron_config.tp_degree

        # Full Replication - If fully replicated but needed to pad Q heads, then
        # also pad KV heads.
        if n_kv_head_padded == n_head:
            n_kv_head_padded = n_head_padded

    return n_head_padded, n_kv_head_padded
