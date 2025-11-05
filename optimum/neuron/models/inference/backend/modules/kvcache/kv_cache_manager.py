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
# Adapted from https://github.com/aws-neuron/neuronx-distributed-inference/blob/9993358ce052fd7a1bb4a7497a6318aac36ed95c/src/neuronx_distributed_inference/modules/kvcache/kv_cache_manager.py

import torch
from neuronx_distributed.parallel_layers import utils
from torch import Tensor, nn
from transformers import PretrainedConfig

from ...config import NxDNeuronConfig
from ..attention.gqa import (
    determine_sharding_strategy,
    get_shardable_head_counts,
)
from .utils import dynamic_update_slice, fill_prefix


def _slice_kv_cacheline(seq_len: int, cache: Tensor):
    # Return the left-most slice (inputs are right padded)
    return torch.ops.aten.slice(cache, dim=2, start=0, end=seq_len)


class KVCacheManager(nn.Module):
    """
    Key Value Cache Management.
    It stores KV cache as a parameter list of the shape (batch_sz, num_kv_head_per_rank, max_len, head_dim),
    and vends out read and write operations.
    """

    def __init__(self, config: PretrainedConfig, neuron_config: NxDNeuronConfig, **kwargs):
        super().__init__()
        self.is_continuous_batching = neuron_config.continuous_batching
        self.num_kv_head = kwargs["num_kv_head"]

        self._init_kv_shape(config, neuron_config)

        num_layer = config.num_hidden_layers
        dtype = neuron_config.torch_dtype
        self.past_key_values = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.kv_shape, dtype=dtype), requires_grad=False) for _ in range(num_layer * 2)]
        )

    def _get_num_kv_heads_per_rank(self, config: PretrainedConfig, neuron_config: NxDNeuronConfig):
        tp_degree = neuron_config.tp_degree
        num_kv_head = self.num_kv_head
        num_atten_head = config.num_attention_heads

        gqa_sharding_strategy = determine_sharding_strategy(tp_degree, num_kv_head)
        _, num_key_value_heads = get_shardable_head_counts(
            tp_degree, num_atten_head, num_kv_head, gqa_sharding_strategy
        )

        return utils.divide(num_key_value_heads, tp_degree)

    def _init_kv_shape(self, config: PretrainedConfig, neuron_config: NxDNeuronConfig):
        max_batch_size = neuron_config.max_batch_size
        max_len = neuron_config.sequence_length
        num_kv_heads_per_rank = self._get_num_kv_heads_per_rank(config, neuron_config)
        hidden_dim_per_head = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        # KV cache layout : BHSD
        self.kv_shape = (
            max_batch_size,
            num_kv_heads_per_rank,
            max_len,
            hidden_dim_per_head,
        )

    def _fetch_cache(self, idx: int, kvcache_buffer=None):
        if kvcache_buffer is not None:
            return kvcache_buffer[idx][0], kvcache_buffer[idx][1]
        k_cache, v_cache = self.past_key_values[idx * 2], self.past_key_values[idx * 2 + 1]
        return k_cache, v_cache

    def get_kv_by_layer_id(self, key_layer_idx, gather_index=None, slice_index=None):
        k_cache = self.past_key_values[key_layer_idx]
        v_cache = self.past_key_values[key_layer_idx + 1]
        return k_cache, v_cache

    def get_cache(self, seq_len: int, skip_slice=False, **kwargs):
        """
        Return network (all layers)'s previously cached K and V, up to seq_len.

        :param seq_len: sequence length
        :return: list of tuple of (K, V)
        """
        slice_index, gather_index = None, None
        past_key_values = []
        for key_layer_idx in range(0, len(self.past_key_values), 2):
            # get kv per layer
            k_cache, v_cache = self.get_kv_by_layer_id(
                key_layer_idx, gather_index=gather_index, slice_index=slice_index
            )

            # slice for partial view
            if not skip_slice:
                k_cache = _slice_kv_cacheline(seq_len, k_cache)
                v_cache = _slice_kv_cacheline(seq_len, v_cache)

            past_key_values.append([k_cache, v_cache])
        return past_key_values

    def update_cache(
        self,
        is_for_context_encoding: bool,
        seq_ids: Tensor,
        position_ids: Tensor,
        new_key_values: list[Tensor],
        seq_len: int,
        scatter_index=None,
        active_mask=None,
        kvcache_buffer=None,
    ):
        """
        Given the passed-in new_key_values, update the cache

        :param scatter_index: tensor representing index to update
        :param is_for_context_encoding: bool
        :param seq_ids: tensor of size (batch_sz)
        :param position_ids: tensor of size (batch_sz, seq_len)
        :param new_key_values: list of tuple, the latest kv obtained at the end of the network from forward pass
        :param seq_len: sequence length
        :param scatter_index: tensor representing index to update
        :param active_mask: tensor representing index to update
        :param kvcache_buffer: if passed key states are updates to this buffer.
               kvcache_buffer is 2D list where, 1st dim for layer and the second denotes K and V.
               For example,
                    kvcache_buffer[1][0] is the K cache of the 1st layer
                    kvcache_buffer[4][1] is the V cache of the 4th layer
        :return: list of tuple of (K, V)
        """
        updated_kv_cache = []
        for idx, kv_per_layer in enumerate(new_key_values):
            latest_k, latest_v = kv_per_layer[0], kv_per_layer[1]
            k_cache, v_cache = self._fetch_cache(idx, kvcache_buffer)

            if is_for_context_encoding:
                if self.is_continuous_batching:
                    assert seq_ids.dim() == 1 and seq_ids.shape[0] == 1, "only supports single seq_id"

                    cache_idx = seq_ids

                    indices = torch.zeros(k_cache.dim(), dtype=seq_ids.dtype, device=seq_ids.device)
                    indices = indices.scatter(
                        dim=0,
                        index=torch.tensor([0], dtype=torch.int64, device=k_cache.device),
                        src=cache_idx,
                    ).to(torch.int32)

                    indices = indices.split(1)
                    indices = [t.squeeze() for t in indices]
                    k_cache = dynamic_update_slice(k_cache, latest_k, indices)
                    v_cache = dynamic_update_slice(v_cache, latest_v, indices)
                else:
                    k_cache = fill_prefix(k_cache, latest_k)
                    v_cache = fill_prefix(v_cache, latest_v)
            else:
                # Copy the tensor of the new position into kv cache (no need to align as inputs are right padded)
                scatter_index_new = self._get_index_to_update_new_position(scatter_index, position_ids, latest_k)
                k_cache = torch.scatter(input=k_cache, dim=2, index=scatter_index_new, src=latest_k)
                v_cache = torch.scatter(input=v_cache, dim=2, index=scatter_index_new, src=latest_v)

            updated_kv_cache.append(k_cache)
            updated_kv_cache.append(v_cache)

        # return updated kv cache to NxD runtime
        return updated_kv_cache

    def _get_index_to_update_new_position(self, scatter_index, position_ids, full_k):
        scatter_index = position_ids.view(-1, 1, position_ids.shape[-1], 1).expand_as(full_k)
        return scatter_index
