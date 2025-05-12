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
import logging
from typing import List

import torch
from neuronx_distributed.parallel_layers import parallel_state, utils
from torch import Tensor, nn
from torch_neuronx.xla_impl.ops import ConcatenateOp
from transformers import PretrainedConfig

from ...config import NxDNeuronConfig
from ..attention.gqa import (
    determine_sharding_strategy,
    get_shardable_head_counts,
)
from ..autobucketing import slice_lhs, slice_rhs
from ..flashdecode.utils import get_cache_size
from .utils import fill_prefix


def _slice_kv_cacheline(padding_side, seq_len: int, cache: Tensor):
    if padding_side == "right":
        return slice_lhs(cache, seq_len, 2)
    else:
        max_idx = cache.shape[2]
        return slice_rhs(cache, seq_len, max_idx, 2)


def _gather_slice_into_kv_cacheline(cache, padding_side, seq_len: int, bucket_slice: Tensor):
    max_idx = cache.shape[2]
    if padding_side == "right":
        remaining = slice_rhs(cache, max_idx - seq_len, max_idx, dim=2)
        if remaining.dtype == torch.float8_e4m3fn:
            return ConcatenateOp.apply(bucket_slice, remaining, dim=2)
        return torch.cat([bucket_slice, remaining], dim=2)
    else:
        remaining = slice_lhs(cache, max_idx - seq_len, dim=2)
        if remaining.dtype == torch.float8_e4m3fn:
            return ConcatenateOp.apply(bucket_slice, remaining, dim=2)
        return torch.cat([remaining, bucket_slice], dim=2)


class KVCacheManager(nn.Module):
    """
    Key Value Cache Management.
    It stores KV cache as a parameter list of the shape (batch_sz, num_kv_head_per_rank, max_len, head_dim),
    and vends out read and write operations.
    """

    def __init__(self, config: PretrainedConfig, neuron_config: NxDNeuronConfig, **kwargs):
        super().__init__()
        self.padding_side = neuron_config.padding_side
        self.is_continuous_batching = neuron_config.continuous_batching
        self.flash_decoding_enabled = neuron_config.flash_decoding_enabled
        self.num_cores_per_group = neuron_config.num_cores_per_group
        self.num_kv_head = kwargs["num_kv_head"]

        # NOTE: Tiling the sequence dimension of the KV cache enables specific compiler optimizations like cascaded reductions
        self.is_kv_cache_tiled = False  # TODO: enable this when compiler fixes CR 158191111 (as per NxDI comment)
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

        if parallel_state.model_parallel_is_initialized():
            num_kv_heads_per_rank = utils.divide(num_key_value_heads, tp_degree)
        else:
            num_kv_heads_per_rank = num_key_value_heads
        return num_kv_heads_per_rank

    def _get_hidden_dim_per_head(self, config: PretrainedConfig):
        hidden_size = config.hidden_size
        num_atten_head = config.num_attention_heads
        hidden_dim_per_head = hidden_size // num_atten_head
        return hidden_dim_per_head

    def _init_kv_shape(self, config: PretrainedConfig, neuron_config: NxDNeuronConfig):
        max_batch_size = neuron_config.max_batch_size
        max_len = neuron_config.sequence_length
        num_kv_heads_per_rank = self._get_num_kv_heads_per_rank(config, neuron_config)
        hidden_dim_per_head = self._get_hidden_dim_per_head(config)

        if self.flash_decoding_enabled:
            padded_max_len = max_len
            if max_len % self.num_cores_per_group != 0:
                padded_max_len += self.num_cores_per_group - max_len % self.num_cores_per_group
                logging.warning(
                    f"Max length needs to be multiples of num_cores_per_group {self.num_cores_per_group}"
                    f" but got {max_len}. Padding it to {padded_max_len} meet the requirement."
                )
            max_len = get_cache_size(padded_max_len, self.num_cores_per_group)

        if self.is_kv_cache_tiled:
            num_tiles = int(max_len / 128)
            # KV cache layout : BHS(128 tiled)D
            self.kv_shape = (
                max_batch_size,
                num_kv_heads_per_rank,
                128,  # Sequence dim is tiled
                num_tiles,  # max_len = 128 * num_tiles
                hidden_dim_per_head,
            )
        else:
            # KV cache layout : BHSD
            self.kv_shape = (
                max_batch_size,
                num_kv_heads_per_rank,
                max_len,
                hidden_dim_per_head,
            )

    def get_kv_by_layer_id(self, key_layer_idx, gather_index=None, slice_index=None):
        k_cache = self.past_key_values[key_layer_idx]
        v_cache = self.past_key_values[key_layer_idx + 1]
        return k_cache, v_cache

    def get_cache(self, seq_len: int, skip_slice=False, **kwargs):
        """
        Return network (all layers)'s previously cached K and V, up to seq_len.

        :param seq_len: sequence length (or bucket size from auto-bucketing e.g. 128, 512, 1024 etc.)
        :return: list of tuple of (K, V)
        """
        slice_index, gather_index = None, None
        past_key_values = []
        for key_layer_idx in range(0, len(self.past_key_values), 2):
            # get kv per layer
            k_cache, v_cache = self.get_kv_by_layer_id(
                key_layer_idx, gather_index=gather_index, slice_index=slice_index
            )

            if self.is_kv_cache_tiled:
                # We merge the tiles BHS(128 tiled)D ->  BHSD
                k_cache = k_cache.reshape(
                    self.kv_shape[0],
                    self.kv_shape[1],
                    self.kv_shape[2] * self.kv_shape[3],
                    self.kv_shape[4],
                )
                v_cache = v_cache.reshape(
                    self.kv_shape[0],
                    self.kv_shape[1],
                    self.kv_shape[2] * self.kv_shape[3],
                    self.kv_shape[4],
                )

            # slice for partial view
            if not skip_slice:
                k_cache = _slice_kv_cacheline(self.padding_side, seq_len, k_cache)
                v_cache = _slice_kv_cacheline(self.padding_side, seq_len, v_cache)

            past_key_values.append([k_cache, v_cache])
        return past_key_values

    def update_cache(
        self,
        is_for_context_encoding: bool,
        seq_ids: Tensor,
        position_ids: Tensor,
        new_key_values: List[Tensor],
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
        :param position_ids: tensor of size (batch_sz, bucket_sz)
        :param new_key_values: list of tuple, the latest kv obtained at the end of the network from forward pass
        :param seq_len: sequence length (or bucket size from auto-bucketing e.g. 128, 512, 1024 etc.)
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

            if kvcache_buffer is None:
                cache_shape = self.past_key_values[idx * 2].shape
                if self.is_kv_cache_tiled:
                    k_cache_buffer = self.past_key_values[idx * 2].reshape(
                        cache_shape[0],
                        cache_shape[1],
                        cache_shape[2] * cache_shape[3],
                        cache_shape[4],
                    )
                    v_cache_buffer = self.past_key_values[idx * 2 + 1].reshape(
                        cache_shape[0],
                        cache_shape[1],
                        cache_shape[2] * cache_shape[3],
                        cache_shape[4],
                    )
                    k_cache = k_cache_buffer
                    v_cache = v_cache_buffer
                else:
                    k_cache = self.past_key_values[idx * 2]
                    v_cache = self.past_key_values[idx * 2 + 1]

            else:
                cache_shape = kvcache_buffer[idx][0].shape
                k_cache = kvcache_buffer[idx][0]
                v_cache = kvcache_buffer[idx][1]

            if is_for_context_encoding:
                if self.is_continuous_batching:
                    # scatter back to the desired seq_ids
                    sliced_k_cache = _slice_kv_cacheline(self.padding_side, seq_len, k_cache)
                    sliced_v_cache = _slice_kv_cacheline(self.padding_side, seq_len, v_cache)

                    seq_id_index_shape = seq_ids.shape[:1] + sliced_k_cache.shape[1:]
                    seq_id_index = seq_ids.view(-1, 1, 1, 1).expand(seq_id_index_shape)

                    sliced_k_cache = torch.scatter(input=sliced_k_cache, dim=0, index=seq_id_index, src=latest_k)
                    sliced_v_cache = torch.scatter(input=sliced_v_cache, dim=0, index=seq_id_index, src=latest_v)

                    read_len = sliced_k_cache.shape[2]
                    k_cache = _gather_slice_into_kv_cacheline(k_cache, self.padding_side, read_len, sliced_k_cache)
                    v_cache = _gather_slice_into_kv_cacheline(v_cache, self.padding_side, read_len, sliced_v_cache)
                else:
                    k_cache = fill_prefix(k_cache, latest_k)
                    v_cache = fill_prefix(v_cache, latest_v)
            else:
                if self.padding_side == "left":
                    k_cache = k_cache[:, :, 1:, :]
                    v_cache = v_cache[:, :, 1:, :]
                    k_cache = torch.cat([k_cache, latest_k], dim=2)
                    v_cache = torch.cat([v_cache, latest_v], dim=2)
                else:
                    # copy the tensor of the new position into kv cache
                    if self.flash_decoding_enabled:
                        assert active_mask is not None, "active_mask should be specified for flash decoding!"
                        garbage_pos = seq_len - 1  # treat last pos as garbage
                        updated_pos_ids = position_ids // self.num_cores_per_group
                        scatter_index = torch.where(active_mask == 1, updated_pos_ids, garbage_pos)
                        scatter_index_new = scatter_index.view(-1, 1, scatter_index.shape[-1], 1).expand_as(latest_k)
                    else:
                        scatter_index_new = self._get_index_to_update_new_position(
                            scatter_index, position_ids, latest_k
                        )
                    k_cache = torch.scatter(input=k_cache, dim=2, index=scatter_index_new, src=latest_k)
                    v_cache = torch.scatter(input=v_cache, dim=2, index=scatter_index_new, src=latest_v)

            # Retiling
            # TODO once compiler fixes CR 158191111 we can turn back output tiling on
            # k_cache = k_cache.view(cache_shape)
            # v_cache = v_cache.view(cache_shape)

            updated_kv_cache.append(k_cache)
            updated_kv_cache.append(v_cache)

        # return updated kv cache to NxD runtime
        return updated_kv_cache

    def _get_index_to_update_new_position(self, scatter_index, position_ids, full_k):
        scatter_index = position_ids.view(-1, 1, position_ids.shape[-1], 1).expand_as(full_k)
        return scatter_index
