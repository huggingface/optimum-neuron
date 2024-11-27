from typing import Optional

import torch
from modules.autobucketing import slice_lhs, slice_rhs
from transformers import Cache


def _slice_kv_cacheline(seq_len: int, cache: torch.Tensor):
    return slice_lhs(cache, seq_len, 2)


def _gather_slice_into_kv_cacheline(cache, seq_len: int, bucket_slice: torch.Tensor):
    max_idx = cache.shape[2]
    remaining = slice_rhs(cache, max_idx - seq_len, max_idx, dim=2)
    return torch.cat([bucket_slice, remaining], dim=2)


class NeuronStaticCache(torch.nn.Module, Cache):

    def __init__(
        self,
        max_batch_size: int,
        max_length: int,
        num_kv_heads_per_partition: int,
        hidden_dim_per_head: int,
        num_hidden_layers: int,
        dtype: torch.dtype,
    ):
        super().__init__()
        kv_shape = (
            max_batch_size,
            num_kv_heads_per_partition,
            max_length,
            hidden_dim_per_head,
        )
        self.past_key_values = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.zeros(kv_shape, dtype=dtype), requires_grad=False)
                for _ in range(num_hidden_layers * 2)
            ]
        )
        self.cache_position = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.full((max_batch_size,), -1), requires_grad=False)
                for _ in range(num_hidden_layers)
            ]
        )

    def reset(self):
        for cache_position in self.cache_position:
            cache_position[:] = -1

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        return self.cache_position[layer_idx]

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length (i.e. max capacity) of the cache object"""
        return self.past_key_values[0].shape[2]

    def __hash__(self):
        return hash(self.past_key_values)

    def get_sliced_kv_cache(self, layer_idx: int, seq_len: int):
        cache_idx = 2 * layer_idx
        k_cache = slice_lhs(self.past_key_values[cache_idx], seq_len, 2)
        v_cache = slice_lhs(self.past_key_values[cache_idx + 1], seq_len, 2)
        return k_cache, v_cache

    def add(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, n_positions: int):
        # assign back to full kv_cacheline
        cache_idx = layer_idx * 2
        k_cache = _gather_slice_into_kv_cacheline(self.past_key_values[cache_idx], n_positions, key_states)
        v_cache = _gather_slice_into_kv_cacheline(self.past_key_values[cache_idx + 1], n_positions, value_states)
        return k_cache, v_cache

    def append(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        position_ids: torch.Tensor,
        layer_idx: int,
        seq_len: int,
    ):
        cache_idx = 2 * layer_idx
        sliced_k_cache, sliced_v_cache = self.get_sliced_kv_cache(layer_idx, seq_len)
        scatter_index = position_ids.view(-1, 1, position_ids.shape[-1], 1).expand_as(key_states)
        k_cache = torch.scatter(input=sliced_k_cache, dim=2, index=scatter_index, src=key_states)
        v_cache = torch.scatter(input=sliced_v_cache, dim=2, index=scatter_index, src=value_states)
        cache_idx = layer_idx * 2
        k_cache = _gather_slice_into_kv_cacheline(self.past_key_values[cache_idx], seq_len, k_cache)
        v_cache = _gather_slice_into_kv_cacheline(self.past_key_values[cache_idx + 1], seq_len, v_cache)
        return k_cache, v_cache
