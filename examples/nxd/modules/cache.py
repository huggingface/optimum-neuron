from typing import Any, Dict, Optional, Tuple

import torch
from modules.autobucketing import slice_lhs, slice_rhs
from transformers import Cache


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

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            A tuple containing the updated key and value states.
        """
        cache_position = cache_kwargs.get("cache_position")
        self.cache_position[layer_idx] = cache_position
        k_out = self.past_key_values[2 * layer_idx]
        v_out = self.past_key_values[2 * layer_idx + 1]
        # For now the actual update is done outside of this class
        # k_out[:, :, cache_position] = key_states
        # v_out[:, :, cache_position] = value_states
        return k_out, v_out

    def _gather_bucket_slice_into_kv_cacheline(self, idx, bucket_slice, padding_side, n_positions):
        max_idx = self.get_max_cache_shape()
        if padding_side == "right":
            remaining = slice_rhs(self.past_key_values[idx], max_idx - n_positions, max_idx, 2)
            return torch.cat([bucket_slice, remaining], dim=2)
        else:
            remaining = slice_lhs(self.past_key_values[idx], max_idx - n_positions, 2)
            return torch.cat([remaining, bucket_slice], dim=2)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        return self.cache_position[layer_idx]

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length (i.e. max capacity) of the cache object"""
        return self.past_key_values[0].shape[2]

    def __hash__(self):
        return hash(self.past_key_values)

    def get_past_key_values(self, layer_idx: int, padding_side, n_positions):
        cache_idx = 2 * layer_idx
        k_cache = self.past_key_values[cache_idx]
        v_cache = self.past_key_values[cache_idx + 1]

        def slice(cache, padding_side, n_positions):
            if padding_side == "right":
                return slice_lhs(cache, n_positions, 2)
            else:
                max_length = cache.shape[2]
                return slice_rhs(cache, n_positions, max_length, 2)

        return slice(k_cache, padding_side, n_positions), slice(v_cache, padding_side, n_positions)

    def add(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, padding_side: str, n_positions: int
    ):
        k_cache, v_cache = self.get_past_key_values(layer_idx, padding_side, n_positions)
        # assign back to full kv_cacheline
        k_cache = key_states
        v_cache = value_states
        k_cache = self._gather_bucket_slice_into_kv_cacheline(layer_idx * 2, k_cache, padding_side, n_positions)
        v_cache = self._gather_bucket_slice_into_kv_cacheline(layer_idx * 2 + 1, v_cache, padding_side, n_positions)
        return k_cache, v_cache

    def append(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        position_ids: torch.Tensor,
        layer_idx: int,
        padding_side: str,
        n_positions: int,
    ):
        k_cache, v_cache = self.get_past_key_values(layer_idx, padding_side, n_positions)
        if padding_side == "left":
            # TODO: fix it with scatter after right padding
            k_cache = k_cache[:, :, 1:, :]
            v_cache = v_cache[:, :, 1:, :]
            k_cache = torch.cat([k_cache, key_states], dim=2)
            v_cache = torch.cat([v_cache, value_states], dim=2)
        else:
            scatter_index_new = position_ids.view(-1, 1, position_ids.shape[-1], 1).expand_as(key_states)
            k_cache = torch.scatter(k_cache, 2, scatter_index_new, key_states)
            v_cache = torch.scatter(v_cache, 2, scatter_index_new, value_states)

        k_cache = self._gather_bucket_slice_into_kv_cacheline(layer_idx * 2, k_cache, padding_side, n_positions)
        v_cache = self._gather_bucket_slice_into_kv_cacheline(layer_idx * 2 + 1, v_cache, padding_side, n_positions)
        return k_cache, v_cache
