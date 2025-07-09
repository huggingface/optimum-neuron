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
# Adapted from https://github.com/aws-neuron/neuronx-distributed-inference/blob/9993358ce052fd7a1bb4a7497a6318aac36ed95c/src/neuronx_distributed_inference/modules/attention/utils.py
import math
from typing import Any

import torch
import torch_xla.core.xla_model as xm
from neuronx_distributed.parallel_layers.parallel_state import (
    get_kv_shared_group,
    get_tensor_model_parallel_group,
)
from neuronx_distributed.parallel_layers.utils import get_padding_length
from torch import Tensor, nn


torch.manual_seed(0)

weight_cache = {}


def _get_weight_from_state_dict(prefix: str, state_dict: dict[str, Any]) -> torch.Tensor:
    if prefix in weight_cache:
        return weight_cache[prefix]

    if (prefix + "weight") in state_dict:
        transposed_weight = state_dict[prefix + "weight"].t()
        weight_cache[prefix] = transposed_weight
        return transposed_weight

    else:
        raise RuntimeError(f"Cannot find {(prefix + 'weight')} in the state_dict")


def _set_weight_to_state_dict(prefix: str, tensor: torch.Tensor, state_dict: dict[str, Any]) -> None:
    if (prefix + "weight") in state_dict:
        state_dict[prefix + "weight"] = tensor.t()
    else:
        raise RuntimeError(f"Cannot find {(prefix + 'weight')} in the state_dict")


def transpose_parallel_linear_layer(parallel_layer):
    """
    This function clones and transposes a ColumnParallelLinear or RowParallelLinear
    The attributes are also cloned and partition_dim is updated
    """
    orig_attrs = vars(parallel_layer)
    new_layer = torch.nn.Parameter(parallel_layer.clone().T, requires_grad=False)
    new_layer.__dict__.update(orig_attrs)
    # flip the partition_dim from 0->1 or 1->0
    setattr(new_layer, "partition_dim", 1 - getattr(new_layer, "partition_dim"))
    setattr(new_layer, "get_tensor_from_state_dict", _get_weight_from_state_dict)
    setattr(new_layer, "set_tensor_to_state_dict", _set_weight_to_state_dict)
    return new_layer


def pad_to_128_multiple(x, dim):
    # Strided padding for unsharded weight, so after sharding
    # each rank will have dense padding at the end.
    # Eg orig shape = [16384, 53248], with dim = 1
    # We reshape to [16384, 128, 416] (TP_degree = 128)
    # Then pad to [16384, 128, 512].
    # Then collapse the original dim [16384, 65536].
    TP_DEGREE = get_tensor_model_parallel_group().size()
    orig_shape = x.shape
    new_shape = list(x.shape)
    new_shape[dim] = orig_shape[dim] // TP_DEGREE
    new_shape.insert(dim, TP_DEGREE)
    x = x.reshape(new_shape)
    dim += 1
    padding_length = get_padding_length(x.shape[dim], 128)
    dimlist = [0] * (len(x.shape) * 2)
    dimlist[dim * 2] = padding_length
    padded = torch.nn.functional.pad(x, tuple(dimlist[::-1]))
    new_padded_shape = list(orig_shape)
    new_padded_shape[dim - 1] = -1
    padded = padded.reshape(new_padded_shape)
    return padded


def move_heads_front(tensor: Tensor, bsz: int, seq_len: int, num_head: int, head_dim: int, layernorm=None) -> Tensor:
    """Reshape input tensor: BSHD -> BHSD, and apply layer normalization if layernorm is specified"""
    tensor = tensor.view(bsz, seq_len, num_head, head_dim)
    if layernorm:
        tensor = layernorm(tensor)
    return tensor.transpose(1, 2).contiguous()


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _rotate_half(x) -> Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_scaling(freqs: torch.Tensor):
    # Values obtained from grid search, specifically for Llama3.2 MM PyTorch Implementation
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    return freqs


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1) -> tuple[Tensor, Tensor]:
    """Applies Rotary Position Embedding to the query and key tensors."""

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_polar_compatible(query, key, freqs_cis):
    freqs_cis_real = freqs_cis.cos().unsqueeze(2)
    freqs_cis_imag = freqs_cis.sin().unsqueeze(2)

    def rotate(input):
        real = input[..., ::2]
        imag = input[..., 1::2]

        # For complex multiplication
        # (a + ib) * (c + id) = (ac - bd) + i(ad + bc)

        # ac - bd
        rot_real = (real * freqs_cis_real) - (imag * freqs_cis_imag)

        # ad + bc
        rot_imag = (real * freqs_cis_imag) + (freqs_cis_real * imag)

        return torch.cat([rot_real.unsqueeze(-1), rot_imag.unsqueeze(-1)], dim=-1).reshape(input.shape)

    query_rot = rotate(query)
    key_rot = rotate(key)

    return query_rot.type_as(query), key_rot.type_as(key)


def manual_softmax(prior_scores, active_scores, is_speculation) -> tuple[Tensor, Tensor]:
    """
    simple softmax computation: denominator is the sum of exp over all vocab and only need compute numerator (exp)
    """
    max_score = torch.max(prior_scores, dim=-1, keepdim=True)[0]
    max_active_score = torch.max(active_scores, dim=-1, keepdim=True)[0]
    max_score = (
        torch.maximum(max_score, max_active_score) if is_speculation else torch.maximum(max_score, active_scores)
    )

    exp_prior = torch.exp(prior_scores - max_score)
    exp_active = torch.exp(active_scores - max_score)
    denominator = exp_prior.sum(dim=-1, keepdim=True) + exp_active.sum(dim=-1, keepdim=True)

    softmax_prior = exp_prior / denominator
    softmax_active = exp_active / denominator
    return softmax_prior, softmax_active


def distributed_softmax(prior_scores, active_scores) -> tuple[Tensor, Tensor]:
    """
    compute partial softmax and then gather and correct final softmax.
    """
    # find local max
    max_score = torch.max(prior_scores, dim=-1, keepdim=True)[0]
    max_active_score = torch.max(active_scores, dim=-1, keepdim=True)[0]
    local_max_score = torch.maximum(max_score, max_active_score)

    exp_prior = torch.exp(prior_scores - local_max_score)
    exp_active = torch.exp(active_scores - local_max_score)
    denominator = exp_prior.sum(dim=-1, keepdim=True) + exp_active.sum(dim=-1, keepdim=True)

    # collect for global max and exp sum (denominator)
    groups = get_kv_shared_group(as_list=True)
    gather_payload = torch.cat((local_max_score, denominator), dim=0)
    gathered_res = xm.all_gather(gather_payload, dim=-1, groups=groups, pin_layout=False)
    gathered_max, gathered_denom = torch.chunk(gathered_res, 2, dim=0)
    global_max = torch.max(gathered_max, dim=-1, keepdim=True)[0]

    # softmax correction
    scaling_factor = torch.exp(gathered_max - global_max.expand(gathered_max.shape))
    corrected_denominator = torch.multiply(scaling_factor, gathered_denom)
    corrected_denominator = torch.sum(corrected_denominator, dim=-1, keepdim=True)

    corrected_exp_prior = torch.exp(prior_scores - global_max)
    corrected_exp_active = torch.exp(active_scores - global_max)

    softmax_prior = corrected_exp_prior / corrected_denominator
    softmax_active = corrected_exp_active / corrected_denominator
    return softmax_prior, softmax_active


class RotaryEmbedding(nn.Module):
    """
    Adapted from Llama 4.0 impl https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models
    /llama/modeling_llama.py#L96-L145
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.register_buffer("inv_freq", None, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.inv_freq is None:
            self.inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
            )
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Utility functions to create attention mask
def create_block_diagonal_attn_mask(
    query_lens: torch.Tensor,
    key_lens: torch.Tensor,
    max_query_len: torch.Tensor,
    max_key_len: torch.Tensor,
):
    """
    Return a block diagonal attention mask which can be used by chunked
    prefill.

    This function is written in a way that it can be traced, so it can
    be used inside the NeuronDecoderModel class.

    Example:
        query_lens = [2,3,1,0]
        key_lens = [4,5,4,0]
        max_query_len = 8
        max_key_len = 16

        mask = [
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # At position 3 attend to 1st sequence
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # At position 4 attend to 1st sequence
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # At position 3 attend to 2nd sequence
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], # At position 4 attend to 2nd sequence
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], # At position 5 attend to 2nd sequence
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0], # At position 3 attend to 3rd sequence
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # padding
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # padding
        ]
    Args:
        query_lens: a list of query lengths for each sequence
        key_lens: a list of key lengths for each sequence
        max_query_len: the max value of the sum of query lengths
        max_key_len: the max value of the sum of key lengths

    Return:
        mask: the causal attention mask for chunked prefill
    """
    batch_size = query_lens.shape[0]

    row_idx = torch.arange(max_query_len, dtype=torch.int).reshape(-1, 1)
    col_idx = torch.arange(max_key_len, dtype=torch.int).reshape(1, -1)

    q_cumsum = torch.cumsum(query_lens, dim=0)
    q_cumsum = torch.cat([torch.tensor(0).reshape(1), q_cumsum])
    k_cumsum = torch.cumsum(key_lens, dim=0)
    k_cumsum = torch.cat([torch.tensor(0).reshape(1), k_cumsum])

    mask = torch.zeros(max_query_len, max_key_len, dtype=torch.bool)
    for seq_id in range(batch_size):
        ri = q_cumsum[seq_id]  # row index
        ci = k_cumsum[seq_id]  # column index
        nr = query_lens[seq_id]  # number of rows
        nc = key_lens[seq_id]  # number of columns

        offset = ci + nc - ri - nr
        # upper right triangle is set to false
        diagonal_mask = (row_idx - col_idx + offset) >= 0

        left_mask = col_idx >= ci
        top_mask = row_idx >= ri
        bottom_mask = row_idx < ri + nr

        mask_per_seq = diagonal_mask & left_mask & top_mask & bottom_mask
        mask = mask | mask_per_seq

    return mask
