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
import math

import torch

from .. import functional


def apply_inv_frequency_scaling(freq, rope_scaling):
    scale_factor = rope_scaling.get("factor")
    low_freq_factor = rope_scaling.get("low_freq_factor")
    high_freq_factor = rope_scaling.get("high_freq_factor")
    old_context_len = rope_scaling.get("original_max_position_embeddings")

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    assert low_freq_wavelen != high_freq_wavelen

    wavelen = 2 * math.pi / freq
    smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)

    new_freq = torch.where(wavelen < high_freq_wavelen, freq, freq / scale_factor)
    smooth_cond = torch.logical_and(wavelen >= high_freq_wavelen, wavelen <= low_freq_wavelen)
    new_freq = torch.where(smooth_cond, (1 - smooth) * freq / scale_factor + smooth * freq, new_freq)
    return new_freq.to(dtype=freq.dtype)


def hlo_rotary_embedding(dtype, head_dim, cache_ids, base=10000, interpolation_factor=None, rope_scaling=None):
    scribe = cache_ids.scribe
    # Using f16 during compute causes relatively high error
    mtype = scribe.f32

    cache_ids = functional.cast(cache_ids, mtype)

    use_2d_cache_ids = len(cache_ids.sizes) > 1
    if use_2d_cache_ids:
        batch_size, n_active_tokens = cache_ids.sizes  # 2d cache_ids
        cache_ids = functional.reshape(cache_ids, [batch_size, n_active_tokens, 1])
        dot_dims = {"lhs_contracting_dimensions": [2], "rhs_contracting_dimensions": [0]}
    else:
        (n_active_tokens,) = cache_ids.sizes  # 1d cache_ids
        cache_ids = functional.reshape(cache_ids, [n_active_tokens, 1])
        dot_dims = {"lhs_contracting_dimensions": [1], "rhs_contracting_dimensions": [0]}
    size = head_dim // 2

    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2) / head_dim))
    if rope_scaling is not None and rope_scaling.get("rope_type", rope_scaling.get("type", None)) == "llama3":
        inv_freq = apply_inv_frequency_scaling(inv_freq, rope_scaling)
    inv_freq = functional.literal(mtype, inv_freq)

    if interpolation_factor:
        cache_ids = functional.divide(cache_ids, interpolation_factor)

    inv_freq = functional.reshape(inv_freq, (1, size))
    sinusoid_inp = functional.dot_general(cache_ids, inv_freq, dimension_numbers=dot_dims)

    sin = functional.sin(sinusoid_inp)
    cos = functional.cos(sinusoid_inp)
    sin = functional.cast(sin, dtype)
    cos = functional.cast(cos, dtype)
    return sin, cos


def get_up_down(q):
    """
    Given a tensor, returns its upper and lower halves (divided in the last dimension)
    """
    head_dim = q.sizes[-1]
    q_up = functional.slice_along(q, -1, head_dim // 2)
    q_down = functional.slice_along(q, -1, head_dim, head_dim // 2)
    return q_up, q_down


def get_up_down_with_percentage(q, percentage):
    """
    Given a tensor, returns its upper and lower halves with given percentage (divided in the last dimension)
    """
    head_dim = q.sizes[-1]
    q_up = functional.slice_along(q, -1, int(head_dim * percentage))
    q_down = functional.slice_along(q, -1, head_dim, int(head_dim * percentage))
    return q_up, q_down


def rotate_vec(q, sin_r, cos_r, rotary_percentage=1):
    """
    Given vectors q, sin, and cos tables, apply rotation to vectors
    """
    if rotary_percentage == 1:
        q_up, q_down = get_up_down(q)
        q_rot_up = functional.ax_minus_by(cos_r, q_up, sin_r, q_down)
        q_rot_down = functional.ax_plus_by(cos_r, q_down, sin_r, q_up)
        q_rot = functional.concatenate([q_rot_up, q_rot_down], dimension=3)
        return q_rot
    else:
        q_rotary, q_pass = get_up_down_with_percentage(q, rotary_percentage)
        q_rotary_up, q_rotary_down = get_up_down(q_rotary)
        q_rotary_rot_up = functional.ax_minus_by(cos_r, q_rotary_up, sin_r, q_rotary_down)
        q_rotary_rot_down = functional.ax_plus_by(cos_r, q_rotary_down, sin_r, q_rotary_up)
        q_rotary_rot = functional.concatenate([q_rotary_rot_up, q_rotary_rot_down], dimension=3)
        return functional.concatenate([q_rotary_rot, q_pass], dimension=3)


def rotate_half(query, key, sin_cos, rotary_percentage=1, tp_degree=None):
    """
    A secondary projection to apply to input query/key projections (used in
    specific models: GPT-J/GPT-NeoX/Llama).

    """
    n_active_tokens, n_seqs, n_kv_heads_tp, d_head = key.sizes
    _, _, n_heads_tp, _ = query.sizes

    """
        Vector approach:
        | q_up cos - q_down sin |
        | q_up sin + q_down cos |
    """
    # Rotate query and key
    broadcast_sizes = (
        n_active_tokens,
        n_seqs,
        n_heads_tp,
        int((d_head // 2) * rotary_percentage),
    )
    kv_broadcast_sizes = (
        n_active_tokens,
        n_seqs,
        n_kv_heads_tp,
        int((d_head // 2) * rotary_percentage),
    )

    def _broadcast_sin_cos(sin_cos, broadcast_sizes):
        sin, cos = sin_cos
        use_2d_cache_ids = len(sin.sizes) > 2
        if use_2d_cache_ids:
            # transpose from (n_seqs, n_active_tokens, d_head) to (n_active_tokens, n_seqs, d_head)
            sin_t = functional.transpose(sin, 0, 1)
            cos_t = functional.transpose(cos, 0, 1)
            # broadcast from (n_active_tokens, n_seqs, d_head) to (n_active_tokens, n_seqs, n_heads_tp, d_head)
            sin_r = functional.broadcast(sin_t, broadcast_sizes, [0, 1, 3])
            cos_r = functional.broadcast(cos_t, broadcast_sizes, [0, 1, 3])
        else:
            # 1D cache_ids
            sin_r = functional.broadcast(sin, broadcast_sizes, [0, 3])
            cos_r = functional.broadcast(cos, broadcast_sizes, [0, 3])
        return sin_r, cos_r

    # Get sin and cos as upper and lower half of input embedding
    sin_r, cos_r = _broadcast_sin_cos(sin_cos, broadcast_sizes)

    # Rotate query
    query = rotate_vec(query, sin_r, cos_r, rotary_percentage)

    # Get sin and cos as upper and lower half of input embedding
    kv_sin_r, kv_cos_r = _broadcast_sin_cos(sin_cos, kv_broadcast_sizes)

    # Rotate key
    key = rotate_vec(key, kv_sin_r, kv_cos_r, rotary_percentage)
    return query, key
