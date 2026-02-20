# coding=utf-8
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""
Custom NKI flash attention kernel for head dimensions > 128.

Uses d-tiling: splits head_dim into 128-element chunks for the QK matmul
(which needs par_dim <= 128) while keeping the PV matmul unchanged
(since head_dim sits in the free dimension, max 512).

The d-tiling pattern is borrowed from the backward pass kernel in
neuronxcc/nki/kernels/attention.py (flash_attn_bwd / _flash_attn_bwd_core),
which already tiles across d_head with:
    d_head_n_tiles, d_head_tile_size = div_ceil(d_head, 128), min(d_head, 128)

Reference: neuronxcc/nki/kernels/attention.py (flash_fwd, _flash_attention_core)
"""

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np
from neuronxcc import nki
from neuronxcc.nki.language import par_dim


def div_ceil(n, d):
    return (n + d - 1) // d


@nki.jit(mode="trace")
def _transpose_p_local(p_local_transposed, p_local, LARGE_TILE_SZ):
    """Transpose p_local in 128x128 blocks using nc_transpose.

    Identical to the reference kernel's transpose_p_local.
    """
    for i in nl.affine_range(LARGE_TILE_SZ // 512):
        p_local_t_tmp = nl.ndarray((par_dim(128), 512), buffer=nl.psum, dtype=np.float32)
        for j in nl.affine_range(512 // 128):
            j_128_slice = nl.ds(j * 128, 128)
            i_j_128_slice = nl.ds(i * 512 + j * 128, 128)
            p_local_t_tmp[:, j_128_slice] = nisa.nc_transpose(p_local[:, i_j_128_slice])
        p_local_transposed[:, nl.ds(i * 512, 512)] = nl.copy(p_local_t_tmp, dtype=p_local_transposed.dtype)


@nki.jit(mode="trace")
def _flash_attention_core_large_d(
    q_tiles,
    k_tiles,
    v,
    d_n_tiles,
    o_buffer,
    l_buffer,
    m_buffer,
    q_pos_base,
    k_pos_base,
    kernel_dtype,
    acc_type,
    use_causal_mask,
    sliding_window,
    B_P_SIZE,
    B_F_SIZE,
    d,
    LARGE_TILE_SZ,
):
    """
    Flash attention core with d-tiling for head_dim > 128.

    Callers must initialize m_buffer and l_buffer to a sentinel value of -1e38
    (representing -inf) before the first call.  This unified update path handles
    both the "first K tile" and "subsequent K tile" cases: when m_buffer=-1e38
    the rescaling factor alpha = exp(-1e38 - max_) ≈ 0, so the o_buffer
    contribution from previous iterations is effectively zero.

    Parameters:
        q_tiles: (d_n_tiles, par_dim(128), B_P_SIZE) -- Q tile split along d
        k_tiles: (d_n_tiles, par_dim(128), LARGE_TILE_SZ) -- K tile split along d
        v: (LARGE_TILE_SZ // B_P_SIZE, par_dim(B_P_SIZE), d) -- V tile (no d split)
        q_pos_base: base position of the Q tile (= q_tile_idx * B_P_SIZE)
        k_pos_base: base position of the K large tile (= local_k_large_tile_idx * LARGE_TILE_SZ)
    """
    num_k_tile_per_large_tile = LARGE_TILE_SZ // B_F_SIZE

    qk_res_buf = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ), buffer=nl.sbuf, dtype=acc_type)
    max_local = nl.ndarray((par_dim(B_P_SIZE), num_k_tile_per_large_tile), dtype=acc_type)

    for k_i in nl.affine_range(num_k_tile_per_large_tile):
        k_i_b_f_slice = nl.ds(k_i * B_F_SIZE, B_F_SIZE)

        # D-tiled QK matmul: accumulate over d_n_tiles
        qk_psum = nl.zeros((par_dim(B_P_SIZE), B_F_SIZE), dtype=np.float32, buffer=nl.psum)
        for dt in nl.affine_range(d_n_tiles):
            qk_psum[:, :] += nl.matmul(q_tiles[dt], k_tiles[dt, :, k_i_b_f_slice], transpose_x=True)

        # Causal mask + sliding window
        if use_causal_mask:
            i_q_p, i_q_f = nl.mgrid[0:B_P_SIZE, 0:B_F_SIZE]
            # Use pre-computed position bases to avoid runtime multiplications
            q_pos = q_pos_base + i_q_p
            k_pos = k_pos_base + k_i * B_F_SIZE + i_q_f

            # Apply causal mask: position i can attend to position j if i >= j
            causal_pred = q_pos >= k_pos
            qk_select_tmp = nl.ndarray(qk_psum.shape, dtype=qk_psum.dtype, buffer=nl.sbuf)
            qk_select_tmp[...] = qk_psum
            qk_masked = nisa.affine_select(
                pred=causal_pred,
                on_true_tile=qk_select_tmp,
                on_false_value=-9984.0,
                dtype=acc_type,
            )

            if sliding_window > 0:
                # Apply sliding window: position i can attend to j if i - j < window
                # Rewrite: (q_pos - k_pos) < sliding_window
                #        = k_pos > q_pos - sliding_window
                #        = k_pos >= q_pos - sliding_window + 1
                #        = k_pos - q_pos + sliding_window - 1 >= 0
                window_pred = k_pos >= (q_pos - sliding_window + 1)
                qk_masked = nisa.affine_select(
                    pred=window_pred,
                    on_true_tile=qk_masked,
                    on_false_value=-9984.0,
                    dtype=acc_type,
                )

            qk_res_buf[:, k_i_b_f_slice] = qk_masked
        else:
            qk_res_buf[:, k_i_b_f_slice] = nl.copy(qk_psum, dtype=acc_type)

        # Max for softmax
        max_local[:, k_i] = nisa.tensor_reduce(
            np.max, qk_res_buf[:, k_i_b_f_slice], axis=(1,), dtype=acc_type, negate=False
        )

    # Online softmax: update running max and rescale accumulated output.
    # When m_buffer is at its sentinel value (-1e38), alpha = exp(-1e38 - max_) ≈ 0
    # so the rescaling of o_buffer is a no-op on the first valid K tile.
    max_ = nisa.tensor_reduce(np.max, max_local[:, :], axis=(1,), dtype=acc_type, negate=False)

    m_previous = nl.copy(m_buffer[:, 0])
    m_buffer[:, 0] = nl.maximum(m_previous, max_)
    m_current = nl.ndarray((par_dim(B_P_SIZE), 1), dtype=acc_type, buffer=nl.sbuf)
    m_current[...] = m_buffer[:, 0]
    alpha = nisa.activation(np.exp, m_current, bias=m_previous, scale=-1.0)
    o_previous_scaled = nl.ndarray((par_dim(B_P_SIZE), d), dtype=o_buffer.dtype)
    o_previous_scaled[...] = nl.multiply(o_buffer[:, :], alpha)

    # Compute exp(QK - max) and partial sums
    p_local = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
    REDUCTION_TILE = min(2048, LARGE_TILE_SZ // 2)
    p_partial_sum = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ // REDUCTION_TILE), dtype=acc_type)

    for k_r_i in nl.affine_range(LARGE_TILE_SZ // REDUCTION_TILE):
        k_r_i_reduce_slice = nl.ds(k_r_i * REDUCTION_TILE, REDUCTION_TILE)
        p_local[:, k_r_i_reduce_slice] = nisa.activation_reduce(
            np.exp,
            qk_res_buf[:, k_r_i_reduce_slice],
            bias=-1 * m_current,
            scale=1.0,
            reduce_op=nl.add,
            reduce_res=p_partial_sum[:, k_r_i],
            dtype=kernel_dtype,
        )

    ps = nl.sum(p_partial_sum, axis=1, dtype=acc_type)

    # Transpose p_local for PV matmul
    p_local_transposed = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
    _transpose_p_local(
        p_local_transposed=p_local_transposed,
        p_local=p_local,
        LARGE_TILE_SZ=LARGE_TILE_SZ,
    )

    # PV matmul -- NO d-tiling needed (d in free dimension, <=512)
    pv_psum = nl.zeros((par_dim(B_P_SIZE), d), dtype=np.float32, buffer=nl.psum, lazy_initialization=True)
    for k_i in nl.affine_range(LARGE_TILE_SZ // B_P_SIZE):
        pv_psum[:, :] += nl.matmul(
            p_local_transposed[:, nl.ds(k_i * B_P_SIZE, B_P_SIZE)],
            v[k_i, :, :],
            transpose_x=True,
        )

    # Update output accumulator and log-sum-exp
    o_buffer[:, :] = nl.add(o_previous_scaled, pv_psum)
    exp = nisa.activation(nl.exp, m_current, bias=l_buffer[:, 0], scale=-1.0)
    l_buffer[:, 0] = nl.add(m_current, nisa.activation(nl.log, exp, bias=ps))


@nki.jit
def flash_fwd_large_d(
    q,
    k,
    v,
    softmax_scale=1.0,
    output=None,
    use_causal_mask=True,
    sliding_window=0,
):
    """
    Flash attention forward pass for head dimensions that are multiples of 128.

    Uses d-tiling: splits head_dim into 128-element chunks for the QK matmul
    while keeping the PV matmul unchanged (d sits in the free dimension, max 512).

    IO tensor layouts:
        q: (BH, D, S) -- batch*heads, head_dim, seq_len
        k: (BH, D, S)
        v: (BH, S, D)
        output: (BH, D, S) -- pre-allocated output tensor

    Parameters:
        softmax_scale: scaling factor (typically 1.0, caller pre-scales Q)
        output: pre-allocated output tensor (BH, D, S)
        use_causal_mask: whether to apply causal mask
        sliding_window: sliding window size (0 = no window)
    """
    bh, d, seqlen = q.shape
    _, _, seqlen_k = k.shape

    assert d % 128 == 0, f"head_dim must be divisible by 128, got {d}"
    assert d <= 512, f"head_dim must be <= 512, got {d}"
    assert seqlen == seqlen_k, f"seq_q ({seqlen}) must equal seq_k ({seqlen_k})"

    B_P_SIZE = 128
    B_F_SIZE = 512
    B_D_SIZE = 128  # Head dim tile size
    d_n_tiles = d // B_D_SIZE

    kernel_dtype = nl.bfloat16
    acc_type = np.dtype(np.float32)

    # Scale tile sizes inversely with d to keep SBUF usage manageable.
    # LARGE_TILE_SZ controls qk_res_buf, p_local, p_local_transposed, k_tiles, v_tile.
    # For d>128 the extra d_n_tiles multiply SBUF; reduce LARGE_TILE_SZ to compensate.
    LARGE_TILE_SZ = 2048 if d_n_tiles == 1 else 1024
    # o_buffer = (attn_core_tile_size, par_dim(128), d) in fp32.
    attn_core_tile_size = max(4, 64 // d_n_tiles)

    assert LARGE_TILE_SZ >= 512, f"LARGE_TILE_SZ ({LARGE_TILE_SZ}) must be >= 512"
    assert seqlen % LARGE_TILE_SZ == 0, f"seq_len ({seqlen}) must be divisible by {LARGE_TILE_SZ}"

    # SPMD initialization (following Flux kernel pattern)
    n_prgs = 1
    prg_id = 0
    grid_ndim = nl.program_ndim()
    assert grid_ndim <= 1, f"Expected 0 or 1 grid dimensions, got {grid_ndim}"
    if grid_ndim > 0 and nl.num_programs(axes=0) > 1:
        n_prgs = nl.num_programs(axes=0)
        prg_id = nl.program_id(axis=0)

    n_tile_q = seqlen // B_P_SIZE
    num_large_k_tile = seqlen // LARGE_TILE_SZ
    n_remat = div_ceil(n_tile_q, attn_core_tile_size)
    attn_core_tile_size = min(n_tile_q, attn_core_tile_size)

    # Iterate over batch*head items assigned to this program (blocked distribution)
    bh_per_prg = bh // n_prgs

    for bh_i in nl.sequential_range(bh_per_prg):
        bh_id = prg_id * bh_per_prg + bh_i

        # Global Flash Attention accumulators.
        # Initialized to -1e38 (sentinel for -inf) so that the unified update
        # path in _flash_attention_core_large_d works correctly for every K tile,
        # including when sliding_window causes the first few K-large-tiles to be
        # skipped for high-index Q tiles.  With m=-1e38, the rescale factor
        # alpha = exp(-1e38 - max_) ≈ 0, giving a clean "first iteration" result
        # without a separate initialize flag.
        l_buffer = nl.full(
            (par_dim(B_P_SIZE), n_tile_q),
            fill_value=-1e38,
            dtype=acc_type,
            buffer=nl.sbuf,
        )

        for i0 in nl.sequential_range(n_remat):
            o_buffer = nl.zeros(
                (attn_core_tile_size, par_dim(B_P_SIZE), d),
                dtype=acc_type,
                buffer=nl.sbuf,
            )
            m_buffer = nl.full(
                (attn_core_tile_size, par_dim(B_P_SIZE), 1),
                fill_value=-1e38,
                dtype=acc_type,
                buffer=nl.sbuf,
            )

            for j in nl.sequential_range(0, num_large_k_tile):
                # Load K tiles: split d into d_n_tiles chunks
                cur_k_tiles = nl.ndarray(
                    (d_n_tiles, par_dim(B_D_SIZE), LARGE_TILE_SZ),
                    dtype=kernel_dtype,
                )
                for dt in nl.affine_range(d_n_tiles):
                    cur_k_tiles[dt, :, :] = nl.load(
                        k[bh_id, nl.ds(dt * B_D_SIZE, B_D_SIZE), nl.ds(j * LARGE_TILE_SZ, LARGE_TILE_SZ)]
                    )

                # Load V tile: no d-split (d in free dim, <=512)
                cur_v_tile = nl.ndarray(
                    (LARGE_TILE_SZ // B_P_SIZE, par_dim(B_P_SIZE), d),
                    dtype=kernel_dtype,
                )
                for v_i in nl.affine_range(LARGE_TILE_SZ // B_P_SIZE):
                    cur_v_tile[v_i, :, :] = nl.load(
                        v[bh_id, nl.ds(j * LARGE_TILE_SZ + B_P_SIZE * v_i, B_P_SIZE), :],
                        dtype=kernel_dtype,
                    )

                for i1 in nl.affine_range(attn_core_tile_size):
                    i = i0 * attn_core_tile_size + i1

                    if use_causal_mask:
                        forward_mask = i * B_P_SIZE >= j * LARGE_TILE_SZ
                        if sliding_window > 0:
                            # Also skip LARGE tiles entirely outside the window
                            k_large_max_pos = (j + 1) * LARGE_TILE_SZ - 1
                            forward_mask = forward_mask & ((i * B_P_SIZE - k_large_max_pos) < sliding_window)
                    else:
                        forward_mask = True

                    if (i < n_tile_q) & forward_mask:
                        # Load Q tiles: split d into d_n_tiles chunks
                        q_tiles = nl.ndarray(
                            (d_n_tiles, par_dim(B_D_SIZE), B_P_SIZE),
                            dtype=kernel_dtype,
                        )
                        for dt in nl.affine_range(d_n_tiles):
                            q_sbuf = nl.load(
                                q[bh_id, nl.ds(dt * B_D_SIZE, B_D_SIZE), nl.ds(i * B_P_SIZE, B_P_SIZE)],
                                dtype=kernel_dtype,
                            )
                            q_tiles[dt, :, :] = q_sbuf * softmax_scale

                        _flash_attention_core_large_d(
                            q_tiles=q_tiles,
                            k_tiles=cur_k_tiles,
                            v=cur_v_tile,
                            d_n_tiles=d_n_tiles,
                            o_buffer=o_buffer[i1],
                            l_buffer=l_buffer[:, i],
                            m_buffer=m_buffer[i1],
                            q_pos_base=i * B_P_SIZE,
                            k_pos_base=j * LARGE_TILE_SZ,
                            kernel_dtype=kernel_dtype,
                            acc_type=acc_type,
                            use_causal_mask=use_causal_mask,
                            sliding_window=sliding_window,
                            B_P_SIZE=B_P_SIZE,
                            B_F_SIZE=B_F_SIZE,
                            d=d,
                            LARGE_TILE_SZ=LARGE_TILE_SZ,
                        )

            # Write output with transpose: (S_tile, D) -> (D, S_tile)
            for i1 in nl.affine_range(attn_core_tile_size):
                i = i0 * attn_core_tile_size + i1

                if i < n_tile_q:
                    exp = nisa.activation(
                        np.exp,
                        l_buffer[:, i],
                        bias=m_buffer[i1, :, :],
                        scale=-1.0,
                    )
                    out = nl.multiply(o_buffer[i1, :, :], exp, dtype=kernel_dtype)

                    # out: (par_dim(128), d) = (S_tile, D)
                    # Store as (D, S_tile) via nc_transpose in 128x128 blocks
                    for dt in nl.affine_range(d_n_tiles):
                        out_transposed = nisa.nc_transpose(out[:, nl.ds(dt * B_D_SIZE, B_D_SIZE)])
                        nl.store(
                            output[bh_id, nl.ds(dt * B_D_SIZE, B_D_SIZE), nl.ds(i * B_P_SIZE, B_P_SIZE)],
                            out_transposed,
                        )
