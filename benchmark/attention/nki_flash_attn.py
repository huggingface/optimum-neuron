"""
NKI flash attention kernel with standard transformers tensor layouts.

This kernel is inspired by neuronxcc.nki.kernels.attention.flash_fwd but
accepts the tensor layouts produced by HuggingFace transformers models:

    Q: (batch, n_heads, seq_q, d)       — standard, not transposed
    K: (batch, nk_heads, seq_k, d)      — standard, not transposed
    V: (batch, nv_heads, seq_v, d)      — standard (same as flash_fwd with should_transpose_v=False)
    attn_mask: (batch, 1, seq_q, seq_k) — per-batch, broadcast over heads
    output: (batch, n_heads, seq_q, d)

Compared to flash_fwd which requires Q, K in transposed (d, seq) layout and
restricts the mask to (1, 1, seq_q, seq_k), this kernel handles the layout
conversion internally via DMA transpose during tile loads.

The flash attention algorithm (tiling, online softmax, GQA) is identical to
flash_fwd. The core computation (_flash_attention_core) and helper functions
(transpose_p_local, dropout_p_local) are copied verbatim.

Inference only — no dropout, no LSE, no training path.
"""

from dataclasses import dataclass

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np
from neuronxcc import nki
from neuronxcc.nki.language import par_dim


def _div_ceil(n, d):
    return (n + d - 1) // d


@dataclass(frozen=True)
class _FlashConfig:
    """Minimal config for the flash attention core (inference only)."""

    seq_tile_size: int = 2048
    attn_core_tile_size: int = 256


# ── Helpers copied verbatim from neuronxcc.nki.kernels.attention ─────────
#
# These are internal to flash_fwd. We copy them here so the kernel is
# self-contained and can evolve independently.


@nki.jit(mode="trace")
def transpose_p_local(p_local_transposed, p_local, LARGE_TILE_SZ, use_dma_transpose=False):
    for i in nl.affine_range(LARGE_TILE_SZ // 512):
        # Temporarily disable use_dma_tranpose by default until we stablized it
        if use_dma_transpose and nisa.get_nc_version() >= nisa.nc_version.gen3:
            p_local_t_tmp = nl.ndarray((par_dim(128), 512), buffer=nl.sbuf, dtype=p_local.dtype)
        else:
            p_local_t_tmp = nl.ndarray((par_dim(128), 512), buffer=nl.psum, dtype=np.float32)

        for j in nl.affine_range(512 // 128):
            j_128_slice = nl.ds(j * 128, 128)
            i_j_128_slice = nl.ds(i * 512 + j * 128, 128)

            if use_dma_transpose and nisa.get_nc_version() >= nisa.nc_version.gen3:
                p_local_t_tmp[:, j_128_slice] = nisa.dma_transpose(p_local[:, i_j_128_slice])
            else:
                p_local_t_tmp[:, j_128_slice] = nisa.nc_transpose(p_local[:, i_j_128_slice])

        p_local_transposed[:, nl.ds(i * 512, 512)] = nl.copy(p_local_t_tmp, dtype=p_local_transposed.dtype)


@nki.jit(mode="trace")
def dropout_p_local(p_local, dropout_p, dropout_p_tensor, seed_tensor, seed_offset_base, k_r_i, REDUCTION_TILE):
    B_F_SIZE = 512
    for k_d_i in nl.sequential_range(REDUCTION_TILE // B_F_SIZE):
        p_local_f_slice = nl.ds(k_r_i * REDUCTION_TILE + k_d_i * B_F_SIZE, B_F_SIZE)

        offset = k_d_i + seed_offset_base
        offset_seed = nl.add(seed_tensor, offset, dtype=nl.int32)
        nl.random_seed(seed=offset_seed)
        softmax_dropout = nl.dropout(p_local[:, p_local_f_slice], rate=dropout_p_tensor[:, 0])
        p_local[:, p_local_f_slice] = nl.multiply(softmax_dropout, 1 / (1 - dropout_p))


@nki.jit(mode="trace")
def _flash_attention_core(
    q_local_tile,
    k,
    v,
    q_h_per_k_h,
    seqlen_q,
    nheads,
    o_buffer,
    l_buffer,
    m_buffer,
    batch_id,
    head_id,
    gqa_head_idx,
    q_tile_idx,
    local_k_large_tile_idx,
    kernel_dtype,
    acc_type,
    flash_config,
    use_causal_mask,
    sliding_window,
    B_P_SIZE=128,
    B_F_SIZE=512,
    B_D_SIZE=128,
    dropout_p=0.0,
    dropout_p_tensor=None,
    seed_tensor=None,
    logit_bias_tile=None,
):
    """
    The flash attention core function to calcualte self attention between a tile of q and a block of K and V.
    The q_local_tile has (B_P_SIZE, B_F_SIZE), which is loaded into the SBUF already. The block size of K and V
    is defined in the seq_tile_size of the flash_config. The results are stored in the following three buffers
    o_buffer: (B_P_SIZE, d)
    l_buffer: (B_P_SIZE, 1)
    m_buffer: (B_P_SIZE, 1)
    """
    NEG_INFINITY = nl.fp32.min
    LARGE_TILE_SZ = flash_config.seq_tile_size
    num_k_tile_per_large_tile = LARGE_TILE_SZ // B_F_SIZE
    seqlen_k = k.shape[-1]
    seq_q_num_tiles = seqlen_q // B_P_SIZE
    seq_k_num_tiles = seqlen_k // B_F_SIZE

    qk_res_buf = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ), buffer=nl.sbuf, dtype=acc_type)
    max_local = nl.ndarray((par_dim(B_P_SIZE), num_k_tile_per_large_tile), dtype=acc_type)

    for k_i in nl.affine_range(num_k_tile_per_large_tile):
        k_i_b_f_slice = nl.ds(k_i * B_F_SIZE, B_F_SIZE)

        qk_psum = nl.ndarray((par_dim(B_P_SIZE), B_F_SIZE), dtype=np.float32, buffer=nl.psum)  # (128, 512)
        if use_causal_mask:
            multiplication_required_selection = (
                q_tile_idx * B_P_SIZE >= local_k_large_tile_idx * LARGE_TILE_SZ + k_i * B_F_SIZE
            )
        else:
            multiplication_required_selection = True

        if multiplication_required_selection:
            qk_psum[:, :] = nl.matmul(q_local_tile, k[:, k_i_b_f_slice], transpose_x=True)  # (p(128), 512)
        else:
            qk_psum[:, :] = 0

        if use_causal_mask:
            left_diagonal_selection = (
                q_tile_idx * B_P_SIZE >= local_k_large_tile_idx * LARGE_TILE_SZ + (k_i + 1) * B_F_SIZE
            )
            right_diagonal_selection = (
                q_tile_idx + 1
            ) * B_P_SIZE <= local_k_large_tile_idx * LARGE_TILE_SZ + k_i * B_F_SIZE
            diagonal_and_left_selection = (
                q_tile_idx + 1
            ) * B_P_SIZE > local_k_large_tile_idx * LARGE_TILE_SZ + k_i * B_F_SIZE
            diagonal = (q_tile_idx * B_P_SIZE < local_k_large_tile_idx * LARGE_TILE_SZ + (k_i + 1) * B_F_SIZE) & (
                (q_tile_idx + 1) * B_P_SIZE > local_k_large_tile_idx * LARGE_TILE_SZ + k_i * B_F_SIZE
            )

            i_q_p, i_q_f = nl.mgrid[0:B_P_SIZE, 0:B_F_SIZE]
            q_pos = q_tile_idx * B_P_SIZE + i_q_p
            k_pos = local_k_large_tile_idx * LARGE_TILE_SZ + k_i * B_F_SIZE + i_q_f
            pred_causal = q_pos >= k_pos  # causal mask
            pred_sliding = k_pos > q_pos - sliding_window  # sliding window mask

            qk_select_tmp = nl.ndarray(qk_psum.shape, dtype=qk_psum.dtype, buffer=nl.sbuf)

            if logit_bias_tile is not None:
                if right_diagonal_selection:
                    qk_select_tmp[...] = qk_psum

                    # For tiles to the right of the diagonal, do affine_select.
                    qk_res_buf[:, k_i_b_f_slice] = nisa.affine_select(
                        pred=pred_causal, on_true_tile=qk_select_tmp, on_false_value=NEG_INFINITY, dtype=acc_type
                    )

                # For tiles on the diagonal, add logit bias and need to do affine_select.
                intermediate = nl.add(qk_psum, logit_bias_tile[:, k_i_b_f_slice], dtype=acc_type, mask=diagonal)
                qk_res_buf[:, k_i_b_f_slice] = nisa.affine_select(
                    pred=pred_causal,
                    on_true_tile=intermediate,
                    on_false_value=NEG_INFINITY,
                    dtype=acc_type,
                    mask=diagonal,
                )

                # For tiles on the left of the diagonal, add logit bias.
                qk_res_buf[:, k_i_b_f_slice] = nl.add(
                    qk_psum, logit_bias_tile[:, k_i_b_f_slice], dtype=acc_type, mask=left_diagonal_selection
                )

                if sliding_window > 0:  # Apply sliding window mask
                    qk_res_buf[:, k_i_b_f_slice] = nisa.affine_select(
                        pred=pred_sliding,
                        on_true_tile=intermediate,
                        on_false_value=NEG_INFINITY,
                        dtype=acc_type,
                        mask=left_diagonal_selection,
                    )
            else:
                # Apply causal mask
                qk_res_buf[:, k_i_b_f_slice] = nisa.affine_select(
                    pred=pred_causal, on_true_tile=qk_psum, on_false_value=NEG_INFINITY, dtype=acc_type
                )
                if sliding_window > 0:  # Apply sliding window mask
                    qk_res_buf[:, k_i_b_f_slice] = nisa.affine_select(
                        pred=pred_sliding,
                        on_true_tile=qk_res_buf[:, k_i_b_f_slice],
                        on_false_value=NEG_INFINITY,
                        dtype=acc_type,
                        mask=diagonal_and_left_selection,
                    )
        else:
            if logit_bias_tile is not None:
                # Simply add logit bias which copies back to sbuf at the same time
                qk_res_buf[:, k_i_b_f_slice] = nl.add(qk_psum, logit_bias_tile[:, k_i_b_f_slice], dtype=acc_type)
            else:
                # Simply send psum result back to sbuf
                qk_res_buf[:, k_i_b_f_slice] = nl.copy(qk_psum, dtype=acc_type)

        # Calculate max of the current tile
        max_local[:, k_i] = nisa.tensor_reduce(
            np.max, qk_res_buf[:, k_i_b_f_slice], axis=(1,), dtype=acc_type, negate=False
        )

    max_ = nisa.tensor_reduce(np.max, max_local[:, :], axis=(1,), dtype=acc_type, negate=False)

    o_previous_scaled = nl.ndarray((par_dim(B_P_SIZE), B_D_SIZE), dtype=o_buffer.dtype)

    m_previous = nl.copy(m_buffer[:, 0])
    m_buffer[:, 0] = nl.maximum(m_previous, max_)  # (128,1)

    m_current = m_buffer[:, 0]
    # Compute scaling factor
    alpha = nisa.activation(np.exp, m_current, bias=m_previous, scale=-1.0)
    o_previous_scaled[...] = nl.multiply(o_buffer[:, :], alpha)

    p_local = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
    REDUCTION_TILE = min(2048, LARGE_TILE_SZ // 2)

    p_partial_sum = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ // REDUCTION_TILE), dtype=acc_type)

    for k_r_i in nl.affine_range(LARGE_TILE_SZ // REDUCTION_TILE):
        k_r_i_reduce_slice = nl.ds(k_r_i * REDUCTION_TILE, REDUCTION_TILE)
        # compute exp(qk-max)
        # Compute partial row-tile sum of exp(qk-max))
        p_local[:, k_r_i_reduce_slice] = nisa.activation_reduce(
            np.exp,
            qk_res_buf[:, k_r_i_reduce_slice],
            bias=-1 * m_current,
            scale=1.0,
            reduce_op=nl.add,
            reduce_res=p_partial_sum[:, k_r_i],
            dtype=kernel_dtype,
        )

        # dropout
        if dropout_p > 0.0:
            seed_offset_base = (
                k_r_i * (REDUCTION_TILE // B_F_SIZE)
                + local_k_large_tile_idx * (LARGE_TILE_SZ // B_F_SIZE)
                + q_tile_idx * seq_k_num_tiles
                + (head_id * q_h_per_k_h + gqa_head_idx) * seq_k_num_tiles * seq_q_num_tiles
                + batch_id * nheads * seq_k_num_tiles * seq_q_num_tiles
            )

            dropout_p_local(
                p_local=p_local,
                dropout_p=dropout_p,
                dropout_p_tensor=dropout_p_tensor,
                seed_tensor=seed_tensor,
                seed_offset_base=seed_offset_base,
                k_r_i=k_r_i,
                REDUCTION_TILE=REDUCTION_TILE,
            )

    ps = nl.sum(p_partial_sum, axis=1, dtype=acc_type)

    p_local_transposed = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
    transpose_p_local(p_local_transposed=p_local_transposed, p_local=p_local, LARGE_TILE_SZ=LARGE_TILE_SZ)

    pv_psum = nl.zeros((par_dim(B_P_SIZE), B_D_SIZE), dtype=np.float32, buffer=nl.psum, lazy_initialization=True)
    for k_i in nl.affine_range(LARGE_TILE_SZ // B_P_SIZE):
        pv_psum[:, :] += nl.matmul(
            p_local_transposed[:, nl.ds(k_i * B_P_SIZE, B_P_SIZE)], v[k_i, :, :], transpose_x=True
        )  # (128, 128) (p(Br), d)

    o_buffer[:, :] = nl.add(o_previous_scaled, pv_psum)

    exp = nisa.activation(nl.exp, m_current, bias=l_buffer[:, 0], scale=-1.0)
    l_buffer[:, 0] = nl.add(m_current, nisa.activation(nl.log, exp, bias=ps))


# ── New kernel ───────────────────────────────────────────────────────────


@nki.jit
def flash_attn_neuron(q, k, v, attn_mask=None, softmax_scale=None, use_causal_mask=False):
    """
    Flash attention kernel accepting standard transformers tensor layouts.

    Same flash attention algorithm as neuronxcc.nki.kernels.attention.flash_fwd
    but with a transformers-native interface:

    IO tensor layouts:
      - q:         (bs, n_heads, seq_q, d)   — standard transformers layout
      - k:         (bs, nk_heads, seq_k, d)  — standard transformers layout
      - v:         (bs, nv_heads, seq_v, d)   — standard transformers layout
      - attn_mask:  (bs, 1, seq_q, seq_k)     — additive float mask, broadcast over heads
      - returns:   (bs, seq_q, n_heads, d)   — output in transformers layout

    Compared to flash_fwd:
      - Q, K accepted in (seq, d) layout, transposed on load via DMA transpose
      - V accepted in (seq, d) layout (same as flash_fwd with should_transpose_v=False)
      - Attention mask indexed per-batch (not restricted to batch=1)
      - Inference only (no dropout, training, seed, LSE)

    Constraints:
      - d <= 128
      - seq_q % 128 == 0
      - seq_k % 512 == 0

    GQA: launch grid on (batch, kv_heads); kernel iterates q_heads per kv_head.

    Example:
      MHA: q [b, h, s, d], k [b, h, s, d], v [b, h, s, d]
        flash_attn_neuron[b, h](q, k, v, ...)
      GQA: q [b, h, s, d], k [b, kv_h, s, d], v [b, kv_h, s, d]
        flash_attn_neuron[b, kv_h](q, k, v, ...)
    """
    B_F_SIZE = 512
    B_P_SIZE = 128

    b, h, seqlen_q, d = q.shape
    _, k_h, seqlen_k, _ = k.shape
    B_D_SIZE = d

    assert d <= 128, f"head_dim must be <= 128, got {d}"
    assert seqlen_q % B_P_SIZE == 0, f"seq_q must be divisible by {B_P_SIZE}, got {seqlen_q}"
    assert seqlen_k % B_F_SIZE == 0, f"seq_k must be divisible by {B_F_SIZE}, got {seqlen_k}"
    assert tuple(v.shape) == (b, k_h, seqlen_k, d), f"V shape must be {(b, k_h, seqlen_k, d)}, got {v.shape}"

    kernel_dtype = nl.bfloat16
    acc_type = np.dtype(np.float32)

    q_h_per_k_h = h // k_h

    # Allocate the full output tensor in SHARED HBM (nl.shared_hbm, not nl.hbm).
    # nl.shared_hbm is a single allocation visible to ALL SPMD instances; each
    # instance at (batch_id, head_id) writes its own slice via nl.store.
    # Shape: (batch, seqlen_q, n_q_heads, head_dim) — transformers layout.
    o = nl.ndarray((b, seqlen_q, h, d), dtype=kernel_dtype, buffer=nl.shared_hbm)

    # Select seq_tile_size based on K sequence length
    if seqlen_k % 2048 == 0:
        LARGE_TILE_SZ = 2048
    elif seqlen_k % 1024 == 0:
        LARGE_TILE_SZ = 1024
    else:
        LARGE_TILE_SZ = 512

    config = _FlashConfig(seq_tile_size=LARGE_TILE_SZ)
    attn_core_tile_size = config.attn_core_tile_size

    assert nl.program_ndim() == 2, f"Expected 2D SPMD grid, got {nl.program_ndim()}"
    batch_id = nl.program_id(axis=0)
    head_id = nl.program_id(axis=1)

    softmax_scale = softmax_scale or (1.0 / (d**0.5))
    n_tile_q = seqlen_q // B_P_SIZE
    num_large_k_tile = seqlen_k // LARGE_TILE_SZ

    n_remat = _div_ceil(n_tile_q, attn_core_tile_size)
    attn_core_tile_size = min(n_tile_q, attn_core_tile_size)

    for i_q_h in nl.affine_range(q_h_per_k_h):
        # ============ Global Flash Attention accumulators ============ #
        l_buffer = nl.full(
            (par_dim(B_P_SIZE), n_tile_q),
            fill_value=nl.fp32.min,
            dtype=acc_type,
            buffer=nl.sbuf,
            lazy_initialization=False,
        )

        for i0 in nl.sequential_range(n_remat):
            o_buffer = nl.zeros(
                (attn_core_tile_size, par_dim(B_P_SIZE), d), dtype=acc_type, buffer=nl.sbuf, lazy_initialization=False
            )
            m_buffer = nl.full(
                (attn_core_tile_size, par_dim(B_P_SIZE), 1),
                fill_value=nl.fp32.min,
                dtype=acc_type,
                buffer=nl.sbuf,
                lazy_initialization=False,
            )

            for j in nl.sequential_range(0, num_large_k_tile):
                # ── Load K tile with transpose ──────────────────────
                # K on HBM: (seq_k, d). We need (par_dim(d), LARGE_TILE_SZ) in SBUF.
                # Load in 128-row chunks and transpose each (128, d) → (d, 128).
                cur_k_tile = nl.ndarray((par_dim(B_D_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)

                for ki in nl.affine_range(LARGE_TILE_SZ // B_P_SIZE):
                    k_hbm_ref = k[batch_id, head_id, nl.ds(j * LARGE_TILE_SZ + ki * B_P_SIZE, B_P_SIZE), :]
                    if nisa.get_nc_version() >= nisa.nc_version.gen3:
                        k_transposed = nisa.dma_transpose(k_hbm_ref)
                        cur_k_tile[:, nl.ds(ki * B_P_SIZE, B_P_SIZE)] = nisa.tensor_copy(
                            k_transposed, dtype=kernel_dtype
                        )
                    else:
                        cur_k_tile[:, nl.ds(ki * B_P_SIZE, B_P_SIZE)] = nl.load_transpose2d(
                            k_hbm_ref, dtype=kernel_dtype
                        )

                # ── Load V tile (no transpose needed) ───────────────
                # V on HBM: (seq_v, d). Load (128, d) tiles directly.
                cur_v_tile = nl.ndarray((LARGE_TILE_SZ // B_P_SIZE, par_dim(B_P_SIZE), B_D_SIZE), dtype=kernel_dtype)
                v_hbm_tile = v[batch_id, head_id]
                for v_i in nl.affine_range(LARGE_TILE_SZ // B_P_SIZE):
                    cur_v_tile[v_i, :, :] = nl.load(
                        v_hbm_tile[nl.ds(j * LARGE_TILE_SZ + B_P_SIZE * v_i, B_P_SIZE), :], dtype=cur_v_tile.dtype
                    )

                # ── Process Q tiles ─────────────────────────────────
                for i1 in nl.affine_range(attn_core_tile_size):
                    i = i0 * attn_core_tile_size + i1

                    if use_causal_mask:
                        causal_mask = i * B_P_SIZE >= j * LARGE_TILE_SZ
                    else:
                        causal_mask = True

                    if (i < n_tile_q) & causal_mask:
                        # ── Load Q tile with transpose ──────────────
                        # Q on HBM: (seq_q, d). Load (128, d) and transpose → (d, 128).
                        q_tile = nl.ndarray((B_D_SIZE, B_P_SIZE), dtype=kernel_dtype)
                        q_hbm_tile = q[batch_id, head_id * q_h_per_k_h + i_q_h]
                        q_hbm_ref = q_hbm_tile[nl.ds(i * B_P_SIZE, B_P_SIZE), :]

                        if nisa.get_nc_version() >= nisa.nc_version.gen3:
                            q_transposed = nisa.dma_transpose(q_hbm_ref)
                            q_sbuf_tile = nisa.tensor_copy(q_transposed, dtype=kernel_dtype)
                        else:
                            q_sbuf_tile = nl.load_transpose2d(q_hbm_ref, dtype=kernel_dtype)
                        q_tile[:, :] = q_sbuf_tile * softmax_scale

                        # ── Load attention mask tile ────────────────
                        # attn_mask: (bs, 1, seq_q, seq_k) — index per batch, head=0
                        logit_bias_tile = None
                        if attn_mask is not None:
                            logit_bias_tile = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
                            logit_bias_tile[:, :] = nl.load(
                                attn_mask[
                                    batch_id, 0, nl.ds(i * B_P_SIZE, B_P_SIZE), nl.ds(j * LARGE_TILE_SZ, LARGE_TILE_SZ)
                                ]
                            )

                        # ── Core flash attention (unchanged) ────────
                        _flash_attention_core(
                            q_local_tile=q_tile,
                            k=cur_k_tile,
                            v=cur_v_tile,
                            q_h_per_k_h=q_h_per_k_h,
                            seqlen_q=seqlen_q,
                            nheads=h,
                            o_buffer=o_buffer[i1],
                            l_buffer=l_buffer[:, i],
                            m_buffer=m_buffer[i1],
                            batch_id=batch_id,
                            head_id=head_id,
                            gqa_head_idx=i_q_h,
                            q_tile_idx=i,
                            local_k_large_tile_idx=j,
                            kernel_dtype=kernel_dtype,
                            acc_type=acc_type,
                            flash_config=config,
                            use_causal_mask=use_causal_mask,
                            sliding_window=-1,
                            B_P_SIZE=B_P_SIZE,
                            B_F_SIZE=B_F_SIZE,
                            B_D_SIZE=B_D_SIZE,
                            dropout_p=0.0,
                            dropout_p_tensor=None,
                            seed_tensor=None,
                            logit_bias_tile=logit_bias_tile,
                        )

            # ── Write output to HBM ────────────────────────────────
            for i1 in nl.affine_range(attn_core_tile_size):
                i = i0 * attn_core_tile_size + i1

                if i < n_tile_q:
                    exp = nisa.activation(np.exp, l_buffer[:, i], bias=m_buffer[i1, :, :], scale=-1.0)
                    out = nl.multiply(o_buffer[i1, :, :], exp, dtype=kernel_dtype)
                    nl.store(o[batch_id, nl.ds(i * B_P_SIZE, B_P_SIZE), head_id * q_h_per_k_h + i_q_h, :], out)

    return o
