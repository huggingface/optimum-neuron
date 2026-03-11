"""Tests for the custom NKI flash attention kernel for head_dim > 128.

These tests verify correctness of the d-tiling flash attention kernel by comparing
against a PyTorch reference implementation (softmax(QK^T) @ V).

Tests require Neuron hardware to run.
"""

import pytest
import torch
from nxd_testing import subprocess_test
from transformers import set_seed

from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


def _reference_attention(Q, K, V, use_causal_mask=True, sliding_window=0):
    """PyTorch reference implementation: softmax(scale * Q^T K) V.

    All inputs should be on CPU.

    Args:
        Q: (BH, D, S) -- pre-scaled, CPU tensor
        K: (BH, D, S) -- CPU tensor
        V: (BH, S, D) -- CPU tensor
        use_causal_mask: whether to apply causal mask
        sliding_window: sliding window size (0 = no window)

    Returns:
        (BH, D, S) CPU tensor
    """
    bh, d, seqlen = Q.shape
    # QK^T: (BH, S, S)
    scores = torch.matmul(Q.transpose(1, 2).float(), K.float())  # (BH, S, D) @ (BH, D, S) = (BH, S, S)

    if use_causal_mask:
        # Create causal mask: position i can attend to position j if i >= j
        # positions.unsqueeze(1) is (S, 1), positions.unsqueeze(0) is (1, S)
        # Broadcasting gives mask[i, j] = i >= j (lower triangular)
        positions = torch.arange(seqlen)
        mask = positions.unsqueeze(1) >= positions.unsqueeze(0)  # (S, S)

        if sliding_window > 0:
            # Sliding window: position i can attend to j if i - j < sliding_window
            window_mask = (positions.unsqueeze(1) - positions.unsqueeze(0)) < sliding_window
            mask = mask & window_mask

        mask = mask.unsqueeze(0).expand(bh, -1, -1)  # (BH, S, S)
        scores = scores.masked_fill(~mask, torch.finfo(torch.float32).min)

    attn_weights = torch.softmax(scores, dim=-1)
    # attn_weights @ V: (BH, S, S) @ (BH, S, D) = (BH, S, D)
    output = torch.matmul(attn_weights, V.float())
    # Transpose to (BH, D, S) to match kernel output
    return output.transpose(1, 2).to(Q.dtype)


def _run_kernel(kernel_call, Q, K, V, use_causal_mask=True, sliding_window=0):
    """Run the NKI kernel on XLA device and return CPU result."""
    bh, head_dim, seq_len = Q.shape
    dtype = Q.dtype

    Q_xla = Q.to(device="xla")
    K_xla = K.to(device="xla")
    V_xla = V.to(device="xla")

    output = torch.zeros(bh, head_dim, seq_len, dtype=dtype, device="xla")
    kernel_call(Q_xla, K_xla, V_xla, 1.0, output, use_causal_mask=use_causal_mask, sliding_window=sliding_window)

    return output.cpu()


@is_inferentia_test
@requires_neuronx
@subprocess_test
def test_flash_attention_large_d_sliding_window():
    """Verify that sliding window attention correctly limits the attention span."""
    from torch_neuronx.xla_impl.ops import nki_jit

    from optimum.neuron.models.inference.backend.modules.attention.flash_attention_nki import flash_fwd_large_d

    set_seed(42)
    bh = 2
    head_dim = 256
    seq_len = 4096
    sliding_window = 1024
    dtype = torch.bfloat16

    Q = torch.randn(bh, head_dim, seq_len, dtype=dtype)
    K = torch.randn(bh, head_dim, seq_len, dtype=dtype)
    V = torch.randn(bh, seq_len, head_dim, dtype=dtype)

    scale = 1.0 / (head_dim**0.5)
    Q_scaled = Q * scale

    kernel_call = nki_jit()(flash_fwd_large_d)
    sw_output = _run_kernel(kernel_call, Q_scaled, K, V, use_causal_mask=True, sliding_window=sliding_window)
    ref_output = _reference_attention(Q_scaled, K, V, use_causal_mask=True, sliding_window=sliding_window)

    torch.testing.assert_close(
        sw_output.float(),
        ref_output.cpu().float(),
        atol=5e-2,
        rtol=1e-1,
    )


@is_inferentia_test
@requires_neuronx
@pytest.mark.parametrize("seq_len", [2048, 4096])
@pytest.mark.parametrize("head_dim", [128, 256, 384, 512])
@subprocess_test
def test_flash_attention_various_d(head_dim, seq_len):
    """Test correctness across different head dimensions and sequence lengths.

    seq_len=2048 exercises LARGE_TILE_SZ=1024 (d_n_tiles>1) and LARGE_TILE_SZ=2048 (d_n_tiles=1)
    with a single large K tile; seq_len=4096 uses two large K tiles for both cases.
    """
    from torch_neuronx.xla_impl.ops import nki_jit

    from optimum.neuron.models.inference.backend.modules.attention.flash_attention_nki import flash_fwd_large_d

    set_seed(42)
    bh = 2
    dtype = torch.bfloat16

    Q = torch.randn(bh, head_dim, seq_len, dtype=dtype)
    K = torch.randn(bh, head_dim, seq_len, dtype=dtype)
    V = torch.randn(bh, seq_len, head_dim, dtype=dtype)

    scale = 1.0 / (head_dim**0.5)
    Q_scaled = Q * scale

    kernel_call = nki_jit()(flash_fwd_large_d)
    kernel_output = _run_kernel(kernel_call, Q_scaled, K, V, use_causal_mask=True, sliding_window=0)
    ref_output = _reference_attention(Q_scaled, K, V, use_causal_mask=True, sliding_window=0)

    torch.testing.assert_close(
        kernel_output.float(),
        ref_output.cpu().float(),
        atol=5e-2,
        rtol=1e-1,
    )
