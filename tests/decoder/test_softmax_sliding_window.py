"""Test old vs new softmax with sliding-window mask on Neuron.

The full gemma3 model test fails with the mask-path softmax but not the
pre-fill approach.  Gemma3 has sliding_window=512 out of seq_len=8192,
meaning ~94% of prior positions are masked.  This test checks whether a
full attention block (Q*K → softmax → softmax*V) with that mask pattern
diverges between the two approaches when compiled on Neuron.
"""

import torch
from nxd_testing import build_module, subprocess_test
from transformers import set_seed

from optimum.neuron.models.inference.backend.modules.attention.utils import manual_softmax
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


# Gemma3-270m-it dimensions (as used in the failing test)
NUM_HEADS = 2  # after TP=2 split: 4 heads / 2
NUM_KV_HEADS = 1  # after TP=2 split: adjusted to 2, then repeat_kv
HEAD_DIM = 256
SEQ_LEN = 8192
SLIDING_WINDOW = 512
FILLED = 5102  # prompt length from the test
SCALE = HEAD_DIM**-0.5


class OldAttentionBlock(torch.nn.Module):
    """Main's approach: pre-fill with finfo.min, softmax without masks, matmul V."""

    def forward(self, Q, K_prior, V_prior, K_active, V_active, prior_mask):
        prior_scores = torch.matmul(Q, K_prior.transpose(2, 3)) * SCALE
        prior_scores = torch.where(prior_mask, prior_scores, torch.finfo(prior_scores.dtype).min)
        active_scores = torch.matmul(Q, K_active.transpose(2, 3)) * SCALE

        softmax_prior, softmax_active = manual_softmax(prior_scores, active_scores)
        softmax_prior = softmax_prior.to(Q.dtype)
        softmax_active = softmax_active.to(Q.dtype)

        attn_prior = torch.matmul(softmax_prior, V_prior)
        attn_active = torch.matmul(softmax_active, V_active)
        return attn_prior + attn_active


class NewAttentionBlock(torch.nn.Module):
    """Branch approach: pass boolean masks into manual_softmax."""

    def forward(self, Q, K_prior, V_prior, K_active, V_active, prior_mask):
        prior_scores = torch.matmul(Q, K_prior.transpose(2, 3)) * SCALE
        active_scores = torch.matmul(Q, K_active.transpose(2, 3)) * SCALE

        softmax_prior, softmax_active = manual_softmax(
            prior_scores,
            active_scores,
            prior_mask=prior_mask,
        )
        softmax_prior = softmax_prior.to(Q.dtype)
        softmax_active = softmax_active.to(Q.dtype)

        attn_prior = torch.matmul(softmax_prior, V_prior)
        attn_active = torch.matmul(softmax_active, V_active)
        return attn_prior + attn_active


def _make_sliding_window_inputs():
    """Create inputs with a sliding-window mask (512 out of 8192 unmasked)."""
    set_seed(42)
    batch = 1
    active_len = 1  # single token generation

    Q = torch.randn(batch, NUM_HEADS, active_len, HEAD_DIM, dtype=torch.bfloat16)
    K_prior = torch.randn(batch, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16)
    V_prior = torch.randn(batch, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16)
    K_active = torch.randn(batch, NUM_HEADS, active_len, HEAD_DIM, dtype=torch.bfloat16)
    V_active = torch.randn(batch, NUM_HEADS, active_len, HEAD_DIM, dtype=torch.bfloat16)

    # Sliding window mask for a token at position FILLED:
    #   - full_mask: positions 0..FILLED-1 are True
    #   - sliding_mask: positions (FILLED - SLIDING_WINDOW)..FILLED-1 are True
    current_pos = FILLED
    prior_mask = torch.zeros(batch, 1, active_len, SEQ_LEN, dtype=torch.bool)
    window_start = max(0, current_pos - SLIDING_WINDOW)
    prior_mask[:, :, :, window_start:current_pos] = True

    unmasked = prior_mask.sum().item()
    total = prior_mask.numel()
    print(f"Mask: {unmasked}/{total} unmasked ({100 * unmasked / total:.1f}%)")

    return Q, K_prior, V_prior, K_active, V_active, prior_mask


def _make_full_attention_inputs():
    """Create inputs with a full-attention mask (all filled positions unmasked)."""
    set_seed(42)
    batch = 1
    active_len = 1

    Q = torch.randn(batch, NUM_HEADS, active_len, HEAD_DIM, dtype=torch.bfloat16)
    K_prior = torch.randn(batch, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16)
    V_prior = torch.randn(batch, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16)
    K_active = torch.randn(batch, NUM_HEADS, active_len, HEAD_DIM, dtype=torch.bfloat16)
    V_active = torch.randn(batch, NUM_HEADS, active_len, HEAD_DIM, dtype=torch.bfloat16)

    # Full attention mask: positions 0..FILLED-1 are True
    prior_mask = torch.zeros(batch, 1, active_len, SEQ_LEN, dtype=torch.bool)
    prior_mask[:, :, :, :FILLED] = True

    unmasked = prior_mask.sum().item()
    total = prior_mask.numel()
    print(f"Mask: {unmasked}/{total} unmasked ({100 * unmasked / total:.1f}%)")

    return Q, K_prior, V_prior, K_active, V_active, prior_mask


def _run_comparison(inputs, label):
    example_inputs = [inputs]

    # Compile and run old approach
    old_module = build_module(OldAttentionBlock, example_inputs)
    old_output = old_module(*inputs)

    # Compile and run new approach
    new_module = build_module(NewAttentionBlock, example_inputs)
    new_output = new_module(*inputs)

    diff = (old_output - new_output).abs()

    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"  Output shape: {old_output.shape}")
    print(f"{'=' * 70}")
    print(f"  bitwise identical: {torch.equal(old_output, new_output)}")
    print(f"  max |diff|:        {diff.max().item():.6e}")
    print(f"  mean |diff|:       {diff.mean().item():.6e}")
    print(f"  NaN in old:        {torch.isnan(old_output).any().item()}")
    print(f"  NaN in new:        {torch.isnan(new_output).any().item()}")

    if not torch.equal(old_output, new_output):
        nonzero = (diff > 0).sum().item()
        print(f"  nonzero diffs:     {nonzero}/{diff.numel()}")

        # Show worst element
        flat_idx = diff.view(-1).argmax().item()
        print(f"  worst element idx: {flat_idx}")
        print(f"    old: {old_output.view(-1)[flat_idx].item():.10e}")
        print(f"    new: {new_output.view(-1)[flat_idx].item():.10e}")

    print(f"{'=' * 70}\n")
    return torch.equal(old_output, new_output)


@is_inferentia_test
@requires_neuronx
@subprocess_test
def test_attention_block_sliding_window():
    """Attention block with sliding-window mask (512/8192 = 6% unmasked)."""
    inputs = _make_sliding_window_inputs()
    identical = _run_comparison(inputs, "Sliding window (512/8192)")
    assert identical, "Old and new attention blocks diverge with sliding window mask"


@is_inferentia_test
@requires_neuronx
@subprocess_test
def test_attention_block_full_attention():
    """Attention block with full-attention mask (5102/8192 = 62% unmasked)."""
    inputs = _make_full_attention_inputs()
    identical = _run_comparison(inputs, "Full attention (5102/8192)")
    assert identical, "Old and new attention blocks diverge with full attention mask"
