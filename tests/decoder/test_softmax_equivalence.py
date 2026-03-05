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
"""Compare old (pre-fill + no-mask) vs new (boolean-mask) softmax on Neuron.

The two approaches are bitwise identical on CPU.  This test checks whether
the XLA compiler produces numerically different results for the different
computation graphs when compiled and executed on NeuronCores.
"""

import torch
from nxd_testing import build_module, subprocess_test
from transformers import set_seed

from optimum.neuron.models.inference.backend.modules.attention.utils import manual_softmax
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


# Gemma3-270m-like dimensions
NUM_HEADS = 8
HEAD_DIM = 256
SEQ_LEN = 8192
FILLED = 5000
SCALE = HEAD_DIM**-0.5


class OldSoftmaxModule(torch.nn.Module):
    """Main's approach: pre-fill masked positions with finfo.min, then softmax without masks."""

    def forward(self, prior_scores, active_scores, prior_mask):
        fill_val = torch.finfo(prior_scores.dtype).min
        prior_filled = torch.where(prior_mask, prior_scores, fill_val)
        softmax_prior, softmax_active = manual_softmax(
            prior_filled,
            active_scores,
        )
        return softmax_prior, softmax_active


class NewSoftmaxModule(torch.nn.Module):
    """Our branch's approach: pass boolean masks into manual_softmax."""

    def forward(self, prior_scores, active_scores, prior_mask):
        softmax_prior, softmax_active = manual_softmax(
            prior_scores,
            active_scores,
            prior_mask=prior_mask,
        )
        return softmax_prior, softmax_active


def _make_inputs(active_len=1):
    """Create gemma3-representative inputs in bf16."""
    set_seed(42)
    batch = 1
    Q = torch.randn(batch, NUM_HEADS, active_len, HEAD_DIM, dtype=torch.bfloat16) * SCALE
    K_prior = torch.randn(batch, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    K_active = torch.randn(batch, NUM_HEADS, active_len, HEAD_DIM, dtype=torch.bfloat16) * 0.1

    prior_scores = torch.matmul(Q, K_prior.transpose(2, 3)) * SCALE
    active_scores = torch.matmul(Q, K_active.transpose(2, 3)) * SCALE

    # Partial-cache mask: first 5000 positions filled, rest empty
    prior_mask = torch.zeros(batch, 1, active_len, SEQ_LEN, dtype=torch.bool)
    prior_mask[:, :, :, :FILLED] = True

    return prior_scores, active_scores, prior_mask


@is_inferentia_test
@requires_neuronx
@subprocess_test
def test_softmax_old_vs_new_on_neuron():
    """Verify whether old and new softmax produce identical results on Neuron XLA."""
    prior_scores, active_scores, prior_mask = _make_inputs(active_len=1)
    example_inputs = [(prior_scores, active_scores, prior_mask)]

    # Compile and run old approach
    old_module = build_module(OldSoftmaxModule, example_inputs)
    old_prior, old_active = old_module(*example_inputs[0])

    # Compile and run new approach
    new_module = build_module(NewSoftmaxModule, example_inputs)
    new_prior, new_active = new_module(*example_inputs[0])

    # Report
    prior_diff = (old_prior - new_prior).abs()
    active_diff = (old_active - new_active).abs()

    # Separate masked vs unmasked
    mask_expanded = prior_mask.expand_as(old_prior)
    unmasked_diff = prior_diff[mask_expanded]
    masked_diff = prior_diff[~mask_expanded]

    print(f"\n{'=' * 70}")
    print(f"  Neuron XLA: old vs new softmax (single token, {FILLED}/{SEQ_LEN} filled)")
    print(f"{'=' * 70}")
    print("  prior softmax:")
    print(f"    bitwise identical:      {torch.equal(old_prior, new_prior)}")
    print(f"    max |diff| overall:     {prior_diff.max().item():.6e}")
    print(f"    max |diff| UNMASKED:    {unmasked_diff.max().item():.6e}")
    print(f"    max |diff| MASKED:      {masked_diff.max().item():.6e}")
    print(f"    mean |diff| UNMASKED:   {unmasked_diff.mean().item():.6e}")
    print("  active softmax:")
    print(f"    bitwise identical:      {torch.equal(old_active, new_active)}")
    print(f"    max |diff|:             {active_diff.max().item():.6e}")

    # Show worst-case row if different
    if not torch.equal(old_prior, new_prior):
        worst = prior_diff.view(-1).argmax().item()
        b = worst // (prior_diff.shape[1] * prior_diff.shape[2] * prior_diff.shape[3])
        rem = worst % (prior_diff.shape[1] * prior_diff.shape[2] * prior_diff.shape[3])
        h = rem // (prior_diff.shape[2] * prior_diff.shape[3])
        rem2 = rem % (prior_diff.shape[2] * prior_diff.shape[3])
        q = rem2 // prior_diff.shape[3]
        k = rem2 % prior_diff.shape[3]
        print(f"  worst position: b={b} h={h} q={q} k={k}")
        print(f"    old: {old_prior[b, h, q, k].item():.10e}")
        print(f"    new: {new_prior[b, h, q, k].item():.10e}")
        is_masked = not prior_mask[0, 0, min(q, prior_mask.shape[2] - 1), k].item()
        print(f"    masked: {is_masked}")

    # Check for NaN
    has_nan_old = torch.isnan(old_prior).any().item() or torch.isnan(old_active).any().item()
    has_nan_new = torch.isnan(new_prior).any().item() or torch.isnan(new_active).any().item()
    print(f"  NaN in old: {has_nan_old}")
    print(f"  NaN in new: {has_nan_new}")
    print(f"{'=' * 70}\n")

    assert not has_nan_new, "New approach produced NaN"
    # Don't assert equality — we want to observe the difference


PREFILL_SEQ_LEN = 2048  # Smaller than full 8192 to fit in device memory


class ScaledQKComparisonModule(torch.nn.Module):
    """Both scaled_qk variants in one module — returns diff for comparison."""

    def forward(self, Q, K, attention_mask):
        QK = torch.matmul(Q, K.transpose(2, 3)) * SCALE

        old_filled = torch.where(attention_mask, QK, torch.finfo(QK.dtype).min)
        old_softmax = torch.nn.functional.softmax(old_filled, dim=-1, dtype=torch.float32)

        new_filled = torch.where(attention_mask, QK, torch.finfo(torch.float16).min)
        new_softmax = torch.nn.functional.softmax(new_filled, dim=-1, dtype=torch.float32)

        return old_softmax, new_softmax


@is_inferentia_test
@requires_neuronx
@subprocess_test
def test_scaled_qk_old_vs_new_on_neuron():
    """Verify whether old and new scaled_qk fill values produce identical results on Neuron."""
    set_seed(42)
    batch = 1
    Q = torch.randn(batch, NUM_HEADS, PREFILL_SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16) * SCALE
    K = torch.randn(batch, NUM_HEADS, PREFILL_SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16) * 0.1

    # Full causal mask (like gemma3 prefill)
    attention_mask = torch.ones(batch, 1, PREFILL_SEQ_LEN, PREFILL_SEQ_LEN, dtype=torch.bool).tril()

    example_inputs = [(Q, K, attention_mask)]

    module = build_module(ScaledQKComparisonModule, example_inputs)
    old_softmax, new_softmax = module(*example_inputs[0])

    diff = (old_softmax - new_softmax).abs()
    total = old_softmax.numel()

    print(f"\n{'=' * 70}")
    print("  Neuron XLA: scaled_qk finfo(bf16).min vs finfo(f16).min")
    print(f"  shape: {old_softmax.shape}")
    print(f"{'=' * 70}")
    print(f"  bitwise identical: {torch.equal(old_softmax, new_softmax)}")
    print(f"  max |diff|:        {diff.max().item():.6e}")
    print(f"  mean |diff|:       {diff.mean().item():.6e}")
    print(f"  nonzero diffs:     {(diff > 0).sum().item()} / {total}")
    print(f"  NaN in old:        {torch.isnan(old_softmax).any().item()}")
    print(f"  NaN in new:        {torch.isnan(new_softmax).any().item()}")
    if not torch.equal(old_softmax, new_softmax):
        worst = diff.view(-1).argmax().item()
        b = worst // (diff.shape[1] * diff.shape[2] * diff.shape[3])
        rem = worst % (diff.shape[1] * diff.shape[2] * diff.shape[3])
        h = rem // (diff.shape[2] * diff.shape[3])
        rem2 = rem % (diff.shape[2] * diff.shape[3])
        q = rem2 // diff.shape[3]
        k = rem2 % diff.shape[3]
        print(f"  worst: b={b} h={h} q={q} k={k}")
        print(f"    old: {old_softmax[b, h, q, k].item():.10e}")
        print(f"    new: {new_softmax[b, h, q, k].item():.10e}")
        print(f"    masked: {not attention_mask[0, 0, q, k].item()}")
    print(f"{'=' * 70}\n")
