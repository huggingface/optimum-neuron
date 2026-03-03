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
"""CPU vs Neuron equivalence tests for manual_softmax.

Verifies that the manual_softmax implementation produces equivalent results
when compiled and run on NeuronCores versus CPU, using dimensions
representative of Llama-3.2-1B and Gemma3-1B models.
"""

import pytest
import torch
from nxd_testing import build_module, subprocess_test
from transformers import set_seed

from optimum.neuron.models.inference.backend.modules.attention.utils import manual_softmax
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


# --- Module wrappers for Neuron tracing ---


class SoftmaxWithMask(torch.nn.Module):
    """manual_softmax with boolean prior_mask."""

    def forward(self, prior_scores, active_scores, prior_mask):
        return manual_softmax(prior_scores, active_scores, prior_mask=prior_mask)


class SoftmaxNoMask(torch.nn.Module):
    """manual_softmax without masks (legacy path)."""

    def forward(self, prior_scores, active_scores):
        return manual_softmax(prior_scores, active_scores)


# --- Model-representative configs ---
# Llama-3.2-1B: 32 heads, head_dim=64 (8 heads after TP=4)
# Gemma3-1B:     4 heads, head_dim=256, sliding_window=512

CONFIGS = {
    "llama": {"num_heads": 8, "head_dim": 64, "seq_len": 2048, "filled": 1024, "sliding_window": None},
    "gemma3": {"num_heads": 4, "head_dim": 256, "seq_len": 2048, "filled": 1024, "sliding_window": 512},
}


def _make_inputs(cfg, with_mask):
    """Create model-representative inputs for manual_softmax."""
    set_seed(42)
    batch = 1
    h, d, s, f = cfg["num_heads"], cfg["head_dim"], cfg["seq_len"], cfg["filled"]
    active_len = 1
    scale = d**-0.5

    prior_scores = torch.randn(batch, h, active_len, s, dtype=torch.bfloat16) * scale
    active_scores = torch.randn(batch, h, active_len, active_len, dtype=torch.bfloat16) * scale

    if not with_mask:
        return prior_scores, active_scores

    prior_mask = torch.zeros(batch, 1, active_len, s, dtype=torch.bool)
    if cfg["sliding_window"]:
        window_start = max(0, f - cfg["sliding_window"])
        prior_mask[:, :, :, window_start:f] = True
    else:
        prior_mask[:, :, :, :f] = True
    return prior_scores, active_scores, prior_mask


# --- Tests ---


@is_inferentia_test
@requires_neuronx
@subprocess_test
@pytest.mark.parametrize("model_style", ["llama", "gemma3"])
def test_softmax_with_mask_cpu_vs_neuron(model_style):
    """Masked manual_softmax: CPU and Neuron produce close results."""
    cfg = CONFIGS[model_style]
    inputs = _make_inputs(cfg, with_mask=True)

    # CPU reference
    cpu_prior, cpu_active = manual_softmax(inputs[0], inputs[1], prior_mask=inputs[2])

    # Neuron
    neuron_model = build_module(SoftmaxWithMask, [inputs])
    neuron_prior, neuron_active = neuron_model(*inputs)

    assert not torch.isnan(neuron_prior).any(), "NaN in neuron prior output"
    assert not torch.isnan(neuron_active).any(), "NaN in neuron active output"
    torch.testing.assert_close(neuron_prior, cpu_prior, atol=1e-5, rtol=1e-3)
    torch.testing.assert_close(neuron_active, cpu_active, atol=1e-5, rtol=1e-3)


@is_inferentia_test
@requires_neuronx
@subprocess_test
@pytest.mark.parametrize("model_style", ["llama", "gemma3"])
def test_softmax_no_mask_cpu_vs_neuron(model_style):
    """Unmasked manual_softmax: CPU and Neuron produce close results."""
    cfg = CONFIGS[model_style]
    inputs = _make_inputs(cfg, with_mask=False)

    # CPU reference
    cpu_prior, cpu_active = manual_softmax(inputs[0], inputs[1])

    # Neuron
    neuron_model = build_module(SoftmaxNoMask, [inputs])
    neuron_prior, neuron_active = neuron_model(*inputs)

    assert not torch.isnan(neuron_prior).any(), "NaN in neuron prior output"
    assert not torch.isnan(neuron_active).any(), "NaN in neuron active output"
    torch.testing.assert_close(neuron_prior, cpu_prior, atol=1e-5, rtol=1e-3)
    torch.testing.assert_close(neuron_active, cpu_active, atol=1e-5, rtol=1e-3)
