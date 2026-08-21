# coding=utf-8
# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Tests for the Neuron tracing compatibility patches.

These run on CPU: they check that each patch is mathematically equivalent to the
code it replaces. Whether the patches actually make tracing succeed is covered by
the existing exporter and inference tests for yolos / wav2vec2 / hubert / convbert,
which crash without them.
"""

import unittest

import torch
from parameterized import parameterized
from transformers import AutoModel, AutoModelForObjectDetection

from optimum.exporters.neuron.neuron_trace_patches import (
    fold_weight_norm_parametrizations,
    patch_model_for_neuron_tracing,
    unfold_via_slices,
)


class UnfoldViaSlicesTest(unittest.TestCase):
    """`unfold_via_slices` must match `nn.functional.unfold` exactly."""

    @parameterized.expand([(3,), (9,)])
    def test_matches_torch_unfold(self, kernel):
        # Only the kernel size changes behaviour: it is the loop bound, and it sets
        # the padding. Batch/channel/length flow through pad, slice, stack and
        # reshape untouched, so sweeping them adds runtime without adding coverage.
        # ConvBert uses odd kernels; 3 and 9 cover the small and large ends.
        inputs = torch.randn(2, 16, 32, 1)
        padding = (kernel - 1) // 2

        expected = torch.nn.functional.unfold(
            inputs, kernel_size=[kernel, 1], dilation=1, padding=[padding, 0], stride=1
        )
        actual = unfold_via_slices(inputs, kernel_size=[kernel, 1], dilation=1, padding=[padding, 0], stride=1)

        self.assertEqual(expected.shape, actual.shape)
        # Both are pure gather/copy over the same elements, so this is exact.
        self.assertTrue(torch.equal(expected, actual))

    @parameterized.expand(
        [
            # One case per clause of the guard, so each is actually exercised.
            ("non_unit_kernel_width", (1, 8, 16, 1), {"kernel_size": [3, 2], "padding": [1, 0]}),
            ("non_zero_width_padding", (1, 8, 16, 1), {"kernel_size": [3, 1], "padding": [1, 1]}),
            ("non_unit_stride", (1, 8, 16, 1), {"kernel_size": [3, 1], "padding": [1, 0], "stride": 2}),
            ("non_unit_dilation", (1, 8, 16, 1), {"kernel_size": [3, 1], "padding": [1, 0], "dilation": 2}),
            ("non_unit_trailing_dim", (1, 8, 16, 2), {"kernel_size": [3, 1], "padding": [1, 0]}),
        ]
    )
    def test_rejects_unsupported_arguments(self, _, shape, kwargs):
        # Unsupported cases must raise rather than silently compute something else.
        with self.assertRaises(ValueError):
            unfold_via_slices(torch.randn(*shape), **kwargs)


class FoldWeightNormTest(unittest.TestCase):
    def test_folding_preserves_outputs(self):
        torch.manual_seed(0)
        module = torch.nn.utils.weight_norm(torch.nn.Conv1d(8, 8, 3, padding=1), dim=2)
        module.eval()
        inputs = torch.randn(1, 8, 32)

        with torch.no_grad():
            before = module(inputs)

        folded = fold_weight_norm_parametrizations(module)

        with torch.no_grad():
            after = module(inputs)

        self.assertEqual(len(folded), 1)
        self.assertFalse(hasattr(module, "parametrizations"))
        self.assertTrue(torch.allclose(before, after, atol=1e-6))

    def test_is_a_no_op_without_weight_norm(self):
        module = torch.nn.Conv1d(4, 4, 3)
        self.assertEqual(fold_weight_norm_parametrizations(module), [])


class PatchModelForNeuronTracingTest(unittest.TestCase):
    """End-to-end equivalence on the models that need patching."""

    def _assert_outputs_unchanged(self, model, inputs, atol=1e-6):
        model.eval()
        with torch.no_grad():
            before = model(**inputs)

        patch_model_for_neuron_tracing(model)

        with torch.no_grad():
            after = model(**inputs)

        before_tensors = [value for value in before.to_tuple() if isinstance(value, torch.Tensor)]
        after_tensors = [value for value in after.to_tuple() if isinstance(value, torch.Tensor)]
        self.assertEqual(len(before_tensors), len(after_tensors))
        for expected, actual in zip(before_tensors, after_tensors):
            self.assertTrue(torch.allclose(expected, actual, atol=atol))

    def test_wav2vec2_outputs_unchanged(self):
        model = AutoModel.from_pretrained("hf-internal-testing/tiny-random-Wav2Vec2Model")
        self._assert_outputs_unchanged(model, {"input_values": torch.rand(1, 16000)})

    def test_hubert_outputs_unchanged(self):
        model = AutoModel.from_pretrained("hf-internal-testing/tiny-random-HubertModel")
        self._assert_outputs_unchanged(model, {"input_values": torch.rand(1, 16000)})

    def test_convbert_outputs_unchanged(self):
        model = AutoModel.from_pretrained("hf-internal-testing/tiny-random-ConvBertModel")
        self._assert_outputs_unchanged(model, {"input_ids": torch.randint(0, 100, (1, 32))})

    def test_yolos_outputs_unchanged(self):
        model = AutoModelForObjectDetection.from_pretrained("hf-internal-testing/tiny-random-YolosModel")
        self._assert_outputs_unchanged(model, {"pixel_values": torch.rand(1, 3, 30, 30)})


if __name__ == "__main__":
    unittest.main()
