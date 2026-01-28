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
"""Tests for attention mask padding detection."""

import pytest
import torch

from optimum.neuron.models.inference.backend.modules.attention.utils import (
    create_causal_attention_mask_from_position_ids,
)


def _assert_mask_validity(attention_mask, expected_padding_positions):
    """Helper to validate attention mask structure.

    Args:
        attention_mask: Tensor of shape (batch_size, 1, seq_len, seq_len)
        expected_padding_positions: List of lists indicating padding positions per batch
    """
    batch_size, _, seq_len, _ = attention_mask.shape
    assert batch_size == len(expected_padding_positions), "Batch size mismatch"

    for b in range(batch_size):
        for i in range(seq_len):
            for j in range(seq_len):
                is_padding_query = i in expected_padding_positions[b]
                is_padding_key = j in expected_padding_positions[b]
                is_causal = j <= i

                expected = is_causal and not is_padding_query and not is_padding_key
                actual = attention_mask[b, 0, i, j].item()

                assert actual == expected, (
                    f"Mismatch at batch {b}, query {i}, key {j}: expected {expected}, got {actual}"
                )


@pytest.mark.parametrize(
    "position_ids,expected_padding",
    [
        # Test case 1: Left padding
        # position_ids: [1, 1, 1, 0, 1, 2, 3] -> padding at positions 0, 1, 2
        pytest.param(
            torch.tensor([[1, 1, 1, 0, 1, 2, 3]]),
            [[0, 1, 2]],
            id="left_padding",
        ),
        # Test case 2: Right padding
        # position_ids: [0, 1, 2, 3, 1, 1, 1] -> padding at positions 4, 5, 6
        pytest.param(
            torch.tensor([[0, 1, 2, 3, 1, 1, 1]]),
            [[4, 5, 6]],
            id="right_padding",
        ),
        # Test case 3: No padding
        # position_ids: [0, 1, 2, 3, 4, 5, 6] -> no padding
        pytest.param(
            torch.tensor([[0, 1, 2, 3, 4, 5, 6]]),
            [[]],
            id="no_padding",
        ),
        # Test case 4: Left padding, shorter content
        # position_ids: [1, 0, 1, 2, 3, 4, 5] -> padding at position 0
        pytest.param(
            torch.tensor([[1, 0, 1, 2, 3, 4, 5]]),
            [[0]],
            id="left_padding_short",
        ),
        # Test case 5: Batch with mixed padding
        pytest.param(
            torch.tensor(
                [
                    [1, 1, 1, 0, 1, 2, 3],  # left padding
                    [0, 1, 2, 3, 1, 1, 1],  # right padding
                ]
            ),
            [[0, 1, 2], [4, 5, 6]],
            id="batch_mixed_padding",
        ),
    ],
)
def test_attention_mask_padding_detection(position_ids, expected_padding):
    """Test that attention masks correctly identify padding positions."""
    attention_mask = create_causal_attention_mask_from_position_ids(position_ids=position_ids)

    _assert_mask_validity(attention_mask, expected_padding)


def test_attention_mask_shape():
    """Test that the returned attention mask has the correct shape."""
    batch_size = 2
    seq_len = 10
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    attention_mask = create_causal_attention_mask_from_position_ids(position_ids=position_ids)

    assert attention_mask.shape == (batch_size, 1, seq_len, seq_len)
    assert attention_mask.dtype == torch.bool


def test_attention_mask_causality_preserved():
    """Test that causal structure is preserved for non-padded positions."""
    position_ids = torch.tensor([[0, 1, 2, 3, 4]])
    seq_len = 5

    attention_mask = create_causal_attention_mask_from_position_ids(position_ids=position_ids)

    # For no padding, should have standard causal structure
    # Token i can attend to tokens 0...i
    for i in range(seq_len):
        for j in range(seq_len):
            expected = j <= i
            assert attention_mask[0, 0, i, j].item() == expected


def test_left_padding_blocks_attention():
    """Test that left-padded tokens don't participate in attention."""
    position_ids = torch.tensor([[1, 1, 0, 1, 2]])
    seq_len = 5

    attention_mask = create_causal_attention_mask_from_position_ids(position_ids=position_ids)

    # Positions 0 and 1 are padding
    # They should not be queried or keyed
    for i in range(seq_len):
        assert not attention_mask[0, 0, 0, i].item(), "Padded position 0 should not query any position"
        assert not attention_mask[0, 0, 1, i].item(), "Padded position 1 should not query any position"
        assert not attention_mask[0, 0, i, 0].item(), "No position should attend to padded position 0"
        assert not attention_mask[0, 0, i, 1].item(), "No position should attend to padded position 1"
