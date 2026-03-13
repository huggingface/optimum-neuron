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
"""Unit tests for internal port allocation in the serve command."""

import pytest

from optimum.commands.neuron.serve import _allocate_internal_ports


def test_normal_allocation():
    assert _allocate_internal_ports(8080, 2) == [8081, 8082]


def test_single_replica():
    assert _allocate_internal_ports(8080, 1) == [8081]


def test_port_overflow_raises():
    with pytest.raises(ValueError, match="exceed the valid range"):
        _allocate_internal_ports(65534, 2)


def test_edge_valid():
    """Port 65534 + 1 replica = [65535], still valid."""
    assert _allocate_internal_ports(65534, 1) == [65535]


def test_exact_boundary_raises():
    """Port 65535 + 1 replica = [65536], invalid."""
    with pytest.raises(ValueError, match="exceed the valid range"):
        _allocate_internal_ports(65535, 1)
