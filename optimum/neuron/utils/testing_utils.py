# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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
"""Utilities for tests."""

import unittest

from .import_utils import is_neuron_available, is_neuronx_available


def requires_neuron(test_case):
    return unittest.skipUnless(is_neuron_available(), "test requires Neuron compiler")(test_case)


def requires_neuronx(test_case):
    return unittest.skipUnless(is_neuronx_available(), "test requires Neuron X compiler")(test_case)


def requires_neuron_or_neuronx(test_case):
    return unittest.skipUnless(
        is_neuron_available() or is_neuronx_available(), "test requires either Neuron or Neuron X compiler"
    )(test_case)


def is_trainium_test(test_case):
    test_case = requires_neuronx(test_case)
    try:
        import pytest
    except ImportError:
        return test_case
    else:
        return pytest.mark.is_trainium_test()(test_case)


def slow(test_case):
    test_case = requires_neuron_or_neuronx(test_case)
    try:
        import pytest
    except ImportError:
        return test_case
    else:
        return pytest.mark.slow()(test_case)
