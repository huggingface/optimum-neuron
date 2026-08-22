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

from .import_utils import is_neuronx_available


# Conv-based models whose tracing segfaults/aborts inside torch_neuronx HLO
# generation with Neuron SDK 2.31 (torch-neuronx 2.9 / torch-xla 2.9). The crash
# is in the compiler's tracer, not in optimum-neuron code, so it cannot be caught
# and takes the whole process down; skip them before export until the SDK is fixed.
SDK_231_TRACE_CRASH_MODEL_TYPES = {"convbert", "hubert", "wav2vec2", "yolos"}


def skip_if_sdk_231_trace_crash(model_type: str):
    """Skip the current test if exporting `model_type` crashes the Neuron SDK 2.31 tracer."""
    if model_type not in SDK_231_TRACE_CRASH_MODEL_TYPES:
        return
    import pytest

    pytest.skip(f"{model_type} export crashes the Neuron SDK 2.31 tracer (see SDK_231_TRACE_CRASH_MODEL_TYPES)")


def requires_neuronx(test_case):
    return unittest.skipUnless(is_neuronx_available(), "test requires Neuron X compiler")(test_case)


def is_trainium_test(test_case):
    test_case = requires_neuronx(test_case)
    try:
        import pytest
    except ImportError:
        return test_case
    else:
        return pytest.mark.is_trainium_test()(test_case)


def is_inferentia_test(test_case):
    test_case = requires_neuronx(test_case)
    try:
        import pytest
    except ImportError:
        return test_case
    else:
        return pytest.mark.is_inferentia_test()(test_case)


def slow(test_case):
    test_case = requires_neuronx(test_case)
    try:
        import pytest
    except ImportError:
        return test_case
    else:
        return pytest.mark.slow()(test_case)
