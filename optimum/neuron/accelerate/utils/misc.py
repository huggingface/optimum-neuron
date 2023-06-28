# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Utilities of various sorts related to accelerate with Neuron."""

from ...utils import is_torch_xla_available, patch_everywhere


def is_tpu_available(check_device=True):
    """
    Fake `is_tpu_available` that returns `is_torch_xla_available` to patch `accelerate`.
    """
    return is_torch_xla_available()


def patch_accelerate_is_tpu_available():
    patch_everywhere("is_tpu_available", is_tpu_available, module_name_prefix="accelerate")
