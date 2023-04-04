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
"""Import utilities."""

import importlib.util


def is_neuron_available() -> bool:
    return importlib.util.find_spec("torch_neuron") is not None


def is_neuronx_available() -> bool:
    return importlib.util.find_spec("torch_neuronx") is not None


def is_torch_xla_available() -> bool:
    found_torch_xla = importlib.util.find_spec("torch_xla") is not None
    import_succeeded = True
    if found_torch_xla:
        try:
            pass
        except Exception:
            import_succeeded = False
    return found_torch_xla and import_succeeded
