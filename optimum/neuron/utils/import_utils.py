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
"""Import utilities."""

import importlib.util
from typing import Optional

from packaging import version


MIN_ACCELERATE_VERSION = "0.20.1"


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


def is_neuronx_distributed_available() -> bool:
    return importlib.util.find_spec("neuronx_distributed") is not None


def is_transformers_neuronx_available() -> bool:
    return importlib.util.find_spec("transformers_neuronx") is not None


def is_accelerate_available(min_version: Optional[str] = MIN_ACCELERATE_VERSION) -> bool:
    _accelerate_available = importlib.util.find_spec("accelerate") is not None
    if min_version is not None:
        if _accelerate_available:
            import accelerate

            _accelerate_version = accelerate.__version__
            return version.parse(_accelerate_version) >= version.parse(min_version)
        else:
            return False
    return _accelerate_available
