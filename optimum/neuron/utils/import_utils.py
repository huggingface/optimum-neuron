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

from packaging import version


MIN_ACCELERATE_VERSION = "0.20.1"
MIN_PEFT_VERSION = "0.14.0"


def is_neuron_available() -> bool:
    return importlib.util.find_spec("torch_neuron") is not None


def is_neuronx_available() -> bool:
    return importlib.util.find_spec("torch_neuronx") is not None


def is_accelerate_available(min_version: str | None = MIN_ACCELERATE_VERSION) -> bool:
    _accelerate_available = importlib.util.find_spec("accelerate") is not None
    if min_version is not None:
        if _accelerate_available:
            import accelerate

            _accelerate_version = accelerate.__version__
            return version.parse(_accelerate_version) >= version.parse(min_version)
        else:
            return False
    return _accelerate_available


def is_torch_neuronx_available() -> bool:
    return importlib.util.find_spec("torch_neuronx") is not None


def is_trl_available(required_version: str | None = None) -> bool:
    trl_available = importlib.util.find_spec("trl") is not None
    if trl_available:
        import trl

        if required_version is None:
            required_version = trl.__version__

        if version.parse(trl.__version__) == version.parse(required_version):
            return True

        raise RuntimeError(f"Only `trl=={required_version}` is supported, but {trl.__version__} is installed.")
    return False


def is_peft_available(min_version: str | None = MIN_PEFT_VERSION) -> bool:
    _peft_available = importlib.util.find_spec("peft") is not None
    if min_version is not None:
        if _peft_available:
            import peft

            _peft_version = peft.__version__
            return version.parse(_peft_version) >= version.parse(min_version)
        else:
            return False
    return _peft_available
