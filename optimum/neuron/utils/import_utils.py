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

import importlib.metadata
import importlib.util

from packaging import version


MIN_ACCELERATE_VERSION = "0.20.1"
MIN_PEFT_VERSION = "0.14.0"


def _get_package_version(package_name: str) -> str | None:
    package_exists = importlib.util.find_spec(package_name) is not None
    if package_exists:
        try:
            package_version = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            return None
        return package_version
    return None


def is_package_available(package_name: str, min_version: str | None = None) -> bool:
    package_version = _get_package_version(package_name)
    if package_version is None:
        return False
    if min_version is None:
        return True
    return version.parse(package_version) >= version.parse(min_version)


def is_neuronx_available() -> bool:
    return is_package_available("torch_neuronx")


def is_accelerate_available(min_version: str | None = MIN_ACCELERATE_VERSION) -> bool:
    return is_package_available("accelerate", min_version=min_version)


def is_torch_neuronx_available() -> bool:
    return is_package_available("torch_neuronx")


def is_trl_available(required_version: str | None = None) -> bool:
    trl_version = _get_package_version("trl")
    if trl_version is None:
        return False
    if required_version is not None:
        if version.parse(trl_version) == version.parse(required_version):
            return True

        raise RuntimeError(f"Only `trl=={required_version}` is supported, but {trl_version} is installed.")
    return True


def is_peft_available(min_version: str | None = MIN_PEFT_VERSION) -> bool:
    return is_package_available("peft", min_version=min_version)


def is_vllm_available() -> bool:
    return is_package_available("vllm")
