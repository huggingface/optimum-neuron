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
"""Utility functions to handle deprecation."""

import functools
import inspect
import warnings
from typing import Callable

from packaging import version

from ..version import __version__
from .version_utils import (
    get_neuroncc_version,
    get_neuronx_distributed_version,
    get_neuronxcc_version,
    get_torch_version,
    get_torch_xla_version,
)


def get_transformers_version() -> str:
    import transformers

    return transformers.__version__


PACKAGE_NAME_TO_GET_VERSION_FUNCTION: dict[str, Callable[[], str]] = {
    "transformers": get_transformers_version,
    "optimum-neuron": lambda: __version__,
    "neuroncc": get_neuroncc_version,
    "neuronxcc": get_neuronxcc_version,
    "torch": get_torch_version,
    "torch_xla": get_torch_xla_version,
    "neuronx_distributed": get_neuronx_distributed_version,
}


def deprecate(deprecate_version: str, package_name: str = "optimum-neuron", reason: str = ""):
    if package_name not in PACKAGE_NAME_TO_GET_VERSION_FUNCTION:
        raise ValueError(f"Do not know how to retrieve the version of the package called {package_name}.")
    deprecate_version = version.parse(deprecate_version)
    try:
        package_version = PACKAGE_NAME_TO_GET_VERSION_FUNCTION[package_name]()
    except ModuleNotFoundError:
        # We do not want to fail if the package is not available, otherwise it will make developping locally impossible.
        package_version = "0.0.0"

    package_version = version.parse(package_version)

    def deprecator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if package_version >= deprecate_version:
                msg = [f"{func.__name__} is deprecated."]
                if reason:
                    msg.append(f"Reason: {reason}")
                msg = " ".join(msg)
                warnings.warn(msg, category=DeprecationWarning)

            if inspect.isgeneratorfunction(func):
                yield from func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    return deprecator
