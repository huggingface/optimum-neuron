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
"""Utilities to handle the several optional requirement packages."""

import functools
from typing import Any, Callable

from transformers.utils import is_safetensors_available

from .import_utils import (
    is_peft_available,
    is_torch_neuronx_available,
    is_vllm_available,
)


_AVAILABILITIES: dict[str, Callable] = {
    "safetensors": is_safetensors_available,
    "torch_neuronx": is_torch_neuronx_available,
    "peft": is_peft_available,
    "vllm": is_vllm_available,
}


def _create_requires_function(package_name: str) -> Callable[..., Any]:
    availability_function = _AVAILABILITIES[package_name]

    def require_func(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not availability_function():
                raise ModuleNotFoundError(
                    f"{func.__name__} requires the `{package_name}` package. You can install it by running: pip "
                    f"install {package_name}"
                )
            return func(*args, **kwargs)

        return wrapper

    return require_func


requires_safetensors = _create_requires_function("safetensors")
requires_torch_neuronx = _create_requires_function("torch_neuronx")
requires_peft = _create_requires_function("peft")
requires_vllm = _create_requires_function("vllm")
