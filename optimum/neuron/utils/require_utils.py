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

from typing import Callable, Any
import functools

from transformers.utils import is_safetensors_available
from . import is_neuronx_distributed_available, is_torch_xla_available


def _create_requires_function(availability_function: Callable[[], bool], package_name: str) -> Callable[..., Any], 
    def require_func(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not is_safetensors_available():
                raise ModuleNotFoundError(
                    f"{func.__name__} requires the `safetensors` package. You can install it by running: pip install "
                    "safetensors"
                )
            return func(*args, **kwargs)

        return wrapper
    return require_func

def requires_safetensors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not is_safetensors_available():
            raise ModuleNotFoundError(
                f"{func.__name__} requires the `safetensors` package. You can install it by running: pip install "
                "safetensors"
            )
        return func(*args, **kwargs)

    return wrapper


requires_torch_xla = _create_requires_function(is_torch_xla_available, "torch_xla")

# def requires_torch_xla(func):
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         if not is_torch_xla_available():
#             raise ModuleNotFoundError(
#                 f"{func.__name__} requires the `torch_xla` package. You can install it by running: pip install "
#                 "torch_xla"
#             )
#         return func(*args, **kwargs)
# 
#     return wrapper
