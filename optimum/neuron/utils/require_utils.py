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

from transformers.utils import is_safetensors_available


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
