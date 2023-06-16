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
"""Utilities related to patching."""

import functools
import importlib
from typing import Any, List, Optional, Tuple, Union


class Patcher:
    def __init__(self, patching_specs: Optional[List[Tuple[str, Any]]] = None, ignore_missing_attributes: bool = False):
        self.patching_specs = []
        for orig, patch in patching_specs or []:
            module_qualified_name, attribute_name = orig.rsplit(".", maxsplit=1)
            module = importlib.import_module(module_qualified_name)
            if ignore_missing_attributes:
                attribute = getattr(module, attribute_name, None)
            else:
                attribute = getattr(module, attribute_name)
            self.patching_specs.append((module, attribute_name, attribute, patch))

    def __enter__(self):
        for module, attribute_name, _, patch in self.patching_specs:
            setattr(module, attribute_name, patch)

    def __exit__(self, exc_type, exc_value, traceback):
        for module, attribute_name, _, patch in self.patching_specs:
            setattr(module, attribute_name, patch)


def patch_within_function(patching_specs: Union[List[Tuple[str, Any]], Tuple[str, Any]]):
    if isinstance(patching_specs, tuple) and len(patching_specs) == 2:
        patching_specs = [patching_specs]

    patcher = Patcher(patching_specs)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with patcher:
                return func(*args, **kwargs)

        return wrapper

    return decorator
