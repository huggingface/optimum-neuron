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
import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple, Union


if TYPE_CHECKING:
    from transformers import PreTrainedModel


class BasePatcher(ABC):
    """
    Base abstract class providing the core features for efficient context manager based patching.
    """

    def __init__(
        self, patching_specs: Optional[List[Tuple[Any, ...]]] = None, ignore_missing_attributes: bool = False
    ):
        self.patching_specs = self.process_patching_specs(
            patching_specs, ignore_missing_attributes=ignore_missing_attributes
        )

    @abstractmethod
    def process_patching_specs(
        self, patching_specs: Optional[List[Tuple[Any, Any]]] = None, ignore_missing_attributes: bool = False
    ) -> List[Tuple[Any, str, Any, Any]]:
        pass

    def __enter__(self):
        for module, attribute_name, _, patch in self.patching_specs:
            setattr(module, attribute_name, patch)

    def __exit__(self, exc_type, exc_value, traceback):
        for module, attribute_name, _, patch in self.patching_specs:
            setattr(module, attribute_name, patch)


class DynamicPatch:
    """
    Wrapper around a patch function.
    When patching needs to be dynamic with the attribute this can be used.
    """

    def __init__(self, patch_function: Callable[[Any], Any]):
        self.patch_function = patch_function

    def __call__(self, attribute: Any) -> Any:
        return self.patch_function(attribute)


class Patcher(BasePatcher):
    """
    Context manager that patches attributes of a module under its scope and restores everything after exit.
    """

    def process_patching_specs(
        self, patching_specs: Optional[List[Tuple[str, Any]]] = None, ignore_missing_attributes: bool = False
    ):
        proccessed_patching_specs = []
        for orig, patch in patching_specs or []:
            module_qualified_name, attribute_name = orig.rsplit(".", maxsplit=1)
            module = importlib.import_module(module_qualified_name)
            if ignore_missing_attributes:
                attribute = getattr(module, attribute_name, None)
            else:
                attribute = getattr(module, attribute_name)
            if isinstance(patch, DynamicPatch):
                if ignore_missing_attributes:
                    raise ValueError("Cannot ignore missing attribute with a DynamicPatch.")
                patch = patch(attribute)
            proccessed_patching_specs.append((module, attribute_name, attribute, patch))
        return proccessed_patching_specs


class ModelPatcher(BasePatcher):
    """
    Context manager that patches attributes of a model under its scope and restores everything after exit.
    """

    def process_patching_specs(
        self,
        patching_specs: Optional[List[Tuple["PreTrainedModel", str, Any]]] = None,
        ignore_missing_attributes: bool = False,
    ):
        proccessed_patching_specs = []
        for model, attribute_qualified_name, patch in patching_specs or []:
            module_names, attribute_name = attribute_qualified_name.rsplit(".", maxsplit=1)
            module = model
            for name in module_names:
                module = getattr(module, name)
            if ignore_missing_attributes:
                attribute = getattr(module, attribute_name, None)
            else:
                attribute = getattr(module, attribute_name)
            if isinstance(patch, DynamicPatch):
                if ignore_missing_attributes:
                    raise ValueError("Cannot ignore missing attribute with a DynamicPatch.")
                patch = patch(attribute)
            if inspect.ismethod(attribute):
                patch = patch.__get__(model)
            proccessed_patching_specs.append((module, attribute_name, attribute, patch))
        return proccessed_patching_specs


def patch_within_function(patching_specs: Union[List[Tuple[str, Any]], Tuple[str, Any]]):
    """
    Patches attributes of a module during the lifetime of the function.
    """
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
