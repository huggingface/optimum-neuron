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
import sys
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
        self.already_patched = False

    @abstractmethod
    def process_patching_specs(
        self, patching_specs: Optional[List[Tuple[Any, Any]]] = None, ignore_missing_attributes: bool = False
    ) -> List[Tuple[Any, str, Any, Any, bool]]:
        pass

    def patch(self):
        if self.already_patched:
            return
        for module, attribute_name, _, patch, _ in self.patching_specs:
            setattr(module, attribute_name, patch)
        self.already_patched = True

    def restore(self):
        if not self.already_patched:
            return
        for module, attribute_name, orig, _, should_delete_attribute_at_restore in self.patching_specs:
            if should_delete_attribute_at_restore:
                delattr(module, attribute_name)
            else:
                setattr(module, attribute_name, orig)
        self.already_patched = False

    def __enter__(self):
        return self.patch()

    def __exit__(self, exc_type, exc_value, traceback):
        return self.restore()


class DynamicPatch:
    """
    Wrapper around a patch function.
    This can be used when the patch to apply is a function of the attribute it patches.
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
            try:
                module = importlib.import_module(module_qualified_name)
            except ModuleNotFoundError as e:
                module_qualified_name, module_attribute_containing_attribute_name = module_qualified_name.rsplit(
                    ".", maxsplit=1
                )
                module = importlib.import_module(module_qualified_name)
                try:
                    module = getattr(module, module_attribute_containing_attribute_name)
                except AttributeError:
                    raise e

            module_has_attr = hasattr(module, attribute_name)
            if module_has_attr:
                attribute = getattr(module, attribute_name)
            elif ignore_missing_attributes and not isinstance(patch, DynamicPatch):
                attribute = None
            elif isinstance(patch, DynamicPatch):
                raise ValueError("Cannot ignore missing attribute with a DynamicPatch.")
            else:
                raise AttributeError(
                    f"Attribute {attribute_name} does not exist in {module}, set `ignore_missing_attributes=True` "
                    "to allow not failing when an attribute does not exist."
                )
            if isinstance(patch, DynamicPatch):
                patch = patch(attribute)
            proccessed_patching_specs.append((module, attribute_name, attribute, patch, not module_has_attr))
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
            module_names = attribute_qualified_name.split(".")
            attribute_name = module_names.pop(-1)
            module = model
            for name in module_names:
                module = getattr(module, name)

            module_has_attr = hasattr(module, attribute_name)
            if module_has_attr:
                attribute = getattr(module, attribute_name)
            elif ignore_missing_attributes and not isinstance(patch, DynamicPatch):
                attribute = None
            elif isinstance(patch, DynamicPatch):
                raise ValueError("Cannot ignore missing attribute with a DynamicPatch.")
            else:
                raise AttributeError(
                    f"Attribute {attribute_name} does not exist in {module}, set `ignore_missing_attributes=True` "
                    "to allow not failing when an attribute does not exist."
                )

            if isinstance(patch, DynamicPatch):
                patch = patch(attribute)

            if inspect.ismethod(attribute):
                patch = patch.__get__(model)

            proccessed_patching_specs.append((module, attribute_name, attribute, patch, not module_has_attr))

        return proccessed_patching_specs


def patch_within_function(
    patching_specs: Union[List[Tuple[str, Any]], Tuple[str, Any]], ignore_missing_attributes: bool = False
):
    """
    Decorator that patches attributes of a module during the lifetime of the decorated function.

    Args:
        patching_specs (`Union[List[Tuple[str, Any]], Tuple[str, Any]]`):
            The specifications of what to patch.
        ignore_missing_attributes (`bool`, defaults to `False`):
            Whether or not the patch should fail if the attribute to patch does not exist.

    Returns:
        `Callable`: A patched version of the function.
    """
    if isinstance(patching_specs, tuple) and len(patching_specs) == 2:
        patching_specs = [patching_specs]

    patcher = Patcher(patching_specs, ignore_missing_attributes=ignore_missing_attributes)

    def decorator(func):
        is_bound = hasattr(func, "__self__")

        @functools.wraps(func.__func__ if is_bound else func)
        def wrapper(*args, **kwargs):
            with patcher:
                if is_bound:
                    args = args[1:]
                return func(*args, **kwargs)

        if is_bound:
            wrapper = wrapper.__get__(getattr(func, "__self__"))

        return wrapper

    return decorator


@functools.lru_cache()
def patch_everywhere(attribute_name: str, patch: Any, module_name_prefix: Optional[str] = None):
    """
    Finds all occurences of `attribute_name` in the loaded modules and patches them with `patch`.

    Args:
        attribute_name (`str`):
            The name of attribute to patch.
        patch (`Any`):
            The patch for the attribute.
        module_name_prefix (`Optional[str]`, defaults to `None`):
            If set, only module names starting with this prefix will be considered for patching.
    """
    for name, module in sys.modules.items():
        if module_name_prefix is not None and not name.startswith(module_name_prefix):
            continue
        if hasattr(module, attribute_name):
            setattr(module, attribute_name, patch)
