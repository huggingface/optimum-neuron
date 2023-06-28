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
"""Utilities of various sorts."""

import inspect
from typing import Any, Callable, Dict, Optional, Tuple


def args_and_kwargs_to_kwargs_only(
    f: Callable,
    args: Optional[Tuple[Any, ...]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    include_default_values: bool = False,
) -> Dict[str, Any]:
    """
    Takes a function `f`, the `args` and `kwargs` provided to the function call, and returns the save arguments in the
    keyword arguments format.

    Args:
        f (`Callable`):
            The function that is being called.
        args (`Optional[Tuple[Any, ...]]`, defaults to `None`):
            The args given to `f`.
        kwargs (`Optional[Dict[str, Any]]`, defaults to `None`):
            The kwargs given to `f`.
        include_default_values (`bool`, defaults to `False`):
            Whether or not the return keyword arguments should contain parameters that were not in `args` and `kwargs`
            which have defaults values.

    Returns:
        `Dict[str, Any]`: The same arguments all formated as keyword arguments.
    """
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    sig = inspect.signature(f)
    param_names = list(sig.parameters)
    result = dict(zip(param_names, args))
    result.update(kwargs)
    if include_default_values:
        for param in sig.parameters.values():
            if param.name in result:
                continue
            if param.default != inspect.Parameter.empty:
                result[param.name] = param.default
    return result
