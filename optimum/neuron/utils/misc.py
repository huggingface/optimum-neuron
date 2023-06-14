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
from typing import Callable, Tuple, Any, Dict


def args_and_kwargs_to_kwargs(f: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    TODO
    """
    sig = inspect.signature(f)
    param_names = [p for p in sig.parameters]
    result = dict(zip(param_names, args))
    result.update(kwargs)
    return result

    

