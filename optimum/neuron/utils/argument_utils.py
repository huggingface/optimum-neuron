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
"""Utilities related to CLI arguments."""

import os
from typing import Any, Callable, Optional

from ...utils import logging


logger = logging.get_logger()

DISABLE_ARGUMENT_PATCH = os.environ.get("OPTIMUM_DISABLE_ARGUMENT_PATCH", "0")
DISABLE_STRICT_MODE = os.environ.get("OPTIMUM_DISABLE_STRICT_MODE", "0")


def validate_arg(
    args,
    arg_name: str,
    error_msg: str,
    validation_function: Optional[Callable[[Any], bool]] = None,
    expected_value: Optional[Any] = None,
):
    """
    Checks that the argument called `arg_name` in `args` has a value matching what is expected for AWS Tranium
    to work well. By default it will patch invalid argument values if the environment variable
    `OPTIMUM_DISABLE_ARGUMENT_PATCH` is left to `"0"` (by default) and an expected value is provided.

    Args:
        arg_name (`str`):
            The name of the argument to check.
        error_msg (`str`):
            The error message to show if the argument does not have a proper value.
        validation_function (`Optional[Callable[[Any], bool]]`, defaults to `None`):
            A function taking an argument as input, and returning whether the argument is valid or not.
        expected_value (`Optional[Any]`, defaults to `None`):
            The expected value for the argument:
                - If the environment variable `OPTIMUM_DISABLE_ARGUMENT_PATCH="0"` and the original argument value
                invalid, the argument will be set to this value.
                - If `validation_function` is left unspecified, it will be set to be the following validation
                function:
                    ```python
                    def validation_function(arg):
                        return arg == expected_value
                    ```
    """
    if not hasattr(args, arg_name):
        return

    if expected_value is None and validation_function is None:
        raise ValueError(
            "At least an expected value or a validation_function must be provided, but none was provided here."
        )
    elif validation_function is None and expected_value is not None:

        def expected_validation_function(arg):
            return arg == expected_value

        validation_function = expected_validation_function

    arg = getattr(args, arg_name)
    if not validation_function(arg):
        if DISABLE_ARGUMENT_PATCH == "0" and expected_value is not None:
            patching_msg = (
                f"Setting {arg_name} to {expected_value}. To disable automatic argument patching set the "
                f"environment variable OPTIMUM_DISABLE_ARGUMENT_PATCH to 1."
            )
            logger.warning(f"{error_msg}\n{patching_msg}")
            setattr(args, arg_name, expected_value)
        elif DISABLE_STRICT_MODE == "1":
            logger.warning(error_msg)
        else:
            raise_error_msg = (
                "Aborting training. To disable automatic failure when an argument value is inferred to be wrong for "
                "Tranium, set the environment variable OPTIMUM_DISABLE_STRICT_MODE to 1."
            )
            raise ValueError(f"{error_msg}\n{raise_error_msg}")
