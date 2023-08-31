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
"""Utilities related to CLI arguments."""

import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from ...utils import logging


if TYPE_CHECKING:
    from transformers import PretrainedConfig

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


def convert_neuronx_compiler_args_to_neuron(
    auto_cast: Optional[str],
    auto_cast_type: str,
    disable_fast_relayout: bool,
):
    """
    Builds `compiler_args` for neuron compiler.
    """
    compiler_args = []

    if auto_cast is None:
        auto_cast = "none"
    elif auto_cast == "matmul":
        auto_cast = "matmult"

    if auto_cast == "none":
        compiler_args.extend(["--fast-math", auto_cast])
    elif auto_cast == "all":
        if auto_cast_type == "mixed":
            raise ValueError(
                f"For auto_cast={auto_cast}, cannot set auto_cast_type={auto_cast_type}. "
                "Please choose among `bf16`, `fp16` and `tf32`."
            )
        elif auto_cast_type != "bf16":
            compiler_args.extend(["--fast-math", f"fp32-cast-all-{auto_cast_type}"])
        else:
            compiler_args.extend(["--fast-math", auto_cast])
    elif auto_cast == "matmult":
        if auto_cast_type == "mixed":
            compiler_args.extend(["--fast-math", "fp32-cast-matmult"])
        else:
            compiler_args.extend(["--fast-math", f"fp32-cast-matmult-{auto_cast_type}"])
    else:
        raise ValueError(
            f"The auto_cast value {auto_cast} is not valid. Please use one of the following: None, all or matmul."
        )

    if disable_fast_relayout is True:
        compiler_args.append("no-fast-relayout")

    return compiler_args


def store_compilation_config(
    config: Union["PretrainedConfig", OrderedDict],
    input_shapes: Dict[str, int],
    compiler_kwargs: Dict[str, Any],
    input_names: List[str],
    output_names: List[str],
    dynamic_batch_size: bool,
    compiler_type: str,
    compiler_version: str,
    model_type: Optional[str] = None,
    task: str = None,
    **kwargs,
):
    if isinstance(config, OrderedDict):
        update_func = config.__setitem__
    else:
        update_func = config.__setattr__
    config_args = {}

    # Add neuron version to the config, so it can be checked at load time
    config_args["compiler_type"] = compiler_type
    config_args["compiler_version"] = compiler_version

    # Add input shapes during compilation to the config
    for axis, shape in input_shapes.items():
        axis = f"static_{axis}"
        config_args[axis] = shape

    config_args["dynamic_batch_size"] = dynamic_batch_size

    # Add compilation args to the config
    for arg, value in compiler_kwargs.items():
        config_args[arg] = value

    config_args["input_names"] = input_names
    config_args["output_names"] = output_names

    update_func("neuron", config_args)

    if hasattr(config, "_diffusers_version"):
        import diffusers

        update_func("_diffusers_version", diffusers.__version__)

    model_type = getattr(config, "model_type", None) or model_type
    model_type = str(model_type).replace("_", "-")
    update_func("model_type", model_type)
    update_func("task", task)

    return config
