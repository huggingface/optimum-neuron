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
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import TYPE_CHECKING, Any, Callable

from optimum.utils import logging

from .input_generators import DTYPE_MAPPER


if TYPE_CHECKING:
    from transformers import PretrainedConfig

logger = logging.get_logger()

DISABLE_ARGUMENT_PATCH = os.environ.get("OPTIMUM_DISABLE_ARGUMENT_PATCH", "0")
DISABLE_STRICT_MODE = os.environ.get("OPTIMUM_DISABLE_STRICT_MODE", "0")


@dataclass
class LoRAAdapterArguments:
    model_ids: str | list[str] | None = None
    weight_names: str | list[str] | None = None
    adapter_names: str | list[str] | None = None
    scales: float | list[float] | None = None

    def __post_init__(self):
        if isinstance(self.model_ids, str):
            self.model_ids = [
                self.model_ids,
            ]
        if isinstance(self.weight_names, str):
            self.weight_names = [
                self.weight_names,
            ]
        if isinstance(self.adapter_names, str):
            self.adapter_names = [
                self.adapter_names,
            ]
        if isinstance(self.scales, float):
            self.scales = [
                self.scales,
            ]


@dataclass
class IPAdapterArguments:
    model_id: str | list[str] | None = None
    subfolder: str | list[str] | None = None
    weight_name: str | list[str] | None = None
    scale: float | list[float] | None = None


@dataclass
class ImageEncoderArguments:
    sequence_length: int | None = None
    hidden_size: int | None = None
    projection_dim: int | None = None


@dataclass
class InputShapesArguments:
    batch_size: int | None = None
    text_batch_size: int | None = None
    image_batch_size: int | None = None
    sequence_length: int | None = None
    num_choices: int | None = None
    width: int | None = None
    height: int | None = None
    image_size: int | None = None
    num_images_per_prompt: int | None = None
    patch_size: int | None = None
    num_channels: int | None = None
    feature_size: int | None = None
    nb_max_frames: int | None = None
    audio_sequence_length: int | None = None
    point_batch_size: int | None = None
    nb_points_per_image: int | None = None
    num_beams: int | None = None
    vae_scale_factor: int | None = None
    encoder_hidden_size: int | None = None
    image_encoder_shapes: ImageEncoderArguments | None = None
    rotary_axes_dim: int | None = None


class DataclassParser:
    def __init__(self, **kwargs):
        for name, cls in self.__class__.__annotations__.items():
            if is_dataclass(cls):
                parsed_kwargs = {k: v for k, v in kwargs.items() if k in {f.name for f in fields(cls)}}
                setattr(self, f"{name}", cls(**parsed_kwargs))


class NeuronArgumentParser(DataclassParser):
    input_shapes: InputShapesArguments

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for name, value in kwargs.items():
            if value is not None:
                setattr(self, name, value)

    @property
    def lora_args(self):
        _lora_args = LoRAAdapterArguments(
            model_ids=getattr(self, "lora_model_ids", None),
            weight_names=getattr(self, "lora_weight_names", None),
            adapter_names=getattr(self, "lora_adapter_names", None),
            scales=getattr(self, "lora_scales", None),
        )
        return _lora_args

    @property
    def ip_adapter_args(self):
        _ip_adapter_args = IPAdapterArguments(
            model_id=getattr(self, "ip_adapter_id", None),
            subfolder=getattr(self, "ip_adapter_subfolder", None),
            weight_name=getattr(self, "ip_adapter_weight_name", None),
            scale=getattr(self, "ip_adapter_scale", None),
        )
        return _ip_adapter_args


def validate_arg(
    args,
    arg_name: str,
    error_msg: str,
    validation_function: Callable[[Any], bool] | None = None,
    expected_value: Any | None = None,
):
    """
    Checks that the argument called `arg_name` in `args` has a value matching what is expected for AWS Trainium
    to work well. By default it will patch invalid argument values if the environment variable
    `OPTIMUM_DISABLE_ARGUMENT_PATCH` is left to `"0"` (by default) and an expected value is provided.

    Args:
        arg_name (`str`):
            The name of the argument to check.
        error_msg (`str`):
            The error message to show if the argument does not have a proper value.
        validation_function (`Callable[[Any], bool] | None`, defaults to `None`):
            A function taking an argument as input, and returning whether the argument is valid or not.
        expected_value (`Any | None`, defaults to `None`):
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
                "Trainium, set the environment variable OPTIMUM_DISABLE_STRICT_MODE to 1."
            )
            raise ValueError(f"{error_msg}\n{raise_error_msg}")


def convert_neuronx_compiler_args_to_neuron(
    auto_cast: str | None,
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


def add_shapes_to_config(config_args, input_shapes: dict[str, Any]):
    for axis, shape in input_shapes.items():
        if shape is not None:
            if is_dataclass(shape):
                shape_dict = asdict(shape)
                config_args[axis] = shape_dict
            else:
                axis = f"static_{axis}"
                config_args[axis] = shape
    return config_args


def store_compilation_config(
    config: "PretrainedConfig | dict",
    input_shapes: dict[str, int],
    compiler_kwargs: dict[str, Any],
    int_dtype: str,
    float_dtype: str,
    dynamic_batch_size: bool,
    compiler_type: str,
    compiler_version: str,
    inline_weights_to_neff: bool,
    optlevel: str,
    tensor_parallel_size: int = 1,
    model_type: str | None = None,
    task: str | None = None,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    output_attentions: bool = False,
    output_hidden_states: bool = False,
    **kwargs,
):
    if isinstance(config, dict):
        update_func = config.__setitem__
    else:
        update_func = config.__setattr__
    config_args = {}

    # Add neuron version to the config, so it can be checked at load time
    config_args["compiler_type"] = compiler_type
    config_args["compiler_version"] = compiler_version
    config_args["inline_weights_to_neff"] = inline_weights_to_neff

    # Add input shapes during compilation to the config
    config_args = add_shapes_to_config(config_args, input_shapes)

    config_args["dynamic_batch_size"] = dynamic_batch_size
    config_args["tensor_parallel_size"] = tensor_parallel_size

    # Add compilation args to the config
    config_args["optlevel"] = optlevel
    for arg, value in compiler_kwargs.items():
        config_args[arg] = value

    config_args["input_names"] = input_names
    config_args["output_names"] = output_names
    config_args["int_dtype"] = DTYPE_MAPPER.str(int_dtype)
    config_args["float_dtype"] = DTYPE_MAPPER.str(float_dtype)

    original_model_type = getattr(config, "export_model_type", None) or getattr(
        config, "model_type", None
    )  # prioritize sentence_transformers to transformers
    neuron_model_type = str(model_type).replace("_", "-") if model_type is not None else model_type
    if original_model_type is None:
        update_func(
            "model_type", neuron_model_type
        )  # Add model_type to the config if it doesn't exist before, eg. submodel of Stable Diffusion.
    else:
        config_args["model_type"] = (
            neuron_model_type or original_model_type
        )  # Prioritize Neuron custom model_type, eg. `t5-encoder`.

    # Add args of optional outputs
    config_args["output_attentions"] = output_attentions
    config_args["output_hidden_states"] = output_hidden_states
    config_args["task"] = task

    update_func("neuron", config_args)

    if hasattr(config, "_diffusers_version"):
        import diffusers

        update_func("_diffusers_version", diffusers.__version__)

    return config
