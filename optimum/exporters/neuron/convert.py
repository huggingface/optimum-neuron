# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
"""Neuron compiled model check and export functions."""

import copy
import os
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from transformers import PreTrainedModel

from optimum.neuron.cache.entries.multi_model import MultiModelCacheEntry
from optimum.neuron.cache.entries.single_model import SingleModelCacheEntry
from optimum.neuron.cache.traced import cache_traced_neuron_artifacts
from optimum.neuron.utils import (
    DiffusersPretrainedConfig,
    convert_neuronx_compiler_args_to_neuron,
    is_neuron_available,
    is_neuronx_available,
    store_compilation_config,
)

from ...exporters.error_utils import OutputMatchError, ShapeError
from ...neuron.utils.cache_utils import get_model_name_or_path
from ...neuron.utils.system import get_neuron_major
from ...neuron.utils.version_utils import get_neuroncc_version, get_neuronxcc_version
from ...utils import (
    is_diffusers_available,
    is_sentence_transformers_available,
    logging,
)


if TYPE_CHECKING:
    from .base import NeuronDefaultConfig

if is_neuron_available():
    import torch.neuron as neuron  # noqa: F811

    NEURON_COMPILER_TYPE = "neuron-cc"
    NEURON_COMPILER_VERSION = get_neuroncc_version()

if is_neuronx_available():
    import torch_neuronx as neuronx  # noqa: F811

    NEURON_COMPILER_TYPE = "neuronx-cc"
    NEURON_COMPILER_VERSION = get_neuronxcc_version()

if is_diffusers_available():
    from diffusers import ModelMixin
    from diffusers.configuration_utils import FrozenDict

if is_sentence_transformers_available():
    from sentence_transformers import SentenceTransformer

import neuronx_distributed
from neuronx_distributed.trace.model_builder import BaseModelInstance, ModelBuilder


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def validate_models_outputs(
    models_and_neuron_configs: dict[
        str, tuple["PreTrainedModel | ModelMixin | torch.nn.Module", "NeuronDefaultConfig"]
    ],
    neuron_named_outputs: dict[str, list[str]],
    output_dir: Path,
    atol: float | None = None,
    neuron_files_subpaths: dict[str, str] | None = None,
):
    """
    Validates the export of several models, by checking that the outputs from both the reference and the exported model match.
    The following method validates the Neuron models exported using the `export_models` method.

    Args:
        models_and_neuron_configs (`dict[str, tuple[`PreTrainedModel` | `ModelMixin` | `torch.nn.Module`, `NeuronDefaultConfig`]]):
            A dictionnary containing the models to export and their corresponding neuron configs.
        neuron_named_outputs (`dict[str, list[str]]`):
            The names of the outputs to check.
        output_dir (`Path`):
            Output directory where the exported Neuron models are stored.
        atol (`float | None`, defaults to `None`):
            The absolute tolerance in terms of outputs difference between the reference and the exported model.
        neuron_files_subpaths (`dict[str, str] | None`, defaults to `None`):
            The relative paths from `output_dir` to the Neuron files to do validation on. The order must be the same as the order of submodels
            in the ordered dict `models_and_neuron_configs`. If None, will use the keys from the `models_and_neuron_configs` as names.

    Raises:
        ValueError: If the outputs shapes or values do not match between the reference and the exported model.
    """
    if len(neuron_named_outputs) != len(models_and_neuron_configs.keys()):
        raise ValueError(
            f"Invalid number of Neuron named outputs. Required {models_and_neuron_configs.keys()}, Provided {neuron_named_outputs.keys()}"
        )

    if neuron_named_outputs is not None and len(neuron_named_outputs) != len(models_and_neuron_configs):
        raise ValueError(
            f"Provided custom names {neuron_files_subpaths} for the validation of {len(models_and_neuron_configs)} models. Please provide the same number of Neuron file names as models to export."
        )

    exceptions = []  # run all validations before raising
    neuron_paths = []
    for i, model_name in enumerate(models_and_neuron_configs.keys()):
        submodel, sub_neuron_config = models_and_neuron_configs[model_name]
        ref_submodel = copy.deepcopy(submodel)
        neuron_model_path = (
            output_dir.joinpath(neuron_files_subpaths[model_name])
            if neuron_files_subpaths is not None
            else output_dir.joinpath(model_name + ".neuron")
        )
        neuron_paths.append(neuron_model_path)
        try:
            logger.info(f"Validating {model_name} model...")
            validate_model_outputs(
                config=sub_neuron_config,
                reference_model=ref_submodel,
                neuron_model_path=neuron_model_path,
                neuron_named_outputs=neuron_named_outputs[model_name],
                atol=atol,
            )
        except Exception as e:
            exceptions.append(f"Validation of {model_name} fails: {e}")

    if len(exceptions) != 0:
        for i, exception in enumerate(exceptions[:-1]):
            logger.error(f"Validation {i} for the model {neuron_paths[i].as_posix()} raised: {exception}")
        raise Exception(exceptions[-1])


def validate_model_outputs(
    config: "NeuronDefaultConfig",
    reference_model: "PreTrainedModel | SentenceTransformer | ModelMixin",
    neuron_model_path: Path,
    neuron_named_outputs: list[str],
    atol: float | None = None,
):
    """
    Validates the export by checking that the outputs from both the reference and the exported model match.

    Args:
        config ([`~optimum.neuron.exporter.NeuronDefaultConfig`]:
            The configuration used to export the model.
        reference_model (`"PreTrainedModel | SentenceTransformer | ModelMixin"`):
            The model used for the export.
        neuron_model_path (`Path`):
            The path to the exported model.
        neuron_named_outputs (`list[str]`):
            The names of the outputs to check.
        atol (`float | None`, defaults to `None`):
            The absolute tolerance in terms of outputs difference between the reference and the exported model.

    Raises:
        ValueError: If the outputs shapes or values do not match between the reference and the exported model.
    """
    if atol is None:
        if isinstance(config.ATOL_FOR_VALIDATION, dict):
            atol = config.ATOL_FOR_VALIDATION[config.task]
        else:
            atol = config.ATOL_FOR_VALIDATION

    input_shapes = {}
    for axis in config.mandatory_axes:
        input_shapes[axis] = getattr(config, axis)
    if config.dynamic_batch_size is True and "batch_size" in input_shapes:
        input_shapes["batch_size"] *= 2

    # Reference outputs
    with torch.no_grad():
        reference_model.eval()
        inputs = config.generate_dummy_inputs(return_tuple=False, **input_shapes)
        ref_inputs = config.unflatten_inputs(inputs)
        if hasattr(reference_model, "config") and getattr(reference_model.config, "is_encoder_decoder", False):
            reference_model, _ = config.patch_model_and_prepare_aliases(reference_model, device="cpu", **input_shapes)
        if "SentenceTransformer" in reference_model.__class__.__name__:
            reference_model, _ = config.patch_model_and_prepare_aliases(reference_model, ref_inputs)
            ref_outputs = reference_model(**ref_inputs)
            neuron_inputs = tuple(config.flatten_inputs(inputs).values())
        elif "AutoencoderKL" in getattr(config._config, "_class_name", "") or getattr(
            config._config, "is_encoder_decoder", False
        ):
            # VAE components for stable diffusion or Encoder-Decoder models
            ref_inputs = tuple(ref_inputs.values())
            ref_outputs = reference_model(*ref_inputs)
            neuron_inputs = tuple(inputs.values())
        elif config.CUSTOM_MODEL_WRAPPER is not None:
            ref_inputs = config.flatten_inputs(inputs)
            reference_model, _ = config.patch_model_and_prepare_aliases(reference_model, ref_inputs, device="cpu")
            neuron_inputs = ref_inputs = tuple(ref_inputs.values())
            ref_outputs = reference_model(*ref_inputs)
        else:
            ref_outputs = reference_model(**ref_inputs)
            neuron_inputs = tuple(config.flatten_inputs(inputs).values())

    # Neuron outputs
    neuron_model = torch.jit.load(neuron_model_path)
    neuron_outputs = neuron_model(*neuron_inputs)
    if isinstance(neuron_outputs, dict):
        neuron_outputs = tuple(neuron_outputs.values())
    elif isinstance(neuron_outputs, torch.Tensor):
        neuron_outputs = (neuron_outputs,)

    # Check if we have a subset of the keys into neuron_outputs against ref_outputs
    neuron_output_names_set = set(neuron_named_outputs)
    neuron_output_names_list = sorted(neuron_output_names_set, key=neuron_named_outputs.index)

    if isinstance(ref_outputs, dict):
        ref_output_names_set = set(ref_outputs.keys())
        if not neuron_output_names_set.issubset(ref_output_names_set):
            raise OutputMatchError(
                "Neuron model output names do not match reference model output names.\n"
                f"Reference model output names: {ref_output_names_set}\n"
                f"Neuron model output names: {neuron_output_names_set}\n"
                f"Difference: {neuron_output_names_set.difference(ref_output_names_set)}"
            )
        else:
            neuron_output_names = ", ".join(neuron_output_names_set)
            logger.info(f"\t-[✓] Neuron model output names match reference model ({neuron_output_names})")
    # folowing are cases for diffusers
    elif isinstance(ref_outputs, torch.Tensor):
        ref_outputs = {neuron_named_outputs[0]: ref_outputs}
    elif isinstance(ref_outputs, tuple):
        ref_outputs = dict(zip(neuron_named_outputs, ref_outputs))

    # Check if the number of outputs matches the number of output names
    if len(neuron_output_names_set) != len(neuron_outputs):
        raise OutputMatchError(
            f"The exported Neuron model has {len(neuron_outputs)} outputs while {len(neuron_output_names_set)} are expected."
        )

    # Check the shape and values match
    shape_failures = []
    value_failures = []
    for i, (name, neuron_output) in enumerate(zip(neuron_output_names_list, neuron_outputs)):
        if isinstance(neuron_output, torch.Tensor):
            ref_output = ref_outputs[name] if isinstance(ref_outputs, dict) else ref_outputs[i]
            neuron_output = neuron_output
        elif isinstance(neuron_output, tuple):  # eg. `hidden_states` of `AutoencoderKL` is a tuple of tensors;
            ref_output = torch.stack(ref_outputs[name])
            neuron_output = torch.stack(neuron_output)
        elif isinstance(neuron_output, list):
            ref_output = ref_outputs[name]
            neuron_output = neuron_output

        logger.info(f'\t- Validating Neuron Model output "{name}":')

        # Shape
        output_list = (
            neuron_output if isinstance(neuron_output, list) else [neuron_output]
        )  # eg. `down_block_res_samples` of `ControlNet` is a list of tensors.
        ref_output_list = ref_output if isinstance(ref_output, list) else [ref_output]
        for output, ref_output in zip(output_list, ref_output_list):
            if not output.shape == ref_output.shape:
                logger.error(f"\t\t-[x] shape {output.shape} doesn't match {ref_output.shape}")
                shape_failures.append((name, ref_output.shape, output.shape))
            else:
                logger.info(f"\t\t-[✓] {output.shape} matches {ref_output.shape}")

            # Values
            if not torch.allclose(ref_output, output.to(ref_output.dtype), atol=atol):
                max_diff = torch.max(torch.abs(ref_output - output))
                logger.error(f"\t\t-[x] values not close enough, max diff: {max_diff} (atol: {atol})")
                value_failures.append((name, max_diff))
            else:
                logger.info(f"\t\t-[✓] all values close (atol: {atol})")

    if shape_failures:
        msg = "\n".join(f"- {t[0]}: got {t[1]} (reference) and {t[2]} (neuron)" for t in shape_failures)
        raise ShapeError("Output shapes do not match between reference model and the Neuron exported model:\n{msg}")

    if value_failures:
        msg = "\n".join(f"- {t[0]}: max diff = {t[1]}" for t in value_failures)
        logger.warning(
            "The maximum absolute difference between the output of the reference model and the Neuron "
            f"exported model is not within the set tolerance {atol}:\n{msg}"
        )


def export_models(
    models_and_neuron_configs: dict[
        str, tuple["PreTrainedModel | ModelMixin | torch.nn.Module", "NeuronDefaultConfig"]
    ],
    task: str,
    output_dir: Path,
    disable_neuron_cache: bool | None = False,
    compiler_workdir: Path | None = None,
    inline_weights_to_neff: bool = True,
    optlevel: str = "2",
    output_file_names: dict[str, str] | None = None,
    compiler_kwargs: dict[str, Any] | None = {},
    model_name_or_path: str | None = None,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    Exports a Pytorch model with multiple component models to separate files.

    Args:
        models_and_neuron_configs (`dict[str, tuple[PreTrainedModel | ModelMixin | torch.nn.Module, `NeuronDefaultConfig`]]):
            A dictionnary containing the models to export and their corresponding neuron configs.
        task (`str`):
            The task for which the models should be exported.
        output_dir (`Path`):
            Output directory to store the exported Neuron models.
        disable_neuron_cache (`bool | None`, defaults to `False`):
            Whether to disable automatic caching of AOT compiled models (not applicable for JIT compilation).
        compiler_workdir (`Path | None`, defaults to `None`):
            The directory to store intermediary outputs of the neuron compiler.
        inline_weights_to_neff (`bool`, defaults to `True`):
            Whether to inline the weights to the neff graph. If set to False, weights will be separated from the neff.
        optlevel (`str`, defaults to `"2"`):
            The level of optimization the compiler should perform. Can be `"1"`, `"2"` or `"3"`, defaults to "2".
                1: enables the core performance optimizations in the compiler, while also minimizing compile time.
                2: provides the best balance between model performance and compile time.
                3: may provide additional model execution performance but may incur longer compile times and higher host memory usage during model compilation.
        output_file_names (`dict[str, str] | None`, defaults to `None`):
            The names to use for the exported Neuron files. The order must be the same as the order of submodels in the ordered dict `models_and_neuron_configs`.
            If None, will use the keys from `models_and_neuron_configs` as names.
        compiler_kwargs (`dict[str, Any] | None`, defaults to `None`):
            Arguments to pass to the Neuron(x) compiler for exporting Neuron models.
        model_name_or_path (`str | None`, defaults to `None`):
            Path to pretrained model or model identifier from the Hugging Face Hub.
    Returns:
        `tuple[dict[str, list[str]], dict[str, list[str]]]`: A tuple with two dictionaries containing ordered list of the model's inputs, and the named
        outputs from the Neuron configuration.
    """
    all_inputs = {}
    all_outputs = {}
    if compiler_workdir is not None:
        compiler_workdir = Path(compiler_workdir)

    if output_file_names is not None and len(output_file_names) != len(models_and_neuron_configs):
        raise ValueError(
            f"Provided {len(output_file_names)} custom names for the export of {len(models_and_neuron_configs)} models. Please provide the same number of names as models to export."
        )

    failed_models = []
    total_compilation_time = 0
    compile_configs = {}
    for i, model_name in enumerate(models_and_neuron_configs.keys()):
        logger.info(f"***** Compiling {model_name} *****")
        submodel, sub_neuron_config = models_and_neuron_configs[model_name]
        output_file_name = (
            output_file_names[model_name] if output_file_names is not None else Path(model_name + ".neuron")
        )

        output_path = output_dir / output_file_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # TODO: Remove after the weights/neff separation compilation of sdxl is patched by a neuron sdk release: https://github.com/aws-neuron/aws-neuron-sdk/issues/859
        if not inline_weights_to_neff and getattr(sub_neuron_config, "is_sdxl", False):
            logger.warning(
                "The compilation of SDXL's unet with the weights/neff separation is broken since the Neuron SDK 2.18 release. `inline_weights_to_neff` will be set to True and the caching will be disabled. If you still want to separate the neff and weights, please downgrade your Neuron setup to the 2.17.1 release."
            )
            inline_weights_to_neff = True

        start_time = time.time()
        neuron_inputs, neuron_outputs = export(
            model_or_path=submodel,
            config=sub_neuron_config,
            output=output_path,
            compiler_workdir=compiler_workdir,
            inline_weights_to_neff=inline_weights_to_neff,
            optlevel=optlevel,
            **compiler_kwargs,
        )
        compilation_time = time.time() - start_time
        total_compilation_time += compilation_time
        logger.info(f"[Compilation Time] {np.round(compilation_time, 2)} seconds.")
        all_inputs[model_name] = neuron_inputs
        all_outputs[model_name] = neuron_outputs

        # Add neuron specific configs to model components' original config
        model_config = sub_neuron_config._config
        if is_diffusers_available() and isinstance(model_config, FrozenDict):
            model_config = OrderedDict(model_config)
            model_config = DiffusersPretrainedConfig.from_dict(model_config)

        # only register mandatory input shapes
        input_shapes = sub_neuron_config.input_shapes
        mandatory_shape = [
            elem for arg in sub_neuron_config.INPUT_ARGS for elem in ((arg,) if isinstance(arg, str) else arg[1:])
        ]
        input_shapes = {k: v for k, v in input_shapes.items() if k in mandatory_shape}

        model_config = store_compilation_config(
            config=model_config,
            input_shapes=input_shapes,
            compiler_kwargs=compiler_kwargs,
            int_dtype=sub_neuron_config.int_dtype,
            float_dtype=sub_neuron_config.float_dtype,
            dynamic_batch_size=sub_neuron_config.dynamic_batch_size,
            tensor_parallel_size=sub_neuron_config.tensor_parallel_size,
            compiler_type=NEURON_COMPILER_TYPE,
            compiler_version=NEURON_COMPILER_VERSION,
            inline_weights_to_neff=inline_weights_to_neff,
            optlevel=optlevel,
            model_type=getattr(sub_neuron_config, "MODEL_TYPE", None),
            task=getattr(sub_neuron_config, "task", None),
            output_attentions=getattr(sub_neuron_config, "output_attentions", False),
            output_hidden_states=getattr(sub_neuron_config, "output_hidden_states", False),
        )
        model_config.save_pretrained(output_path.parent)
        compile_configs[model_name] = model_config

    logger.info(f"[Total compilation Time] {np.round(total_compilation_time, 2)} seconds.")

    # cache neuronx model
    if not disable_neuron_cache and is_neuronx_available():
        model_id = get_model_name_or_path(model_config) if model_name_or_path is None else model_name_or_path
        if len(compile_configs) == 1:
            # FIXME: this is overly complicated just to pass the config
            cache_config = list(compile_configs.values())[0]
            cache_entry = SingleModelCacheEntry(model_id=model_id, task=task, config=cache_config)
        else:
            cache_entry = MultiModelCacheEntry(model_id=model_id, configs=compile_configs)

        cache_traced_neuron_artifacts(neuron_dir=output_dir, cache_entry=cache_entry)

    # remove models failed to export
    for i, model_name in failed_models:
        output_file_names.pop(model_name)
        models_and_neuron_configs.pop(model_name)

    return all_inputs, all_outputs


def export(
    model_or_path: "PreTrainedModel | str | Path",
    config: "NeuronDefaultConfig",
    output: Path,
    instance_type: Literal["trn1", "inf2", "trn1n", "trn2"] | None = None,
    compiler_workdir: Path | None = None,
    inline_weights_to_neff: bool = True,
    optlevel: str = "2",
    auto_cast: str | None = None,
    auto_cast_type: str = "bf16",
    disable_fast_relayout: bool = False,
    disable_fallback: bool = False,
) -> tuple[list[str], list[str]]:
    if is_neuron_available():
        return export_neuron(
            model=model_or_path,
            config=config,
            output=output,
            compiler_workdir=compiler_workdir,
            inline_weights_to_neff=inline_weights_to_neff,
            auto_cast=auto_cast,
            auto_cast_type=auto_cast_type,
            disable_fast_relayout=disable_fast_relayout,
            disable_fallback=disable_fallback,
        )
    elif is_neuronx_available():
        return export_neuronx(
            model_or_path=model_or_path,
            config=config,
            output=output,
            compiler_workdir=compiler_workdir,
            inline_weights_to_neff=inline_weights_to_neff,
            optlevel=optlevel,
            instance_type=instance_type,
            auto_cast=auto_cast,
            auto_cast_type=auto_cast_type,
        )
    else:
        raise RuntimeError(
            "Cannot export the model because the neuron(x) compiler is not installed. See https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-setup.html."
        )


def export_neuronx(
    model_or_path: "PreTrainedModel | str | Path",
    config: "NeuronDefaultConfig",
    output: Path,
    instance_type: Literal["trn1", "inf2", "trn1n", "trn2"] | None = None,
    compiler_workdir: Path | None = None,
    inline_weights_to_neff: bool = True,
    optlevel: str = "2",
    auto_cast: str | None = None,
    auto_cast_type: str = "bf16",
) -> tuple[list[str], list[str]]:
    """
    Exports a PyTorch model to a serialized TorchScript module compiled by neuronx-cc compiler.

    Args:
        model_or_path ("PreTrainedModel" | str | Path):
            The model to export or its location(case when applying the parallelism as the model needs to be loaded with the tracing).
        config ([`~exporter.NeuronDefaultConfig`]):
            The Neuron configuration associated with the exported model.
        output (`Path`):
            Directory to store the exported Neuron model.
        instance_type (`Literal["trn1", "inf2", "trn1n", "trn2"] | None`, defaults to `None`):
            Target Neuron instance type on which the compiled model will be run, valid values are: "trn1", "inf2", "trn1n", "trn2".
        compiler_workdir (`Path | None`, defaults to `None`):
            The directory used by neuronx-cc, where you can find intermediary outputs (neff, weight, hlo...).
        inline_weights_to_neff (`bool`, defaults to `True`):
            Whether to inline the weights to the neff graph. If set to False, weights will be separated from the neff.
        optlevel (`str`, defaults to `"2"`):
            The level of optimization the compiler should perform. Can be `"1"`, `"2"` or `"3"`, defaults to "2".
                1: enables the core performance optimizations in the compiler, while also minimizing compile time.
                2: provides the best balance between model performance and compile time.
                3: may provide additional model execution performance but may incur longer compile times and higher host memory usage during model compilation.
        auto_cast (`str | None`, defaults to `None`):
            Whether to cast operations from FP32 to lower precision to speed up the inference. Can be `None`, `"matmul"` or `"all"`, you should use `None` to disable any auto-casting, use `"matmul"` to cast FP32 matrix multiplication operations, and use `"all"` to cast all FP32 operations.
        auto_cast_type (`str`, defaults to `"bf16"`):
            The data type to cast FP32 operations to when auto-cast mode is enabled. Can be `"bf16"`, `"fp16"` or `"tf32"`.

    Returns:
        `tuple[list[str], list[str]]`: A tuple with an ordered list of the model's inputs, and the named inputs from
        the Neuron configuration.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(compiler_workdir, Path):
        compiler_workdir = compiler_workdir.as_posix()

    if hasattr(model_or_path, "config"):
        model_or_path.config.return_dict = True
        model_or_path.config.torchscript = True
    if isinstance(model_or_path, PreTrainedModel):
        model_or_path.eval()

    # Check if we need to override certain configuration item
    if config.values_override is not None:
        logger.info(f"Overriding {len(config.values_override)} configuration item(s)")
        for override_config_key, override_config_value in config.values_override.items():
            logger.info(f"\t- {override_config_key} -> {override_config_value}")
            if isinstance(model_or_path, PreTrainedModel):
                setattr(model_or_path.config, override_config_key, override_config_value)

    # Prepare dummy inputs for tracing
    input_shapes = {}
    for axis in config.mandatory_axes:
        input_shapes[axis] = getattr(config, axis)

    dummy_inputs = prepare_dummy_inputs(config, input_shapes, return_dict=True)
    dummy_inputs_tuple = tuple(dummy_inputs.values())

    # Prepare the model / function(tp) to trace
    if getattr(config, "is_encoder_decoder", False):
        checked_model, aliases = config.patch_model_and_prepare_aliases(model_or_path, **input_shapes)
    else:
        checked_model, aliases = config.patch_model_and_prepare_aliases(model_or_path, dummy_inputs.keys())

    # Construct compiler configurations
    compiler_args = prepare_compiler_flags(
        config=config,
        instance_type=instance_type,
        auto_cast=auto_cast,
        auto_cast_type=auto_cast_type,
        optlevel=optlevel,
    )

    # Incompatibility between dynamic batching and uninlined weights/neff
    if config.dynamic_batch_size and not inline_weights_to_neff:
        logger.warning(
            "Dynamic batching is not yet compatible with the weights/neff non-inlined model. `inline_weights_to_neff` is set to True. If you still want to separate the neff and weights, please set `dynamic_batch_size=False`."
        )
        inline_weights_to_neff = True

    # Start trace
    tensor_parallel_size = config.tensor_parallel_size
    trace_neuronx(
        model=checked_model,
        config=config,
        dummy_inputs=dummy_inputs_tuple,
        compiler_args=compiler_args,
        output=output,
        tensor_parallel_size=tensor_parallel_size,
        aliases=aliases,
        inline_weights_to_neff=inline_weights_to_neff,
        compiler_workdir=compiler_workdir,
    )

    del model_or_path
    return config.inputs, config.outputs


def prepare_dummy_inputs(config: "NeuronDefaultConfig", input_shapes: dict[str, int], return_dict: bool = True):
    """
    Create dummy inputs used for tracing the model.
    """
    dummy_inputs = config.generate_dummy_inputs(**input_shapes)
    dummy_inputs = config.flatten_inputs(dummy_inputs)
    if return_dict:
        return dummy_inputs
    else:
        dummy_inputs_tuple = tuple(dummy_inputs.values())
        return dummy_inputs_tuple


def prepare_compiler_flags(
    config: "NeuronDefaultConfig",
    instance_type: Literal["trn1", "inf2", "trn1n", "trn2"] | None = None,
    auto_cast: str | None = None,
    auto_cast_type: str = "bf16",
    optlevel: str = "2",
):
    if auto_cast is not None:
        logger.info(f"Using Neuron: --auto-cast {auto_cast}")
        auto_cast = "matmult" if auto_cast == "matmul" else auto_cast
        compiler_args = ["--auto-cast", auto_cast]

        logger.info(f"Using Neuron: --auto-cast-type {auto_cast_type}")
        compiler_args.extend(["--auto-cast-type", auto_cast_type])
    else:
        compiler_args = ["--auto-cast", "none"]

    compiler_args.extend(["--optlevel", optlevel])
    logger.info(f"Using Neuron: --optlevel {optlevel}")

    if instance_type is not None:
        compiler_args.extend(["--target", instance_type])

    # `--model-type=transformer`` is now required for all models except those explicitly listed, based on our observations.
    exception_models = {
        "unet",
        "vae-encoder",
        "vae-decoder",
        "hubert",
        "levit",
        "mobilenet-v2",
        "mobilevit",
        "unispeech",
        "unispeech-sat",
        "wav2vec2",
        "wavlm",
    }
    # t5 Encoder
    if config.MODEL_TYPE == "t5-encoder" and config.LIBRARY_NAME == "diffusers":
        exception_models.add("t5-encoder")
        compiler_args.extend(["--model-type", "unet-inference"])
    if config.MODEL_TYPE not in exception_models:
        compiler_args.extend(["--model-type", "transformer"])

    # diffusers specific
    compiler_args = add_stable_diffusion_compiler_args(config, compiler_args)

    compiler_args_str = " ".join(compiler_args)
    return compiler_args_str


def trace_neuronx(
    model,
    config,
    dummy_inputs,
    compiler_args,
    output: Path,
    tensor_parallel_size: int,
    aliases=None,
    inline_weights_to_neff: bool = True,
    compiler_workdir: Path | None = None,
):
    if tensor_parallel_size > 1:
        # Tensor Parallelism
        if isinstance(model, BaseModelInstance):
            # Case 1: Using `neuronx_distributed.trace.model_builder`
            model_builder = ModelBuilder(
                router=None,
                debug=False,
                tp_degree=tensor_parallel_size,
                checkpoint_loader=config.get_checkpoint_loader_fn,
                compiler_workdir=compiler_workdir,
            )
            subfolder = output.parts[1]
            model_builder.add(
                key=subfolder,
                model_instance=model,
                example_inputs=[dummy_inputs],
                priority_model_idx=0,
                compiler_args=compiler_args,
            )
            neuron_model = model_builder.trace(initialize_model_weights=False)

            model_builder.shard_checkpoint(serialize_path=output.parent / "weights/")
            torch.jit.save(neuron_model, output)
        else:
            # Case 2: Using `neuronx_distributed.trace.parallel_model_trace`
            os.environ["LOCAL_WORLD_SIZE"] = str(tensor_parallel_size)
            # TODO: To remove when migrating from `parallel_model_trace` to ModelBuilderV2. `parallel_model_trace` doesn't support custom target.
            if "--target" in compiler_args:
                compiler_args = re.sub(r"--target\s+\S+", "", compiler_args).strip()
            with torch.no_grad():
                neuron_model = neuronx_distributed.trace.parallel_model_trace(
                    model,
                    dummy_inputs,
                    compiler_args=compiler_args,
                    inline_weights_to_neff=inline_weights_to_neff,
                    compiler_workdir=compiler_workdir,
                    tp_degree=tensor_parallel_size,
                )
            neuronx_distributed.trace.parallel_model_save(neuron_model, output)
    else:
        # Case 3: Using `torch_neuronx.trace`
        cpu_backend = get_neuron_major() == -1
        neuron_model = neuronx.trace(
            model,
            dummy_inputs,
            compiler_args=compiler_args,
            input_output_aliases=aliases,
            inline_weights_to_neff=inline_weights_to_neff,
            cpu_backend=cpu_backend,
            compiler_workdir=compiler_workdir,
        )
        if config.dynamic_batch_size is True:
            neuron_model = neuronx.dynamic_batch(neuron_model)
        # diffusers specific
        improve_stable_diffusion_loading(config, neuron_model)
        torch.jit.save(neuron_model, output)

    del model
    del neuron_model
    del dummy_inputs


def add_stable_diffusion_compiler_args(config, compiler_args):
    # Combine the model name and its path to identify which is the subcomponent in Stable Diffusion pipeline
    identifier = getattr(config._config, "_name_or_path", "") + " " + getattr(config._config, "_class_name", "")
    identifier = identifier.lower()

    sd_components = ["text_encoder", "vae", "vae_encoder", "vae_decoder", "controlnet"]
    if any(component in identifier for component in sd_components):
        compiler_args.append("--enable-fast-loading-neuron-binaries")
    # unet or transformer or controlnet
    if any(model_type in identifier for model_type in ["unet", "transformer", "controlnet"]):
        if "flux" in str(getattr(config, "MODEL_TYPE", "")):
            compiler_args.append(" --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=4'")
            return compiler_args
        # SDXL unet doesn't support fast loading neuron binaries(sdk 2.19.1)
        if not getattr(config, "is_sdxl", False):
            compiler_args.append("--enable-fast-loading-neuron-binaries")
    if any(pattern in identifier for pattern in ("unet", "controlnet", "vae")):
        compiler_args.append("--model-type=unet-inference")
    return compiler_args


def improve_stable_diffusion_loading(config, neuron_model):
    # Combine the model name and its path to identify which is the subcomponent in Diffusion pipeline
    identifier = getattr(config._config, "_name_or_path", "") + " " + getattr(config._config, "_class_name", "")
    identifier = identifier.lower()
    sd_components = ["text_encoder", "unet", "transformer", "vae", "vae_encoder", "vae_decoder", "controlnet"]
    if any(component in identifier for component in sd_components):
        neuronx.async_load(neuron_model)
    # unet
    if any(model_type in identifier for model_type in ["unet", "transformer", "controlnet"]):
        neuronx.lazy_load(neuron_model)


def export_neuron(
    model: "PreTrainedModel",
    config: "NeuronDefaultConfig",
    output: Path,
    compiler_workdir: Path | None = None,
    inline_weights_to_neff: bool = True,
    auto_cast: str | None = None,
    auto_cast_type: str = "bf16",
    disable_fast_relayout: bool = False,
    disable_fallback: bool = False,
) -> tuple[list[str], list[str]]:
    """
    Exports a PyTorch model to a serialized TorchScript module compiled by neuron-cc compiler.

    Args:
        model ([`PreTrainedModel`]):
            The model to export.
        config ([`~exporter.NeuronDefaultConfig`]):
            The Neuron configuration associated with the exported model.
        output (`Path`):
            Directory to store the exported Neuron model.
        compiler_workdir (`Path | None`, defaults to `None`):
            The directory used by neuron-cc, where you can find intermediary outputs (neff, weight, hlo...).
        inline_weights_to_neff (`bool`, defaults to `True`):
            Whether to inline the weights to the neff graph. If set to False, weights will be separated from the neff.
        auto_cast (`str | None`, defaults to `None`):
            Whether to cast operations from FP32 to lower precision to speed up the inference. Can be `None`, `"matmul"` or `"all"`, you should use `None` to disable any auto-casting, use `"matmul"` to cast FP32 matrix multiplication operations, and use `"all"` to cast all FP32 operations.
        auto_cast_type (`str`, defaults to `"bf16"`):
            The data type to cast FP32 operations to when auto-cast mode is enabled. Can be `"bf16"`, `"fp16"`, ``"mixed" or `"tf32"`. `"mixed"` is only available when auto_cast is "matmul", it will cast operators that use Neuron Matmult engine to bf16 while using fp16 for matmult-based transpose.
        disable_fast_relayout (`bool`, defaults to `False`):
            Whether to disable fast relayout optimization which improves performance by using the matrix multiplier for tensor transpose.
        disable_fallback (`bool`, defaults to `False`):
            Whether to disable CPU partitioning to force operations to Neuron. Defaults to `False`, as without fallback, there could be some compilation failures or performance problems.

    Returns:
        `tuple[list[str], list[str]]`: A tuple with an ordered list of the model's inputs, and the named inputs from
        the Neuron configuration.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(compiler_workdir, Path):
        compiler_workdir = compiler_workdir.as_posix()

    if hasattr(model, "config"):
        model.config.return_dict = True
        model.config.torchscript = True
    model.eval()

    # Check if we need to override certain configuration item
    if config.values_override is not None:
        logger.info(f"Overriding {len(config.values_override)} configuration item(s)")
        for override_config_key, override_config_value in config.values_override.items():
            logger.info(f"\t- {override_config_key} -> {override_config_value}")
            setattr(model.config, override_config_key, override_config_value)

    input_shapes = {}
    for axis in config.mandatory_axes:
        input_shapes[axis] = getattr(config, axis)

    dummy_inputs = config.generate_dummy_inputs(**input_shapes)
    dummy_inputs_tuple = tuple(dummy_inputs.values())
    checked_model, aliases = config.patch_model_and_prepare_aliases(model, dummy_inputs)
    compiler_args = convert_neuronx_compiler_args_to_neuron(auto_cast, auto_cast_type, disable_fast_relayout)

    if config.dynamic_batch_size is True and not inline_weights_to_neff:
        logger.warning(
            "Dynamic batching is not yet compatible with the weights/neff non-inlined model. `inline_weights_to_neff` is set to True. If you still want to separate the neff and weights, please set `dynamic_batch_size=False`."
        )
        inline_weights_to_neff = True

    neuron_model = neuron.trace(
        checked_model,
        dummy_inputs_tuple,
        dynamic_batch_size=config.dynamic_batch_size,
        compiler_args=compiler_args,
        compiler_workdir=compiler_workdir,
        separate_weights=not inline_weights_to_neff,
        fallback=not disable_fallback,
    )
    torch.jit.save(neuron_model, output)
    del model
    del checked_model
    del dummy_inputs
    del neuron_model

    return config.inputs, config.outputs
