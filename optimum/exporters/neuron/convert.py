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

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import torch

from ...exporters.error_utils import OutputMatchError, ShapeError
from ...neuron.utils import convert_neuronx_compiler_args_to_neuron, is_neuron_available, is_neuronx_available
from ...utils import logging


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from .base import NeuronConfig

if is_neuron_available():
    import torch.neuron as neuron  # noqa: F811

if is_neuronx_available():
    import torch_neuronx as neuronx  # noqa: F811


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def validate_model_outputs(
    config: "NeuronConfig",
    reference_model: "PreTrainedModel",
    neuron_model_path: Path,
    neuron_named_outputs: List[str],
    atol: Optional[float] = None,
):
    """
    Validates the export by checking that the outputs from both the reference and the exported model match.

    Args:
        config ([`~optimum.neuron.exporter.NeuronConfig`]:
            The configuration used to export the model.
        reference_model ([`~PreTrainedModel`]):
            The model used for the export.
        neuron_model_path (`Path`):
            The path to the exported model.
        neuron_named_outputs (`List[str]`):
            The names of the outputs to check.
        atol (`Optional[float]`, defaults to `None`):
            The absolute tolerance in terms of outputs difference between the reference and the exported model.

    Raises:
        ValueError: If the outputs shapes or values do not match between the reference and the exported model.
    """
    logger.info("Validating Neuron model...")

    if atol is None:
        if isinstance(config.ATOL_FOR_VALIDATION, dict):
            atol = config.ATOL_FOR_VALIDATION[config.task]
        else:
            atol = config.ATOL_FOR_VALIDATION

    input_shapes = {}
    for axe in config.mandatory_axes:
        input_shapes[axe] = getattr(config, axe)
    if config.dynamic_batch_size is True:
        input_shapes["batch_size"] *= 2
    ref_inputs = config.generate_dummy_inputs(return_tuple=False, **input_shapes)
    with torch.no_grad():
        reference_model.eval()
        ref_outputs = reference_model(**ref_inputs)

    neuron_inputs = tuple(ref_inputs.values())
    neuron_model = torch.jit.load(neuron_model_path)
    neuron_outputs = neuron_model(*neuron_inputs)

    # Check if we have a subset of the keys into neuron_outputs against ref_outputs
    ref_output_names_set, neuron_output_names_set = set(ref_outputs.keys()), set(neuron_named_outputs)
    neuron_output_names_list = sorted(neuron_output_names_set, key=neuron_named_outputs.index)

    if not neuron_output_names_set.issubset(ref_output_names_set):
        raise OutputMatchError(
            "Neuron model output names do not match reference model output names.\n"
            f"Reference model output names: {ref_output_names_set}\n"
            f"Neuron model output names: {neuron_output_names_set}"
            f"Difference: {neuron_output_names_set.difference(neuron_output_names_set)}"
        )
    else:
        neuron_output_names = ", ".join(neuron_output_names_set)
        logger.info(f"\t-[✓] Neuron model output names match reference model ({neuron_output_names})")

    # Check if the number of outputs matches the number of output names
    if len(neuron_output_names_set) != len(neuron_outputs):
        raise OutputMatchError(
            f"The exported Neuron model has {len(neuron_outputs)} outputs while {len(neuron_output_names_set)} are expected."
        )

    # Check the shape and values match
    shape_failures = []
    value_failures = []
    for name, output in zip(neuron_output_names_list, neuron_outputs):
        ref_output = ref_outputs[name].numpy()
        output = output.numpy()

        logger.info(f'\t- Validating Neuron Model output "{name}":')

        # Shape
        if not output.shape == ref_output.shape:
            logger.error(f"\t\t-[x] shape {output.shape} doesn't match {ref_output.shape}")
            shape_failures.append((name, ref_output.shape, output.shape))
        else:
            logger.info(f"\t\t-[✓] {output.shape} matches {ref_output.shape}")

        # Values
        if not np.allclose(ref_output, output, atol=atol):
            max_diff = np.amax(np.abs(ref_output - output))
            logger.error(f"\t\t-[x] values not close enough, max diff: {max_diff} (atol: {atol})")
            value_failures.append((name, max_diff))
        else:
            logger.info(f"\t\t-[✓] all values close (atol: {atol})")

    if shape_failures:
        msg = "\n".join(f"- {t[0]}: got {t[1]} (reference) and {t[2]} (neuron)" for t in shape_failures)
        raise ShapeError("Output shapes do not match between reference model and the Neuron exported model:\n" "{msg}")

    if value_failures:
        msg = "\n".join(f"- {t[0]}: max diff = {t[1]}" for t in value_failures)
        logger.warning(
            "The maximum absolute difference between the output of the reference model and the Neuron "
            f"exported model is not within the set tolerance {atol}:\n{msg}"
        )


def export(
    model: "PreTrainedModel",
    config: "NeuronConfig",
    output: Path,
    auto_cast: Optional[str] = None,
    auto_cast_type: str = "bf16",
    disable_fast_relayout: bool = False,
    disable_fallback: bool = False,
) -> Tuple[List[str], List[str]]:
    if is_neuron_available():
        return export_neuron(model, config, output, auto_cast, auto_cast_type, disable_fast_relayout, disable_fallback)
    elif is_neuronx_available():
        return export_neuronx(model, config, output, auto_cast, auto_cast_type)
    else:
        raise RuntimeError(
            "Cannot export the model because the neuron(x) compiler is not installed. See https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-setup.html."
        )


def export_neuronx(
    model: "PreTrainedModel",
    config: "NeuronConfig",
    output: Path,
    auto_cast: Optional[str] = None,
    auto_cast_type: str = "bf16",
    dynamic_batch_size: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Exports a PyTorch model to a serialized TorchScript module compiled by neuronx-cc compiler.

    Args:
        model ([`PreTrainedModel`]):
            The model to export.
        config ([`~exporter.NeuronConfig`]):
            The Neuron configuration associated with the exported model.
        output (`Path`):
            Directory to store the exported Neuron model.
        auto_cast (`Optional[str]`, defaults to `None`):
            Whether to cast operations from FP32 to lower precision to speed up the inference. Can be `None`, `"matmul"` or `"all"`, you should use `None` to disable any auto-casting, use `"matmul"` to cast FP32 matrix multiplication operations, and use `"all"` to cast all FP32 operations.
        auto_cast_type (`str`, defaults to `"bf16"`):
            The data type to cast FP32 operations to when auto-cast mode is enabled. Can be `"bf16"`, `"fp16"` or `"tf32"`.

    Returns:
        `Tuple[List[str], List[str]]`: A tuple with an ordered list of the model's inputs, and the named inputs from
        the Neuron configuration.
    """
    output.parent.mkdir(parents=True, exist_ok=True)

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
    for axe in config.mandatory_axes:
        input_shapes[axe] = getattr(config, axe)

    dummy_inputs = config.generate_dummy_inputs(**input_shapes)
    dummy_inputs_tuple = tuple(dummy_inputs.values())
    checked_model = config.check_model_inputs_order(model, dummy_inputs)

    if auto_cast is not None:
        logger.info(f"Using Neuron: --auto-cast {auto_cast}")

        auto_cast = "matmult" if auto_cast == "matmul" else auto_cast
        compiler_args = ["--auto-cast", auto_cast]

        logger.info(f"Using Neuron: --auto-cast-type {auto_cast_type}")
        compiler_args.extend(["--auto-cast-type", auto_cast_type])
    else:
        compiler_args = ["--auto-cast", "none"]

    neuron_model = neuronx.trace(checked_model, dummy_inputs_tuple, compiler_args=compiler_args)

    if config.dynamic_batch_size is True:
        neuron_model = neuronx.dynamic_batch(neuron_model)

    torch.jit.save(neuron_model, output)

    return config.inputs, config.outputs


def export_neuron(
    model: "PreTrainedModel",
    config: "NeuronConfig",
    output: Path,
    auto_cast: Optional[str] = None,
    auto_cast_type: str = "bf16",
    disable_fast_relayout: bool = False,
    disable_fallback: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Exports a PyTorch model to a serialized TorchScript module compiled by neuron-cc compiler.

    Args:
        model ([`PreTrainedModel`]):
            The model to export.
        config ([`~exporter.NeuronConfig`]):
            The Neuron configuration associated with the exported model.
        output (`Path`):
            Directory to store the exported Neuron model.
        auto_cast (`Optional[str]`, defaults to `None`):
            Whether to cast operations from FP32 to lower precision to speed up the inference. Can be `None`, `"matmul"` or `"all"`, you should use `None` to disable any auto-casting, use `"matmul"` to cast FP32 matrix multiplication operations, and use `"all"` to cast all FP32 operations.
        auto_cast_type (`str`, defaults to `"bf16"`):
            The data type to cast FP32 operations to when auto-cast mode is enabled. Can be `"bf16"`, `"fp16"`, ``"mixed" or `"tf32"`. `"mixed"` is only available when auto_cast is "matmul", it will cast operators that use Neuron Matmult engine to bf16 while using fp16 for matmult-based transpose.
        disable_fast_relayout (`bool`, defaults to `False`):
            Whether to disable fast relayout optimization which improves performance by using the matrix multiplier for tensor transpose.
        disable_fallback (`bool`, defaults to `False`):
            Whether to disable CPU partitioning to force operations to Neuron. Defaults to `False`, as without fallback, there could be some compilation failures or performance problems.

    Returns:
        `Tuple[List[str], List[str]]`: A tuple with an ordered list of the model's inputs, and the named inputs from
        the Neuron configuration.
    """
    output.parent.mkdir(parents=True, exist_ok=True)

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
    for axe in config.mandatory_axes:
        input_shapes[axe] = getattr(config, axe)

    dummy_inputs = config.generate_dummy_inputs(**input_shapes)
    dummy_inputs_tuple = tuple(dummy_inputs.values())
    checked_model = config.check_model_inputs_order(model, dummy_inputs)
    compiler_args = convert_neuronx_compiler_args_to_neuron(auto_cast, auto_cast_type, disable_fast_relayout)

    neuron_model = neuron.trace(
        checked_model,
        dummy_inputs_tuple,
        dynamic_batch_size=config.dynamic_batch_size,
        compiler_args=compiler_args,
        fallback=not disable_fallback,
    )
    torch.jit.save(neuron_model, output)

    return config.inputs, config.outputs
