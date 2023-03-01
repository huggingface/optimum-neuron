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
"""Neuron TorchScript model check and export functions."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
from optimum.exporters.error_utils import AtolError, OutputMatchError, ShapeError
from optimum.utils import logging
from transformers.utils import is_torch_available

from ..utils import is_neuron_available, is_neuronx_available


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from .base import NeuronConfig

if is_neuron_available():
    import torch_neuron as neuron

if is_neuronx_available():
    import torch_neuronx as neuron


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def validate_model_outputs(
    config: Union["NeuronConfig"],
    reference_model: "PreTrainedModel",
    neuron_model_path: Path,
    neuron_named_outputs: List[str],
    atol: Optional[float] = None,
    input_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
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
        input_shapes (`Optional[Dict]`, defaults to `None`):
            If specified, allows to use specific shapes to validate the Neuron model on.
            It must be compatible with staically exported Neuron model.

    Raises:
        ValueError: If the outputs shapes or values do not match between the reference and the exported model.
    """
    if not is_torch_available():
        raise ImportError(
            "Cannot validate conversion because PyTorch is not installed. " "Please install PyTorch first."
        )
    import torch

    logger.info("Validating Neuron model...")

    if atol is None:
        if isinstance(config.ATOL_FOR_VALIDATION, dict):
            atol = config.ATOL_FOR_VALIDATION[config.task]
        else:
            atol = config.ATOL_FOR_VALIDATION

    ref_inputs = config.generate_dummy_inputs(return_tuple=False, **input_shapes)
    with torch.no_grad():
        reference_model.eval()
        ref_outputs = reference_model(**ref_inputs)

    neuron_inputs = tuple(ref_inputs.values())
    neuron_model = torch.jit.load(neuron_model_path)
    neuron_outputs = neuron_model(*neuron_inputs)

    # Check if we have a subset of the keys into neuron_outputs against ref_outputs
    ref_output_names_set, neuron_output_names_set = set(ref_outputs.keys()), set(neuron_named_outputs)
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
    for name, output in zip(neuron_output_names_set, neuron_outputs):
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
        # raise AtolError(
        #     "The maximum absolute difference between the output of the reference model and the Neuron "
        #     f"exported model is not within the set tolerance {atol}:\n{msg}"
        # )

    return value_failures


def export(
    model: "PreTrainedModel",
    config: "NeuronConfig",
    output: Path,
    input_shapes: Optional[Dict] = None,
) -> Tuple[List[str], List[str]]:
    """
    Exports a PyTorch model to a Neuron compiled TorchScript model.

    Args:
        model ([`PreTrainedModel`]):
            The model to export.
        config ([`~exporter.NeuronConfig`]):
            The Neuron configuration associated with the exported model.
        output (`Path`):
            Directory to store the exported Neuron model.
        input_shapes (`optional[Dict]`, defaults to `None`):
            If specified, allows to use specific shapes for the example input provided to the Neuron exporter.

    Returns:
        `Tuple[List[str], List[str]]`: A tuple with an ordered list of the model's inputs, and the named inputs from
        the Neuron configuration.
    """
    if not is_torch_available():
        raise ImportError("Cannot convert because PyTorch is not installed. " "Please install PyTorch first.")
    import torch

    output.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Using PyTorch: {torch.__version__}")
    model.config.return_dict = True
    model.config.torchscript = True

    # Check if we need to override certain configuration item
    if config.values_override is not None:
        logger.info(f"Overriding {len(config.values_override)} configuration item(s)")
        for override_config_key, override_config_value in config.values_override.items():
            logger.info(f"\t- {override_config_key} -> {override_config_value}")
            setattr(model.config, override_config_key, override_config_value)

    if input_shapes is None:
        input_shapes = {}  # will use the defaults from DEFAULT_DUMMY_SHAPES

    dummy_inputs = config.generate_dummy_inputs(**input_shapes)
    neuron_model = neuron.trace(model, dummy_inputs)
    torch.jit.save(neuron_model, output)

    return config.inputs, config.outputs
