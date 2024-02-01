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
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from ...exporters.error_utils import OutputMatchError, ShapeError
from ...neuron.utils import (
    convert_neuronx_compiler_args_to_neuron,
    is_neuron_available,
    is_neuronx_available,
    store_compilation_config,
)
from ...neuron.utils.version_utils import get_neuroncc_version, get_neuronxcc_version
from ...utils import (
    is_diffusers_available,
    is_sentence_transformers_available,
    logging,
)
from .utils import DiffusersPretrainedConfig


if TYPE_CHECKING:
    from transformers import PreTrainedModel

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

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def validate_models_outputs(
    models_and_neuron_configs: Dict[
        str, Tuple[Union["PreTrainedModel", "ModelMixin", torch.nn.Module], "NeuronDefaultConfig"]
    ],
    neuron_named_outputs: List[List[str]],
    output_dir: Path,
    atol: Optional[float] = None,
    neuron_files_subpaths: Optional[Dict[str, str]] = None,
):
    """
    Validates the export of several models, by checking that the outputs from both the reference and the exported model match.
    The following method validates the Neuron models exported using the `export_models` method.

    Args:
        models_and_neuron_configs (`Dict[str, Tuple[Union[`PreTrainedModel`, `ModelMixin`, `torch.nn.Module`], `NeuronDefaultConfig`]]):
            A dictionnary containing the models to export and their corresponding neuron configs.
        neuron_named_outputs (`List[List[str]]`):
            The names of the outputs to check.
        output_dir (`Path`):
            Output directory where the exported Neuron models are stored.
        atol (`Optional[float]`, defaults to `None`):
            The absolute tolerance in terms of outputs difference between the reference and the exported model.
        neuron_files_subpaths (`Optional[List[str]]`, defaults to `None`):
            The relative paths from `output_dir` to the Neuron files to do validation on. The order must be the same as the order of submodels
            in the ordered dict `models_and_neuron_configs`. If None, will use the keys from the `models_and_neuron_configs` as names.

    Raises:
        ValueError: If the outputs shapes or values do not match between the reference and the exported model.
    """
    if len(neuron_named_outputs) != len(models_and_neuron_configs.keys()):
        raise ValueError(
            f"Invalid number of Neuron named outputs. Required {len(models_and_neuron_configs.keys())}, Provided {len(neuron_named_outputs)}"
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
                neuron_named_outputs=neuron_named_outputs[i],
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
    reference_model: Union["PreTrainedModel", "SentenceTransformer", "ModelMixin"],
    neuron_model_path: Path,
    neuron_named_outputs: List[str],
    atol: Optional[float] = None,
):
    """
    Validates the export by checking that the outputs from both the reference and the exported model match.

    Args:
        config ([`~optimum.neuron.exporter.NeuronDefaultConfig`]:
            The configuration used to export the model.
        reference_model ([`Union["PreTrainedModel", "SentenceTransformer", "ModelMixin"]`]):
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
    if atol is None:
        if isinstance(config.ATOL_FOR_VALIDATION, dict):
            atol = config.ATOL_FOR_VALIDATION[config.task]
        else:
            atol = config.ATOL_FOR_VALIDATION

    input_shapes = {}
    for axis in config.mandatory_axes:
        input_shapes[axis] = getattr(config, axis)
    if config.dynamic_batch_size is True:
        input_shapes["batch_size"] *= 2

    # Reference outputs
    with torch.no_grad():
        reference_model.eval()
        ref_inputs = config.generate_dummy_inputs(return_tuple=False, **input_shapes)
        if hasattr(reference_model, "config") and getattr(reference_model.config, "is_encoder_decoder", False):
            reference_model = config.patch_model_for_export(reference_model, device="cpu", **input_shapes)
        if "SentenceTransformer" in reference_model.__class__.__name__:
            reference_model = config.patch_model_for_export(reference_model, ref_inputs)
            ref_outputs = reference_model(**ref_inputs)
            neuron_inputs = tuple(config.flatten_inputs(ref_inputs).values())
        elif "AutoencoderKL" in getattr(config._config, "_class_name", "") or getattr(
            reference_model.config, "is_encoder_decoder", False
        ):
            # VAE components for stable diffusion or Encoder-Decoder models
            ref_inputs = tuple(ref_inputs.values())
            ref_outputs = reference_model(*ref_inputs)
            neuron_inputs = ref_inputs
        else:
            ref_outputs = reference_model(**ref_inputs)
            neuron_inputs = tuple(config.flatten_inputs(ref_inputs).values())

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
    for i, (name, output) in enumerate(zip(neuron_output_names_list, neuron_outputs)):
        if isinstance(output, torch.Tensor):
            ref_output = ref_outputs[name].numpy() if isinstance(ref_outputs, dict) else ref_outputs[i].numpy()
            output = output.numpy()
        elif isinstance(output, tuple):  # eg. `hidden_states` of `AutoencoderKL` is a tuple of tensors.
            ref_output = torch.stack(ref_outputs[name]).numpy()
            output = torch.stack(output).numpy()

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


def export_models(
    models_and_neuron_configs: Dict[
        str, Tuple[Union["PreTrainedModel", "ModelMixin", torch.nn.Module], "NeuronDefaultConfig"]
    ],
    output_dir: Path,
    compiler_workdir: Optional[Path] = None,
    inline_weights_to_neff: bool = True,
    optlevel: str = "2",
    output_file_names: Optional[Dict[str, str]] = None,
    compiler_kwargs: Optional[Dict[str, Any]] = {},
    configs: Optional[Dict[str, Any]] = {},
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Exports a Pytorch model with multiple component models to separate files.

    Args:
        models_and_neuron_configs (`Dict[str, Tuple[Union["PreTrainedModel", "ModelMixin", torch.nn.Module], `NeuronDefaultConfig`]]):
            A dictionnary containing the models to export and their corresponding neuron configs.
        output_dir (`Path`):
            Output directory to store the exported Neuron models.
        compiler_workdir (`Optional[Path]`, defaults to `None`):
            The directory to store intermediary outputs of the neuron compiler.
        inline_weights_to_neff (`bool`, defaults to `True`):
            Whether to inline the weights to the neff graph. If set to False, weights will be seperated from the neff.
        optlevel (`str`, defaults to `"2"`):
            The level of optimization the compiler should perform. Can be `"1"`, `"2"` or `"3"`, defaults to "2".
                1: enables the core performance optimizations in the compiler, while also minimizing compile time.
                2: provides the best balance between model performance and compile time.
                3: may provide additional model execution performance but may incur longer compile times and higher host memory usage during model compilation.
        output_file_names (`Optional[List[str]]`, defaults to `None`):
            The names to use for the exported Neuron files. The order must be the same as the order of submodels in the ordered dict `models_and_neuron_configs`.
            If None, will use the keys from `models_and_neuron_configs` as names.
        compiler_kwargs (`Optional[Dict[str, Any]]`, defaults to `None`):
            Arguments to pass to the Neuron(x) compiler for exporting Neuron models.
        configs (`Optional[Dict[str, Any]]`, defaults to `None`):
            A list of pretrained model configs.
    Returns:
        `Tuple[List[List[str]], List[List[str]]]`: A tuple with an ordered list of the model's inputs, and the named
        outputs from the Neuron configuration.
    """
    outputs = []
    if compiler_workdir is not None:
        compiler_workdir = Path(compiler_workdir)

    if output_file_names is not None and len(output_file_names) != len(models_and_neuron_configs):
        raise ValueError(
            f"Provided {len(output_file_names)} custom names for the export of {len(models_and_neuron_configs)} models. Please provide the same number of names as models to export."
        )

    failed_models = []
    total_compilation_time = 0
    for i, model_name in enumerate(models_and_neuron_configs.keys()):
        logger.info(f"***** Compiling {model_name} *****")
        submodel, sub_neuron_config = models_and_neuron_configs[model_name]
        output_file_name = (
            output_file_names[model_name] if output_file_names is not None else Path(model_name + ".neuron")
        )

        output_path = output_dir / output_file_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        compiler_workdir_path = compiler_workdir / model_name if compiler_workdir is not None else None

        try:
            start_time = time.time()
            neuron_inputs, neuron_outputs = export(
                model=submodel,
                config=sub_neuron_config,
                output=output_path,
                compiler_workdir=compiler_workdir_path,
                inline_weights_to_neff=inline_weights_to_neff,
                optlevel=optlevel,
                **compiler_kwargs,
            )
            compilation_time = time.time() - start_time
            total_compilation_time += compilation_time
            logger.info(f"[Compilation Time] {np.round(compilation_time, 2)} seconds.")
            outputs.append((neuron_inputs, neuron_outputs))
            # Add neuron specific configs to model components' original config
            if hasattr(submodel, "config"):
                model_config = submodel.config
            elif configs and (model_name in configs.keys()):
                model_config = configs[model_name]
            else:
                raise AttributeError("Cannot find model's configuration, please pass it with `configs`.")

            if is_diffusers_available() and isinstance(model_config, FrozenDict):
                model_config = OrderedDict(model_config)
                model_config = DiffusersPretrainedConfig.from_dict(model_config)

            model_config = store_compilation_config(
                config=model_config,
                input_shapes=sub_neuron_config.input_shapes,
                compiler_kwargs=compiler_kwargs,
                input_names=neuron_inputs,
                output_names=neuron_outputs,
                dynamic_batch_size=sub_neuron_config.dynamic_batch_size,
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
        except Exception as e:
            failed_models.append((i, model_name))
            output_path.parent.rmdir()
            logger.error(
                f"An error occured when trying to trace {model_name} with the error message: {e}.\n"
                f"The export is failed and {model_name} neuron model won't be stored."
            )
    logger.info(f"[Total compilation Time] {np.round(total_compilation_time, 2)} seconds.")

    # remove models failed to export
    for i, model_name in failed_models:
        output_file_names.pop(model_name)
        models_and_neuron_configs.pop(model_name)

    outputs = list(map(list, zip(*outputs)))
    return outputs


def export(
    model: "PreTrainedModel",
    config: "NeuronDefaultConfig",
    output: Path,
    compiler_workdir: Optional[Path] = None,
    inline_weights_to_neff: bool = True,
    optlevel: str = "2",
    auto_cast: Optional[str] = None,
    auto_cast_type: str = "bf16",
    disable_fast_relayout: bool = False,
    disable_fallback: bool = False,
) -> Tuple[List[str], List[str]]:
    if is_neuron_available():
        return export_neuron(model, config, output, auto_cast, auto_cast_type, disable_fast_relayout, disable_fallback)
    elif is_neuronx_available():
        return export_neuronx(
            model=model,
            config=config,
            output=output,
            compiler_workdir=compiler_workdir,
            inline_weights_to_neff=inline_weights_to_neff,
            optlevel=optlevel,
            auto_cast=auto_cast,
            auto_cast_type=auto_cast_type,
        )
    else:
        raise RuntimeError(
            "Cannot export the model because the neuron(x) compiler is not installed. See https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-setup.html."
        )


def export_neuronx(
    model: "PreTrainedModel",
    config: "NeuronDefaultConfig",
    output: Path,
    compiler_workdir: Optional[Path] = None,
    inline_weights_to_neff: bool = True,
    optlevel: str = "2",
    auto_cast: Optional[str] = None,
    auto_cast_type: str = "bf16",
) -> Tuple[List[str], List[str]]:
    """
    Exports a PyTorch model to a serialized TorchScript module compiled by neuronx-cc compiler.

    Args:
        model ([`PreTrainedModel`]):
            The model to export.
        config ([`~exporter.NeuronDefaultConfig`]):
            The Neuron configuration associated with the exported model.
        output (`Path`):
            Directory to store the exported Neuron model.
        compiler_workdir (`Optional[Path]`, defaults to `None`):
            The directory used by neuronx-cc, where you can find intermediary outputs (neff, weight, hlo...).
        inline_weights_to_neff (`bool`, defaults to `True`):
            Whether to inline the weights to the neff graph. If set to False, weights will be seperated from the neff.
        optlevel (`str`, defaults to `"2"`):
            The level of optimization the compiler should perform. Can be `"1"`, `"2"` or `"3"`, defaults to "2".
                1: enables the core performance optimizations in the compiler, while also minimizing compile time.
                2: provides the best balance between model performance and compile time.
                3: may provide additional model execution performance but may incur longer compile times and higher host memory usage during model compilation.
        auto_cast (`Optional[str]`, defaults to `None`):
            Whether to cast operations from FP32 to lower precision to speed up the inference. Can be `None`, `"matmul"` or `"all"`, you should use `None` to disable any auto-casting, use `"matmul"` to cast FP32 matrix multiplication operations, and use `"all"` to cast all FP32 operations.
        auto_cast_type (`str`, defaults to `"bf16"`):
            The data type to cast FP32 operations to when auto-cast mode is enabled. Can be `"bf16"`, `"fp16"` or `"tf32"`.

    Returns:
        `Tuple[List[str], List[str]]`: A tuple with an ordered list of the model's inputs, and the named inputs from
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
    dummy_inputs = config.flatten_inputs(dummy_inputs)
    dummy_inputs_tuple = tuple(dummy_inputs.values())

    aliases = {}
    if hasattr(model, "config") and getattr(model.config, "is_encoder_decoder", False):
        checked_model = config.patch_model_for_export(model, **input_shapes)
        if getattr(config, "is_decoder", False):
            aliases = config.generate_io_aliases(checked_model)
    else:
        checked_model = config.patch_model_for_export(model, dummy_inputs)

    if auto_cast is not None:
        logger.info(f"Using Neuron: --auto-cast {auto_cast}")

        auto_cast = "matmult" if auto_cast == "matmul" else auto_cast
        compiler_args = ["--auto-cast", auto_cast]

        logger.info(f"Using Neuron: --auto-cast-type {auto_cast_type}")
        compiler_args.extend(["--auto-cast-type", auto_cast_type])
    else:
        compiler_args = ["--auto-cast", "none"]

    compiler_args.extend(["--optlevel", optlevel])

    # diffusers specific
    compiler_args = add_stable_diffusion_compiler_args(config, compiler_args)

    neuron_model = neuronx.trace(
        checked_model,
        dummy_inputs_tuple,
        compiler_args=compiler_args,
        input_output_aliases=aliases,
        inline_weights_to_neff=inline_weights_to_neff,
        compiler_workdir=compiler_workdir,
    )

    if config.dynamic_batch_size is True:
        if not inline_weights_to_neff:
            raise ValueError(
                "Dynamic batching is not yet compatible with the weights/neff non-inlined model. Please set `dynamic_batch_size=False` or `inline_weights_to_neff=True`."
            )
        neuron_model = neuronx.dynamic_batch(neuron_model)

    # diffusers specific
    improve_stable_diffusion_loading(config, neuron_model)

    torch.jit.save(neuron_model, output)
    del model
    del checked_model
    del dummy_inputs
    del neuron_model

    return config.inputs, config.outputs


def add_stable_diffusion_compiler_args(config, compiler_args):
    # Combine the model name and its path to identify which is the subcomponent in Stable Diffusion pipeline
    identifier = getattr(config._config, "_name_or_path", "") + " " + getattr(config._config, "_class_name", "")
    identifier = identifier.lower()

    sd_components = ["text_encoder", "vae", "vae_encoder", "vae_decoder"]
    if any(component in identifier for component in sd_components):
        compiler_args.append("--enable-fast-loading-neuron-binaries")
    # unet
    if "unet" in identifier:
        # SDXL unet doesn't support fast loading neuron binaries
        if not getattr(config, "is_sdxl", False):
            compiler_args.append("--enable-fast-loading-neuron-binaries")
        compiler_args.append("--model-type=unet-inference")
    return compiler_args


def improve_stable_diffusion_loading(config, neuron_model):
    # Combine the model name and its path to identify which is the subcomponent in Stable Diffusion pipeline
    identifier = getattr(config._config, "_name_or_path", "") + " " + getattr(config._config, "_class_name", "")
    identifier = identifier.lower()
    sd_components = ["text_encoder", "unet", "vae", "vae_encoder", "vae_decoder"]
    if any(component in identifier for component in sd_components):
        neuronx.async_load(neuron_model)
    # unet
    if "unet" in identifier:
        neuronx.lazy_load(neuron_model)


def export_neuron(
    model: "PreTrainedModel",
    config: "NeuronDefaultConfig",
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
        config ([`~exporter.NeuronDefaultConfig`]):
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
    checked_model = config.patch_model_for_export(model, dummy_inputs)
    compiler_args = convert_neuronx_compiler_args_to_neuron(auto_cast, auto_cast_type, disable_fast_relayout)

    neuron_model = neuron.trace(
        checked_model,
        dummy_inputs_tuple,
        dynamic_batch_size=config.dynamic_batch_size,
        compiler_args=compiler_args,
        fallback=not disable_fallback,
    )
    torch.jit.save(neuron_model, output)
    del model
    del checked_model
    del dummy_inputs
    del neuron_model

    return config.inputs, config.outputs
