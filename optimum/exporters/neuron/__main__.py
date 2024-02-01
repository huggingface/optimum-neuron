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
"""Entry point to the optimum.exporters.neuron command line."""

import argparse
import inspect
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from requests.exceptions import ConnectionError as RequestsConnectionError
from transformers import AutoConfig, PretrainedConfig

from ...neuron import NeuronModelForCausalLM
from ...neuron.utils import (
    DECODER_NAME,
    DIFFUSION_MODEL_TEXT_ENCODER_2_NAME,
    DIFFUSION_MODEL_TEXT_ENCODER_NAME,
    DIFFUSION_MODEL_UNET_NAME,
    DIFFUSION_MODEL_VAE_DECODER_NAME,
    DIFFUSION_MODEL_VAE_ENCODER_NAME,
    ENCODER_NAME,
    NEURON_FILE_NAME,
    is_neuron_available,
    is_neuronx_available,
)
from ...neuron.utils.version_utils import check_compiler_compatibility_for_stable_diffusion
from ...utils import is_diffusers_available, logging
from ...utils.save_utils import maybe_save_preprocessors
from ..error_utils import AtolError, OutputMatchError, ShapeError
from ..tasks import TasksManager
from .base import NeuronDecoderConfig
from .convert import export_models, validate_models_outputs
from .model_configs import *  # noqa: F403
from .utils import (
    build_stable_diffusion_components_mandatory_shapes,
    get_encoder_decoder_models_for_export,
    get_stable_diffusion_models_for_export,
    replace_stable_diffusion_submodels,
)


if is_neuron_available():
    from ...commands.export.neuron import parse_args_neuron

    NEURON_COMPILER = "Neuron"


if is_neuronx_available():
    from ...commands.export.neuronx import parse_args_neuronx as parse_args_neuron  # noqa: F811

    NEURON_COMPILER = "Neuronx"

if is_diffusers_available():
    from diffusers import StableDiffusionXLPipeline


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    if is_diffusers_available():
        from diffusers import DiffusionPipeline, StableDiffusionPipeline


logger = logging.get_logger()
logger.setLevel(logging.INFO)


def infer_compiler_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    # infer compiler kwargs
    auto_cast = None if args.auto_cast == "none" else args.auto_cast
    auto_cast_type = None if auto_cast is None else args.auto_cast_type
    compiler_kwargs = {"auto_cast": auto_cast, "auto_cast_type": auto_cast_type}
    if hasattr(args, "disable_fast_relayout"):
        compiler_kwargs["disable_fast_relayout"] = getattr(args, "disable_fast_relayout")
    if hasattr(args, "disable_fallback"):
        compiler_kwargs["disable_fallback"] = getattr(args, "disable_fallback")

    return compiler_kwargs


def infer_task(task: str, model_name_or_path: str) -> str:
    if task == "auto":
        try:
            task = TasksManager.infer_task_from_model(model_name_or_path)
        except KeyError as e:
            raise KeyError(
                "The task could not be automatically inferred. Please provide the argument --task with the task "
                f"from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
            )
        except RequestsConnectionError as e:
            raise RequestsConnectionError(
                f"The task could not be automatically inferred as this is available only for models hosted on the Hugging Face Hub. Please provide the argument --task with the relevant task from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
            )
    return task


def get_input_shapes_and_config_class(task: str, args: argparse.Namespace) -> Dict[str, int]:
    config = AutoConfig.from_pretrained(args.model)

    model_type = config.model_type.replace("_", "-")
    if config.is_encoder_decoder:
        model_type = model_type + "-encoder"

    neuron_config_constructor = TasksManager.get_exporter_config_constructor(
        model_type=model_type, exporter="neuron", task=task
    )
    input_args = neuron_config_constructor.func.get_input_args_for_task(task)
    input_shapes = {name: getattr(args, name) for name in input_args}
    return input_shapes, neuron_config_constructor.func


def normalize_sentence_transformers_input_shapes(args: argparse.Namespace) -> Dict[str, int]:
    args = vars(args) if isinstance(args, argparse.Namespace) else args
    mandatory_axes = {"batch_size", "sequence_length"}
    if "clip" in args.get("model", "").lower():
        mandatory_axes.update(["num_channels", "width", "height"])
    if not mandatory_axes.issubset(set(args.keys())):
        raise AttributeError(
            f"Shape of {mandatory_axes} are mandatory for neuron compilation, while {mandatory_axes.difference(args.keys())} are not given."
        )
    mandatory_shapes = {name: args[name] for name in mandatory_axes}
    return mandatory_shapes


def customize_optional_outputs(args: argparse.Namespace) -> Dict[str, bool]:
    """
    Customize optional outputs of the traced model, eg. if `output_attentions=True`, the attentions tensors will be traced.
    """
    possible_outputs = ["output_attentions", "output_hidden_states"]

    customized_outputs = {}
    for name in possible_outputs:
        customized_outputs[name] = getattr(args, name, False)
    return customized_outputs


def parse_optlevel(args: argparse.Namespace) -> Dict[str, bool]:
    """
    Parse the level of optimization the compiler should perform. If not specified apply `O2`(the best balance between model performance and compile time).
    """
    if args.O1:
        optlevel = "1"
    elif args.O2:
        optlevel = "2"
    elif args.O3:
        optlevel = "3"
    else:
        optlevel = "2"
    return optlevel


def normalize_stable_diffusion_input_shapes(
    args: argparse.Namespace,
) -> Dict[str, Dict[str, int]]:
    args = vars(args) if isinstance(args, argparse.Namespace) else args
    mandatory_axes = set(getattr(inspect.getfullargspec(build_stable_diffusion_components_mandatory_shapes), "args"))
    # Remove `sequence_length` as diffusers will pad it to the max and remove number of channels.
    mandatory_axes = mandatory_axes - {
        "sequence_length",
        "unet_num_channels",
        "vae_encoder_num_channels",
        "vae_decoder_num_channels",
        "num_images_per_prompt",  # default to 1
    }
    if not mandatory_axes.issubset(set(args.keys())):
        raise AttributeError(
            f"Shape of {mandatory_axes} are mandatory for neuron compilation, while {mandatory_axes.difference(args.keys())} are not given."
        )
    mandatory_shapes = {name: args[name] for name in mandatory_axes}
    mandatory_shapes["num_images_per_prompt"] = args.get("num_images_per_prompt", 1)
    input_shapes = build_stable_diffusion_components_mandatory_shapes(**mandatory_shapes)
    return input_shapes


def infer_stable_diffusion_shapes_from_diffusers(
    input_shapes: Dict[str, Dict[str, int]],
    model: Union["StableDiffusionPipeline", "StableDiffusionXLPipeline"],
):
    if model.tokenizer is not None:
        sequence_length = model.tokenizer.model_max_length
    elif hasattr(model, "tokenizer_2") and model.tokenizer_2 is not None:
        sequence_length = model.tokenizer_2.model_max_length
    else:
        raise AttributeError(f"Cannot infer sequence_length from {type(model)} as there is no tokenizer as attribute.")
    unet_num_channels = model.unet.config.in_channels
    vae_encoder_num_channels = model.vae.config.in_channels
    vae_decoder_num_channels = model.vae.config.latent_channels
    vae_scale_factor = 2 ** (len(model.vae.config.block_out_channels) - 1) or 8
    height = input_shapes["unet_input_shapes"]["height"]
    scaled_height = height // vae_scale_factor
    width = input_shapes["unet_input_shapes"]["width"]
    scaled_width = width // vae_scale_factor

    input_shapes["text_encoder_input_shapes"].update({"sequence_length": sequence_length})
    input_shapes["unet_input_shapes"].update(
        {
            "sequence_length": sequence_length,
            "num_channels": unet_num_channels,
            "height": scaled_height,
            "width": scaled_width,
        }
    )
    input_shapes["vae_encoder_input_shapes"].update(
        {"num_channels": vae_encoder_num_channels, "height": height, "width": width}
    )
    input_shapes["vae_decoder_input_shapes"].update(
        {"num_channels": vae_decoder_num_channels, "height": scaled_height, "width": scaled_width}
    )

    return input_shapes


def _get_submodels_and_neuron_configs(
    model: Union["PreTrainedModel", "DiffusionPipeline"],
    input_shapes: Dict[str, int],
    task: str,
    output: Path,
    dynamic_batch_size: bool = False,
    model_name_or_path: Optional[Union[str, Path]] = None,
    submodels: Optional[Dict[str, Union[Path, str]]] = None,
    output_attentions: bool = False,
    output_hidden_states: bool = False,
):
    is_stable_diffusion = "stable-diffusion" in task
    is_encoder_decoder = (
        getattr(model.config, "is_encoder_decoder", False) if isinstance(model.config, PretrainedConfig) else False
    )

    if is_stable_diffusion:
        # TODO: Enable optional outputs for Stable Diffusion
        if output_attentions or output_hidden_states:
            raise ValueError(
                f"`output_attentions` and `output_hidden_states` are not supported by the {task} task yet."
            )
        models_and_neuron_configs, output_model_names = _get_submodels_and_neuron_configs_for_stable_diffusion(
            model, input_shapes, task, output, dynamic_batch_size, submodels
        )
    elif is_encoder_decoder:
        optional_outputs = {"output_attentions": output_attentions, "output_hidden_states": output_hidden_states}
        models_and_neuron_configs, output_model_names = _get_submodels_and_neuron_configs_for_encoder_decoder(
            model, input_shapes, task, output, dynamic_batch_size, model_name_or_path, **optional_outputs
        )
    else:
        # TODO: Enable optional outputs for encoders
        if output_attentions or output_hidden_states:
            raise ValueError(
                f"`output_attentions` and `output_hidden_states` are not supported by the {task} task yet."
            )
        neuron_config_constructor = TasksManager.get_exporter_config_constructor(
            model=model, exporter="neuron", task=task
        )
        neuron_config = neuron_config_constructor(model.config, dynamic_batch_size=dynamic_batch_size, **input_shapes)
        model_name = getattr(model, "name_or_path", None) or model_name_or_path
        model_name = model_name.split("/")[-1] if model_name else model.config.model_type
        output_model_names = {model_name: "model.neuron"}
        models_and_neuron_configs = {model_name: (model, neuron_config)}
        maybe_save_preprocessors(model_name_or_path, output)
    return models_and_neuron_configs, output_model_names


def _get_submodels_and_neuron_configs_for_stable_diffusion(
    model: Union["PreTrainedModel", "DiffusionPipeline"],
    input_shapes: Dict[str, int],
    task: str,
    output: Path,
    dynamic_batch_size: bool = False,
    submodels: Optional[Dict[str, Union[Path, str]]] = None,
):
    check_compiler_compatibility_for_stable_diffusion()
    model = replace_stable_diffusion_submodels(model, submodels)
    if is_neuron_available():
        raise RuntimeError(
            "Stable diffusion export is not supported by neuron-cc on inf1, please use neuronx-cc on either inf2/trn1 instead."
        )
    input_shapes = infer_stable_diffusion_shapes_from_diffusers(input_shapes, model)

    # Saving the model config and preprocessor as this is needed sometimes.
    model.scheduler.save_pretrained(output.joinpath("scheduler"))
    if getattr(model, "tokenizer", None) is not None:
        model.tokenizer.save_pretrained(output.joinpath("tokenizer"))
    if getattr(model, "tokenizer_2", None) is not None:
        model.tokenizer_2.save_pretrained(output.joinpath("tokenizer_2"))
    if getattr(model, "feature_extractor", None) is not None:
        model.feature_extractor.save_pretrained(output.joinpath("feature_extractor"))
    model.save_config(output)

    models_and_neuron_configs = get_stable_diffusion_models_for_export(
        pipeline=model,
        task=task,
        dynamic_batch_size=dynamic_batch_size,
        **input_shapes,
    )
    output_model_names = {
        DIFFUSION_MODEL_UNET_NAME: os.path.join(DIFFUSION_MODEL_UNET_NAME, NEURON_FILE_NAME),
        DIFFUSION_MODEL_VAE_ENCODER_NAME: os.path.join(DIFFUSION_MODEL_VAE_ENCODER_NAME, NEURON_FILE_NAME),
        DIFFUSION_MODEL_VAE_DECODER_NAME: os.path.join(DIFFUSION_MODEL_VAE_DECODER_NAME, NEURON_FILE_NAME),
    }
    if getattr(model, "text_encoder", None) is not None:
        output_model_names[DIFFUSION_MODEL_TEXT_ENCODER_NAME] = os.path.join(
            DIFFUSION_MODEL_TEXT_ENCODER_NAME, NEURON_FILE_NAME
        )
    if getattr(model, "text_encoder_2", None) is not None:
        output_model_names[DIFFUSION_MODEL_TEXT_ENCODER_2_NAME] = os.path.join(
            DIFFUSION_MODEL_TEXT_ENCODER_2_NAME, NEURON_FILE_NAME
        )
    del model

    return models_and_neuron_configs, output_model_names


def _get_submodels_and_neuron_configs_for_encoder_decoder(
    model: "PreTrainedModel",
    input_shapes: Dict[str, int],
    task: str,
    output: Path,
    dynamic_batch_size: bool = False,
    model_name_or_path: Optional[Union[str, Path]] = None,
    output_attentions: bool = False,
    output_hidden_states: bool = False,
):
    if is_neuron_available():
        raise RuntimeError(
            "Encoder-decoder models export is not supported by neuron-cc on inf1, please use neuronx-cc on either inf2/trn1 instead."
        )

    models_and_neuron_configs = get_encoder_decoder_models_for_export(
        model=model,
        task=task,
        dynamic_batch_size=dynamic_batch_size,
        input_shapes=input_shapes,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
    )
    output_model_names = {
        ENCODER_NAME: os.path.join(ENCODER_NAME, NEURON_FILE_NAME),
        DECODER_NAME: os.path.join(DECODER_NAME, NEURON_FILE_NAME),
    }
    maybe_save_preprocessors(model_name_or_path, output)

    return models_and_neuron_configs, output_model_names


def main_export(
    model_name_or_path: str,
    output: Union[str, Path],
    compiler_kwargs: Dict[str, Any],
    task: str = "auto",
    dynamic_batch_size: bool = False,
    atol: Optional[float] = None,
    cache_dir: Optional[str] = None,
    compiler_workdir: Optional[Union[str, Path]] = None,
    inline_weights_to_neff: bool = True,
    optlevel: str = "2",
    trust_remote_code: bool = False,
    subfolder: str = "",
    revision: str = "main",
    force_download: bool = False,
    local_files_only: bool = False,
    use_auth_token: Optional[Union[bool, str]] = None,
    do_validation: bool = True,
    submodels: Optional[Dict[str, Union[Path, str]]] = None,
    output_attentions: bool = False,
    output_hidden_states: bool = False,
    library_name: Optional[str] = None,
    **input_shapes,
):
    output = Path(output)
    if not output.parent.exists():
        output.parent.mkdir(parents=True)

    task = TasksManager.map_from_synonym(task)
    is_stable_diffusion = "stable-diffusion" in task

    model_kwargs = {
        "task": task,
        "model_name_or_path": model_name_or_path,
        "subfolder": subfolder,
        "revision": revision,
        "cache_dir": cache_dir,
        "use_auth_token": use_auth_token,
        "local_files_only": local_files_only,
        "force_download": force_download,
        "trust_remote_code": trust_remote_code,
        "framework": "pt",
        "library_name": library_name,
    }
    model = TasksManager.get_model_from_task(**model_kwargs)

    models_and_neuron_configs, output_model_names = _get_submodels_and_neuron_configs(
        model=model,
        input_shapes=input_shapes,
        task=task,
        output=output,
        dynamic_batch_size=dynamic_batch_size,
        model_name_or_path=model_name_or_path,
        submodels=submodels,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
    )

    _, neuron_outputs = export_models(
        models_and_neuron_configs=models_and_neuron_configs,
        output_dir=output,
        compiler_workdir=compiler_workdir,
        inline_weights_to_neff=inline_weights_to_neff,
        optlevel=optlevel,
        output_file_names=output_model_names,
        compiler_kwargs=compiler_kwargs,
    )

    # Validate compiled model
    if do_validation is True:
        if is_stable_diffusion:
            # Do not validate vae encoder due to the sampling randomness
            del neuron_outputs[-2]  # -2 is the index of `vae_encoder`
            models_and_neuron_configs.pop("vae_encoder", None)
            output_model_names.pop("vae_encoder", None)

        try:
            validate_models_outputs(
                models_and_neuron_configs=models_and_neuron_configs,
                neuron_named_outputs=neuron_outputs,
                output_dir=output,
                atol=atol,
                neuron_files_subpaths=output_model_names,
            )

            logger.info(
                f"The {NEURON_COMPILER} export succeeded and the exported model was saved at: " f"{output.as_posix()}"
            )
        except ShapeError as e:
            raise e
        except AtolError as e:
            logger.warning(
                f"The {NEURON_COMPILER} export succeeded with the warning: {e}.\n The exported model was saved at: "
                f"{output.as_posix()}"
            )
        except OutputMatchError as e:
            logger.warning(
                f"The {NEURON_COMPILER} export succeeded with the warning: {e}.\n The exported model was saved at: "
                f"{output.as_posix()}"
            )
        except Exception as e:
            logger.error(
                f"An error occured with the error message: {e}.\n The exported model was saved at: "
                f"{output.as_posix()}"
            )


def decoder_export(
    model_name_or_path: str,
    output: Union[str, Path],
    **kwargs,
):
    output = Path(output)
    if not output.parent.exists():
        output.parent.mkdir(parents=True)

    model = NeuronModelForCausalLM.from_pretrained(model_name_or_path, export=True, **kwargs)
    model.save_pretrained(output)


def main():
    parser = ArgumentParser(f"Hugging Face Optimum {NEURON_COMPILER} exporter")

    parse_args_neuron(parser)

    # Retrieve CLI arguments
    args = parser.parse_args()

    task = infer_task(args.task, args.model)
    is_stable_diffusion = "stable-diffusion" in task
    is_sentence_transformers = args.library_name == "sentence_transformers"

    if is_stable_diffusion:
        input_shapes = normalize_stable_diffusion_input_shapes(args)
        submodels = {"unet": args.unet}
    elif is_sentence_transformers:
        input_shapes = normalize_sentence_transformers_input_shapes(args)
        submodels = None
    else:
        input_shapes, neuron_config_class = get_input_shapes_and_config_class(task, args)
        if NeuronDecoderConfig in inspect.getmro(neuron_config_class):
            # TODO: warn about ignored args:
            # dynamic_batch_size, compiler_workdir, optlevel,
            # atol, disable_validation, library_name
            decoder_export(
                model_name_or_path=args.model,
                output=args.output,
                task=task,
                cache_dir=args.cache_dir,
                trust_remote_code=args.trust_remote_code,
                subfolder=args.subfolder,
                auto_cast_type=args.auto_cast_type,
                num_cores=args.num_cores,
                **input_shapes,
            )
            return
        submodels = None

    compiler_kwargs = infer_compiler_kwargs(args)
    optional_outputs = customize_optional_outputs(args)
    optlevel = parse_optlevel(args)

    main_export(
        model_name_or_path=args.model,
        output=args.output,
        compiler_kwargs=compiler_kwargs,
        task=task,
        dynamic_batch_size=args.dynamic_batch_size,
        atol=args.atol,
        cache_dir=args.cache_dir,
        compiler_workdir=args.compiler_workdir,
        inline_weights_to_neff=not args.disable_weights_neff_inline,
        optlevel=optlevel,
        trust_remote_code=args.trust_remote_code,
        subfolder=args.subfolder,
        do_validation=not args.disable_validation,
        submodels=submodels,
        library_name=args.library_name,
        **optional_outputs,
        **input_shapes,
    )


if __name__ == "__main__":
    main()
