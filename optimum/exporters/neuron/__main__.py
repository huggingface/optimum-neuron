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


os.environ["TORCHDYNAMO_DISABLE"] = "1"  # Always turn off torchdynamo as it's incompatible with neuron
from argparse import ArgumentParser
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from optimum.exporters.error_utils import AtolError, OutputMatchError, ShapeError
from optimum.exporters.tasks import TasksManager
from optimum.utils import is_diffusers_available, logging
from optimum.utils.save_utils import maybe_load_preprocessors, maybe_save_preprocessors
from requests.exceptions import ConnectionError as RequestsConnectionError
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig

from ...neuron.models.auto_model import get_neuron_model_class, has_neuron_model_class
from ...neuron.utils import (
    DECODER_NAME,
    DIFFUSION_MODEL_CONTROLNET_NAME,
    DIFFUSION_MODEL_TEXT_ENCODER_2_NAME,
    DIFFUSION_MODEL_TEXT_ENCODER_NAME,
    DIFFUSION_MODEL_TRANSFORMER_NAME,
    DIFFUSION_MODEL_UNET_NAME,
    DIFFUSION_MODEL_VAE_DECODER_NAME,
    DIFFUSION_MODEL_VAE_ENCODER_NAME,
    DTYPE_MAPPER,
    ENCODER_NAME,
    NEURON_FILE_NAME,
    ImageEncoderArguments,
    InputShapesArguments,
    IPAdapterArguments,
    LoRAAdapterArguments,
    is_neuron_available,
    is_neuronx_available,
)
from ...neuron.utils.version_utils import (
    check_compiler_compatibility_for_stable_diffusion,
)
from .base import NeuronExportConfig
from .convert import export_models, validate_models_outputs
from .model_configs import *  # noqa: F403
from .utils import (
    build_stable_diffusion_components_mandatory_shapes,
    check_mandatory_input_shapes,
    get_diffusion_models_for_export,
    get_encoder_decoder_models_for_export,
    replace_stable_diffusion_submodels,
)


if is_neuron_available():
    from ...commands.export.neuron import parse_args_neuron

    NEURON_COMPILER = "Neuron"


if is_neuronx_available():
    from ...commands.export.neuronx import parse_args_neuronx as parse_args_neuron  # noqa: F811

    NEURON_COMPILER = "Neuronx"

if is_diffusers_available():
    from diffusers import (
        DiffusionPipeline,
        FluxKontextPipeline,
        FluxPipeline,
        QwenImagePipeline,
        ModelMixin,
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
    )

if TYPE_CHECKING:
    from transformers import PreTrainedModel


logger = logging.get_logger()
logger.setLevel(logging.INFO)


def infer_compiler_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    # infer compiler kwargs
    auto_cast = None if args.auto_cast == "none" else args.auto_cast
    auto_cast_type = None if auto_cast is None else args.auto_cast_type
    compiler_kwargs = {"auto_cast": auto_cast, "auto_cast_type": auto_cast_type}
    if hasattr(args, "disable_fast_relayout"):
        compiler_kwargs["disable_fast_relayout"] = getattr(args, "disable_fast_relayout")
    if hasattr(args, "disable_fallback"):
        compiler_kwargs["disable_fallback"] = getattr(args, "disable_fallback")

    return compiler_kwargs


def infer_task(model_name_or_path: str) -> str:
    try:
        return TasksManager.infer_task_from_model(model_name_or_path)
    except KeyError as e:
        raise KeyError(
            "The task could not be automatically inferred. Please provide the argument --task with the task "
            f"from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
        )
    except RequestsConnectionError as e:
        raise RequestsConnectionError(
            f"The task could not be automatically inferred as this is available only for models hosted on the Hugging Face Hub. Please provide the argument --task with the relevant task from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
        )


# This function is not applicable for diffusers / sentence transformers models
def get_input_shapes(task: str, args: argparse.Namespace) -> dict[str, int]:
    neuron_config_constructor = get_neuron_config_class(task, args.model)
    input_args = neuron_config_constructor.func.get_input_args_for_task(task)
    return {name: getattr(args, name) for name in input_args}


def get_neuron_config_class(task: str, model_id: str) -> NeuronExportConfig:
    config = AutoConfig.from_pretrained(model_id)

    model_type = config.model_type.replace("_", "-")
    if config.is_encoder_decoder:
        model_type = model_type + "-encoder"

    neuron_config_constructor = TasksManager.get_exporter_config_constructor(
        model_type=model_type,
        exporter="neuron",
        task=task,
        library_name="transformers",
    )
    return neuron_config_constructor


def normalize_sentence_transformers_input_shapes(args: argparse.Namespace) -> dict[str, int]:
    args = vars(args) if isinstance(args, argparse.Namespace) else args
    if "clip" in args.get("model", "").lower():
        mandatory_axes = {"text_batch_size", "image_batch_size", "sequence_length", "num_channels", "width", "height"}
    else:
        mandatory_axes = {"batch_size", "sequence_length"}

    if not mandatory_axes.issubset(set(args.keys())):
        raise AttributeError(
            f"Shape of {mandatory_axes} are mandatory for neuron compilation, while {mandatory_axes.difference(args.keys())} are not given."
        )
    mandatory_shapes = {name: args[name] for name in mandatory_axes}
    return mandatory_shapes


def customize_optional_outputs(args: argparse.Namespace) -> dict[str, bool]:
    """
    Customize optional outputs of the traced model, eg. if `output_attentions=True`, the attentions tensors will be traced.
    """
    possible_outputs = ["output_attentions", "output_hidden_states"]

    customized_outputs = {}
    for name in possible_outputs:
        customized_outputs[name] = getattr(args, name, False)
    return customized_outputs


def parse_optlevel(args: argparse.Namespace) -> dict[str, bool]:
    """
    (NEURONX ONLY) Parse the level of optimization the compiler should perform. If not specified apply `O2`(the best balance between model performance and compile time).
    """
    if is_neuronx_available():
        if args.O1:
            optlevel = "1"
        elif args.O2:
            optlevel = "2"
        elif args.O3:
            optlevel = "3"
        else:
            optlevel = "2"
    else:
        optlevel = None
    return optlevel


def normalize_diffusers_input_shapes(
    args: argparse.Namespace,
) -> dict[str, dict[str, int]]:
    args = vars(args) if isinstance(args, argparse.Namespace) else args
    mandatory_axes = set(getattr(inspect.getfullargspec(build_stable_diffusion_components_mandatory_shapes), "args"))
    mandatory_axes = mandatory_axes - {
        "sequence_length",  # `sequence_length` is optional, diffusers will pad it to the max if not provided.
        # remove number of channels.
        "unet_or_transformer_num_channels",
        "vae_encoder_num_channels",
        "vae_decoder_num_channels",
        "num_images_per_prompt",  # default to 1
    }
    if not mandatory_axes.issubset(set(args.keys())):
        raise AttributeError(
            f"Shape of {mandatory_axes} are mandatory for neuron compilation, while {mandatory_axes.difference(args.keys())} are not given."
        )
    mandatory_shapes = {name: args[name] for name in mandatory_axes}
    mandatory_shapes["num_images_per_prompt"] = args.get("num_images_per_prompt", 1) or 1
    mandatory_shapes["sequence_length"] = args.get("sequence_length", None)
    input_shapes = build_stable_diffusion_components_mandatory_shapes(**mandatory_shapes)
    return input_shapes


def infer_shapes_of_diffusers(
    input_shapes: dict[str, dict[str, int]],
    model: "StableDiffusionPipeline | StableDiffusionXLPipeline | FluxPipeline | FluxKontextPipeline",
    has_controlnets: bool,
):
    max_sequence_length_1 = model.tokenizer.model_max_length if model.tokenizer is not None else None
    max_sequence_length_2 = (
        model.tokenizer_2.model_max_length if hasattr(model, "tokenizer_2") and model.tokenizer_2 is not None else None
    )
    if isinstance(model, (FluxPipeline, FluxKontextPipeline)):
        max_sequence_length_2 = input_shapes["text_encoder"].get("sequence_length", None) or max_sequence_length_2
        
    if isinstance(model, QwenImagePipeline):
        vae_encoder_num_channels = 3
        vae_decoder_num_channels = model.vae.config.z_dim
        vae_scale_factor = 2 ** len(model.vae.temperal_downsample) or 8
    else:
        vae_encoder_num_channels = model.vae.config.in_channels
        vae_decoder_num_channels = model.vae.config.latent_channels
        vae_scale_factor = 2 ** (len(model.vae.config.block_out_channels) - 1) or 8
    height = input_shapes["unet_or_transformer"]["height"]
    width = input_shapes["unet_or_transformer"]["width"]
    scaled_height = 2 * (int(height) // (vae_scale_factor * 2))
    scaled_width = 2 * (int(width) // (vae_scale_factor * 2))

    # Text encoders
    if input_shapes["text_encoder"].get("sequence_length") is None or hasattr(model, "text_encoder_2"):
        input_shapes["text_encoder"].update({"sequence_length": max_sequence_length_1})
    if hasattr(model, "text_encoder_2"):
        input_shapes["text_encoder_2"] = {
            "batch_size": input_shapes["text_encoder"]["batch_size"],
            "sequence_length": max_sequence_length_2,
        }

    # UNet or Transformer
    unet_or_transformer_name = "transformer" if hasattr(model, "transformer") else "unet"
    unet_or_transformer_num_channels = getattr(model, unet_or_transformer_name).config.in_channels
    input_shapes["unet_or_transformer"].update(
        {
            "num_channels": unet_or_transformer_num_channels,
            "height": scaled_height,
            "width": scaled_width,
        }
    )
    if input_shapes["unet_or_transformer"].get("sequence_length") is None:
        input_shapes["unet_or_transformer"]["sequence_length"] = max_sequence_length_2 or max_sequence_length_1
    input_shapes["unet_or_transformer"]["vae_scale_factor"] = vae_scale_factor
    input_shapes[unet_or_transformer_name] = input_shapes.pop("unet_or_transformer")
    if unet_or_transformer_name == "transformer":
        input_shapes[unet_or_transformer_name]["encoder_hidden_size"] = model.text_encoder.config.hidden_size
        if hasattr(model.transformer, "pos_embed") and hasattr(model.transformer.pos_embed, "axes_dim"):
            input_shapes[unet_or_transformer_name]["rotary_axes_dim"] = sum(model.transformer.pos_embed.axes_dim)

    # VAE
    input_shapes["vae_encoder"].update({"num_channels": vae_encoder_num_channels, "height": height, "width": width})
    input_shapes["vae_decoder"].update(
        {"num_channels": vae_decoder_num_channels, "height": scaled_height, "width": scaled_width}
    )

    # ControlNet
    if has_controlnets:
        encoder_hidden_size = model.text_encoder.config.hidden_size
        if hasattr(model, "text_encoder_2"):
            encoder_hidden_size += model.text_encoder_2.config.hidden_size
        input_shapes["controlnet"] = {
            "batch_size": input_shapes[unet_or_transformer_name]["batch_size"],
            "sequence_length": input_shapes[unet_or_transformer_name]["sequence_length"],
            "num_channels": unet_or_transformer_num_channels,
            "height": scaled_height,
            "width": scaled_width,
            "vae_scale_factor": vae_scale_factor,
            "encoder_hidden_size": encoder_hidden_size,
        }

    # Image encoder
    if getattr(model, "image_encoder", None):
        input_shapes["image_encoder"] = {
            "batch_size": input_shapes[unet_or_transformer_name]["batch_size"],
            "num_channels": model.image_encoder.config.num_channels,
            "width": model.image_encoder.config.image_size,
            "height": model.image_encoder.config.image_size,
        }
        # IP-Adapter: add image_embeds as input for unet/transformer
        # unet has `ip_adapter_image_embeds` with shape [batch_size, 1, (self.image_encoder.config.image_size//patch_size)**2+1, self.image_encoder.config.hidden_size] as input
        if getattr(model.unet.config, "encoder_hid_dim_type", None) == "ip_image_proj":
            input_shapes[unet_or_transformer_name]["image_encoder_shapes"] = ImageEncoderArguments(
                sequence_length=model.image_encoder.vision_model.embeddings.position_embedding.weight.shape[0],
                hidden_size=model.image_encoder.vision_model.embeddings.position_embedding.weight.shape[1],
                projection_dim=getattr(model.image_encoder.config, "projection_dim", None),
            )

    # Format with `InputShapesArguments`
    for sub_model_name in input_shapes.keys():
        input_shapes[sub_model_name] = InputShapesArguments(**input_shapes[sub_model_name])
    return input_shapes


def get_submodels_and_neuron_configs(
    model: "PreTrainedModel | DiffusionPipeline",
    input_shapes: dict[str, int],
    task: str,
    output: Path,
    library_name: str,
    tensor_parallel_size: int = 1,
    subfolder: str = "",
    trust_remote_code: bool = False,
    dynamic_batch_size: bool = False,
    model_name_or_path: str | Path | None = None,
    submodels: dict[str, Path | str] | None = None,
    output_attentions: bool = False,
    output_hidden_states: bool = False,
    controlnet_ids: str | list[str] | None = None,
    lora_args: LoRAAdapterArguments | None = None,
):
    is_encoder_decoder = (
        getattr(model.config, "is_encoder_decoder", False) if isinstance(model.config, PretrainedConfig) else False
    )

    if library_name == "diffusers":
        # TODO: Enable optional outputs for Stable Diffusion
        if output_attentions:
            raise ValueError(f"`output_attentions`is not supported by the {task} task yet.")
        # Custom lowering for Softmax and SILU operations. Mandatory for applying optimized attention score of diffusion models.
        os.environ["NEURON_FUSE_SOFTMAX"] = "1"
        os.environ["NEURON_CUSTOM_SILU"] = "1"
        models_and_neuron_configs, output_model_names = _get_submodels_and_neuron_configs_for_diffusion(
            model=model,
            input_shapes=input_shapes,
            tensor_parallel_size=tensor_parallel_size,
            output=output,
            dynamic_batch_size=dynamic_batch_size,
            model_name_or_path=model_name_or_path,
            submodels=submodels,
            output_hidden_states=output_hidden_states,
            controlnet_ids=controlnet_ids,
            lora_args=lora_args,
        )
    elif is_encoder_decoder:
        optional_outputs = {"output_attentions": output_attentions, "output_hidden_states": output_hidden_states}
        preprocessors = maybe_load_preprocessors(
            src_name_or_path=model_name_or_path,
            subfolder=subfolder,
            trust_remote_code=trust_remote_code,
        )
        models_and_neuron_configs, output_model_names = _get_submodels_and_neuron_configs_for_encoder_decoder(
            model=model,
            input_shapes=input_shapes,
            tensor_parallel_size=tensor_parallel_size,
            task=task,
            output=output,
            dynamic_batch_size=dynamic_batch_size,
            model_name_or_path=model_name_or_path,
            preprocessors=preprocessors,
            **optional_outputs,
        )
    else:
        # TODO: Enable optional outputs for encoders
        if output_attentions or output_hidden_states:
            raise ValueError(
                f"`output_attentions` and `output_hidden_states` are not supported by the {task} task yet."
            )
        neuron_config_constructor = TasksManager.get_exporter_config_constructor(
            model=model,
            exporter="neuron",
            task=task,
            library_name=library_name,
        )
        input_shapes = check_mandatory_input_shapes(neuron_config_constructor, task, input_shapes)
        input_shapes = InputShapesArguments(**input_shapes)
        neuron_config = neuron_config_constructor(
            model.config, dynamic_batch_size=dynamic_batch_size, input_shapes=input_shapes
        )
        model_name = getattr(model, "name_or_path", None) or model_name_or_path
        model_name = model_name.split("/")[-1] if model_name else model.config.model_type
        output_model_names = {model_name: "model.neuron"}
        models_and_neuron_configs = {model_name: (model, neuron_config)}
        maybe_save_preprocessors(model_name_or_path, output, src_subfolder=subfolder)

    models_and_neuron_configs = _reorder_models_and_neuron_configs(models_and_neuron_configs)

    return models_and_neuron_configs, output_model_names


def _reorder_models_and_neuron_configs(models_and_neuron_configs):
    """
    Reorder to ensure that the export starts with NxD backend in case of TP(otherwise, runtime error).
    """
    tp_model = next(
        (
            model_name
            for model_name, (_, config) in models_and_neuron_configs.items()
            if getattr(config, "tensor_parallel_size", 1) > 1
        ),
        None,
    )

    if tp_model:
        models_and_neuron_configs = {
            tp_model: models_and_neuron_configs[tp_model],
            **{k: v for k, v in models_and_neuron_configs.items() if k != tp_model},
        }

    return models_and_neuron_configs


def _get_submodels_and_neuron_configs_for_diffusion(
    model: "PreTrainedModel | DiffusionPipeline",
    input_shapes: dict[str, int],
    tensor_parallel_size: int,
    output: Path,
    dynamic_batch_size: bool = False,
    model_name_or_path: str | Path | None = None,
    submodels: dict[str, Path | str] | None = None,
    output_hidden_states: bool = False,
    controlnet_ids: str | list[str] | None = None,
    lora_args: LoRAAdapterArguments | None = None,
):
    check_compiler_compatibility_for_stable_diffusion()
    model = replace_stable_diffusion_submodels(model, submodels)
    if is_neuron_available():
        raise RuntimeError(
            "Stable diffusion export is not supported by neuron-cc on inf1, please use neuronx-cc on either inf2/trn1 instead."
        )
    input_shapes = infer_shapes_of_diffusers(
        input_shapes=input_shapes,
        model=model,
        has_controlnets=controlnet_ids is not None,
    )
    # Saving the model config and preprocessor as this is needed sometimes.
    model.scheduler.save_pretrained(output.joinpath("scheduler"))
    if getattr(model, "tokenizer", None) is not None:
        model.tokenizer.save_pretrained(output.joinpath("tokenizer"))
    if getattr(model, "tokenizer_2", None) is not None:
        model.tokenizer_2.save_pretrained(output.joinpath("tokenizer_2"))
    if getattr(model, "tokenizer_3", None) is not None:
        model.tokenizer_3.save_pretrained(output.joinpath("tokenizer_3"))
    if getattr(model, "feature_extractor", None) is not None:
        model.feature_extractor.save_pretrained(output.joinpath("feature_extractor"))
    model.save_config(output)

    models_and_neuron_configs = get_diffusion_models_for_export(
        pipeline=model,
        tensor_parallel_size=tensor_parallel_size,
        text_encoder_input_shapes=input_shapes["text_encoder"],
        unet_input_shapes=input_shapes.get("unet", None),
        transformer_input_shapes=input_shapes.get("transformer", None),
        vae_encoder_input_shapes=input_shapes["vae_encoder"],
        vae_decoder_input_shapes=input_shapes["vae_decoder"],
        lora_args=lora_args,
        dynamic_batch_size=dynamic_batch_size,
        output_hidden_states=output_hidden_states,
        controlnet_ids=controlnet_ids,
        controlnet_input_shapes=input_shapes.get("controlnet", None),
        image_encoder_input_shapes=input_shapes.get("image_encoder", None),
        text_encoder_2_input_shapes=input_shapes.get("text_encoder_2", None),
        model_name_or_path=model_name_or_path,
    )
    output_model_names = {
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
    if getattr(model, "unet", None) is not None:
        output_model_names[DIFFUSION_MODEL_UNET_NAME] = os.path.join(DIFFUSION_MODEL_UNET_NAME, NEURON_FILE_NAME)
    if getattr(model, "transformer", None) is not None:
        output_model_names[DIFFUSION_MODEL_TRANSFORMER_NAME] = os.path.join(
            DIFFUSION_MODEL_TRANSFORMER_NAME, NEURON_FILE_NAME
        )
    if getattr(model, "image_encoder", None) is not None:
        output_model_names["image_encoder"] = os.path.join("image_encoder", NEURON_FILE_NAME)

    # ControlNet models
    if controlnet_ids:
        if isinstance(controlnet_ids, str):
            controlnet_ids = [controlnet_ids]
        for idx in range(len(controlnet_ids)):
            controlnet_name = DIFFUSION_MODEL_CONTROLNET_NAME + "_" + str(idx)
            output_model_names[controlnet_name] = os.path.join(controlnet_name, NEURON_FILE_NAME)

    del model

    return models_and_neuron_configs, output_model_names


def _get_submodels_and_neuron_configs_for_encoder_decoder(
    model: "PreTrainedModel",
    input_shapes: dict[str, int],
    tensor_parallel_size: int,
    task: str,
    output: Path,
    preprocessors: list | None = None,
    dynamic_batch_size: bool = False,
    model_name_or_path: str | Path | None = None,
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
        tensor_parallel_size=tensor_parallel_size,
        dynamic_batch_size=dynamic_batch_size,
        input_shapes=input_shapes,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        model_name_or_path=model_name_or_path,
        preprocessors=preprocessors,
    )
    output_model_names = {
        ENCODER_NAME: os.path.join(ENCODER_NAME, NEURON_FILE_NAME),
        DECODER_NAME: os.path.join(DECODER_NAME, NEURON_FILE_NAME),
    }
    model.config.save_pretrained(output)
    model.generation_config.save_pretrained(output)
    maybe_save_preprocessors(model_name_or_path, output)

    return models_and_neuron_configs, output_model_names


def load_models_and_neuron_configs(
    model_name_or_path: str,
    output: Path,
    model: "PreTrainedModel | ModelMixin | None",
    task: str,
    dynamic_batch_size: bool,
    cache_dir: str | None,
    trust_remote_code: bool,
    subfolder: str,
    revision: str,
    library_name: str,
    force_download: bool,
    local_files_only: bool,
    token: bool | str | None,
    submodels: dict[str, Path | str] | None,
    torch_dtype: str | torch.dtype = torch.float32,
    tensor_parallel_size: int = 1,
    controlnet_ids: str | list[str] | None = None,
    lora_args: LoRAAdapterArguments | None = None,
    ip_adapter_args: IPAdapterArguments | None = None,
    output_attentions: bool = False,
    output_hidden_states: bool = False,
    **input_shapes,
):
    model_kwargs = {
        "task": task,
        "model_name_or_path": model_name_or_path,
        "subfolder": subfolder,
        "revision": revision,
        "cache_dir": cache_dir,
        "token": token,
        "local_files_only": local_files_only,
        "force_download": force_download,
        "trust_remote_code": trust_remote_code,
        "framework": "pt",
        "library_name": library_name,
        "torch_dtype": torch_dtype,
    }
    if model is None:
        model = TasksManager.get_model_from_task(**model_kwargs)
        # Load IP-Adapter if it exists
        if ip_adapter_args is not None and not all(
            getattr(ip_adapter_args, field.name) is None for field in fields(ip_adapter_args)
        ):
            model.load_ip_adapter(
                ip_adapter_args.model_id, subfolder=ip_adapter_args.subfolder, weight_name=ip_adapter_args.weight_name
            )
            model.set_ip_adapter_scale(scale=ip_adapter_args.scale)

    models_and_neuron_configs, output_model_names = get_submodels_and_neuron_configs(
        model=model,
        input_shapes=input_shapes,
        tensor_parallel_size=tensor_parallel_size,
        task=task,
        library_name=library_name,
        output=output,
        subfolder=subfolder,
        trust_remote_code=trust_remote_code,
        dynamic_batch_size=dynamic_batch_size,
        model_name_or_path=model_name_or_path,
        submodels=submodels,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        controlnet_ids=controlnet_ids,
        lora_args=lora_args,
    )

    return models_and_neuron_configs, output_model_names


def main_export(
    model_name_or_path: str,
    output: str | Path,
    compiler_kwargs: dict[str, Any],
    torch_dtype: str | torch.dtype = torch.float32,
    tensor_parallel_size: int = 1,
    model: "PreTrainedModel | ModelMixin | None" = None,
    task: str = "auto",
    dynamic_batch_size: bool = False,
    atol: float | None = None,
    cache_dir: str | None = None,
    disable_neuron_cache: bool | None = False,
    compiler_workdir: str | Path | None = None,
    inline_weights_to_neff: bool = True,
    optlevel: str = "2",
    trust_remote_code: bool = False,
    subfolder: str = "",
    revision: str = "main",
    force_download: bool = False,
    local_files_only: bool = False,
    token: bool | str | None = None,
    do_validation: bool = True,
    submodels: dict[str, Path | str] | None = None,
    output_attentions: bool = False,
    output_hidden_states: bool = False,
    library_name: str | None = None,
    controlnet_ids: str | list[str] | None = None,
    lora_args: LoRAAdapterArguments | None = None,
    ip_adapter_args: IPAdapterArguments | None = None,
    **input_shapes,
):
    output = Path(output)
    torch_dtype = DTYPE_MAPPER.pt(torch_dtype)
    if not output.parent.exists():
        output.parent.mkdir(parents=True)

    task = TasksManager.map_from_synonym(task)
    if library_name is None:
        library_name = TasksManager.infer_library_from_model(
            model_name_or_path, revision=revision, cache_dir=cache_dir, token=token
        )

    models_and_neuron_configs, output_model_names = load_models_and_neuron_configs(
        model_name_or_path=model_name_or_path,
        output=output,
        model=model,
        torch_dtype=torch_dtype,
        tensor_parallel_size=tensor_parallel_size,
        task=task,
        dynamic_batch_size=dynamic_batch_size,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        subfolder=subfolder,
        revision=revision,
        library_name=library_name,
        force_download=force_download,
        local_files_only=local_files_only,
        token=token,
        submodels=submodels,
        lora_args=lora_args,
        ip_adapter_args=ip_adapter_args,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        controlnet_ids=controlnet_ids,
        **input_shapes,
    )

    _, neuron_outputs = export_models(
        models_and_neuron_configs=models_and_neuron_configs,
        task=task,
        output_dir=output,
        disable_neuron_cache=disable_neuron_cache,
        compiler_workdir=compiler_workdir,
        inline_weights_to_neff=inline_weights_to_neff,
        optlevel=optlevel,
        output_file_names=output_model_names,
        compiler_kwargs=compiler_kwargs,
        model_name_or_path=model_name_or_path,
    )

    # Validate compiled model
    if do_validation and tensor_parallel_size > 1:
        # TODO: support the validation of tp models.
        logger.warning(
            "The validation is not yet supported for tensor parallel model, the validation will be turned off."
        )
        do_validation = False
    if do_validation is True:
        try:
            validate_models_outputs(
                models_and_neuron_configs=models_and_neuron_configs,
                neuron_named_outputs=neuron_outputs,
                output_dir=output,
                atol=atol,
                neuron_files_subpaths=output_model_names,
            )

            logger.info(
                f"The {NEURON_COMPILER} export succeeded and the exported model was saved at: {output.as_posix()}"
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
                f"An error occurred with the error message: {e}.\n The exported model was saved at: {output.as_posix()}"
            )


def maybe_export_from_neuron_model_class(
    model: str,
    output: str | Path,
    task: str = "auto",
    cache_dir: str | None = None,
    subfolder: str = "",
    trust_remote_code: bool = False,
    **kwargs,
):
    """Export the model from the neuron model class if it exists."""
    if task == "auto":
        task = infer_task(model)
    output = Path(output)
    # Remove None values from the kwargs
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    # Also remove some arguments that are not supported in this context
    kwargs.pop("disable_neuron_cache", None)
    kwargs.pop("inline_weights_neff", None)
    kwargs.pop("O1", None)
    kwargs.pop("O2", None)
    kwargs.pop("O3", None)
    kwargs.pop("disable_validation", None)
    kwargs.pop("dynamic_batch_size", None)
    kwargs.pop("output_hidden_states", None)
    kwargs.pop("output_attentions", None)
    kwargs.pop("tensor_parallel_size", None)
    # Fetch the model config
    config = AutoConfig.from_pretrained(model)
    if task == "text-generation":
        # In case a multi-modal model is being exported, extract the text model config
        config = config.get_text_config()
    # Check if we have an auto-model class for the model_type and task
    if not has_neuron_model_class(model_type=config.model_type, task=task, mode="inference"):
        return False
    neuron_model_class = get_neuron_model_class(model_type=config.model_type, task=task, mode="inference")
    batch_size = kwargs.pop("batch_size", None)
    sequence_length = kwargs.pop("sequence_length", None)
    tensor_parallel_size = kwargs.pop("num_cores", None)
    auto_cast_type = kwargs.pop("auto_cast_type", None)
    neuron_config = neuron_model_class.get_neuron_config(
        model_name_or_path=model,
        config=config,
        token=kwargs.get("token", None),
        revision=kwargs.get("revision", "main"),
        batch_size=batch_size,
        sequence_length=sequence_length,
        tensor_parallel_size=tensor_parallel_size,
        auto_cast_type=auto_cast_type,
    )
    neuron_model = neuron_model_class.export(
        model_id=model,
        config=config,
        neuron_config=neuron_config,
        cache_dir=cache_dir,
        subfolder=subfolder,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )
    if not output.parent.exists():
        output.parent.mkdir(parents=True)
    neuron_model.save_pretrained(output)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
        tokenizer.save_pretrained(output)
    except Exception:
        logger.info(f"No tokenizer found while exporting {model}.")
    return True


def main():
    parser = ArgumentParser(f"Hugging Face Optimum {NEURON_COMPILER} exporter")

    parse_args_neuron(parser)

    # Retrieve CLI arguments
    args = parser.parse_args()

    task = infer_task(args.model) if args.task == "auto" else args.task
    library_name = TasksManager.infer_library_from_model(args.model, cache_dir=args.cache_dir)

    if library_name == "diffusers":
        input_shapes = normalize_diffusers_input_shapes(args)
        submodels = {"unet": args.unet}
    elif library_name == "sentence_transformers":
        input_shapes = normalize_sentence_transformers_input_shapes(args)
        submodels = None
    else:
        # New export mode using dedicated neuron model classes
        kwargs = vars(args).copy()
        if maybe_export_from_neuron_model_class(**kwargs):
            return
        # Fallback to legacy export
        input_shapes = get_input_shapes(task, args)
        submodels = None

    disable_neuron_cache = args.disable_neuron_cache
    compiler_kwargs = infer_compiler_kwargs(args)
    optional_outputs = customize_optional_outputs(args)
    optlevel = parse_optlevel(args)
    lora_args = LoRAAdapterArguments(
        model_ids=getattr(args, "lora_model_ids", None),
        weight_names=getattr(args, "lora_weight_names", None),
        adapter_names=getattr(args, "lora_adapter_names", None),
        scales=getattr(args, "lora_scales", None),
    )
    ip_adapter_args = IPAdapterArguments(
        model_id=getattr(args, "ip_adapter_id", None),
        subfolder=getattr(args, "ip_adapter_subfolder", None),
        weight_name=getattr(args, "ip_adapter_weight_name", None),
        scale=getattr(args, "ip_adapter_scale", None),
    )

    main_export(
        model_name_or_path=args.model,
        output=args.output,
        compiler_kwargs=compiler_kwargs,
        torch_dtype=args.torch_dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        task=task,
        dynamic_batch_size=args.dynamic_batch_size,
        atol=args.atol,
        cache_dir=args.cache_dir,
        disable_neuron_cache=disable_neuron_cache,
        compiler_workdir=args.compiler_workdir,
        inline_weights_to_neff=args.inline_weights_neff,
        optlevel=optlevel,
        trust_remote_code=args.trust_remote_code,
        subfolder=args.subfolder,
        do_validation=not args.disable_validation,
        submodels=submodels,
        library_name=library_name,
        controlnet_ids=getattr(args, "controlnet_ids", None),
        lora_args=lora_args,
        ip_adapter_args=ip_adapter_args,
        **optional_outputs,
        **input_shapes,
    )


if __name__ == "__main__":
    main()
