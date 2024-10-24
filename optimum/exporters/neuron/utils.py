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
"""Utility functions for neuron export."""

import copy
import os
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch

from ...neuron.utils import (
    DECODER_NAME,
    DIFFUSION_MODEL_CONTROLNET_NAME,
    DIFFUSION_MODEL_TEXT_ENCODER_2_NAME,
    DIFFUSION_MODEL_TEXT_ENCODER_NAME,
    DIFFUSION_MODEL_UNET_NAME,
    DIFFUSION_MODEL_VAE_DECODER_NAME,
    DIFFUSION_MODEL_VAE_ENCODER_NAME,
    ENCODER_NAME,
    get_attention_scores_sd,
    get_attention_scores_sdxl,
)
from ...utils import (
    DIFFUSERS_MINIMUM_VERSION,
    check_if_diffusers_greater,
    is_diffusers_available,
    logging,
)
from ...utils.import_utils import _diffusers_version
from ..tasks import TasksManager


logger = logging.get_logger()


if is_diffusers_available():
    if not check_if_diffusers_greater(DIFFUSERS_MINIMUM_VERSION.base_version):
        raise ImportError(
            f"We found an older version of diffusers {_diffusers_version} but we require diffusers to be >= {DIFFUSERS_MINIMUM_VERSION}. "
            "Please update diffusers by running `pip install --upgrade diffusers`"
        )
    from diffusers import (
        ControlNetModel,
        ModelMixin,
        StableDiffusionPipeline,
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLInpaintPipeline,
        StableDiffusionXLPipeline,
        UNet2DConditionModel,
    )
    from diffusers.models.attention_processor import Attention


if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel

    from .base import NeuronDefaultConfig


def build_stable_diffusion_components_mandatory_shapes(
    batch_size: Optional[int] = None,
    sequence_length: Optional[int] = None,
    unet_num_channels: Optional[int] = None,
    vae_encoder_num_channels: Optional[int] = None,
    vae_decoder_num_channels: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_images_per_prompt: Optional[int] = 1,
):
    text_encoder_input_shapes = {"batch_size": batch_size, "sequence_length": sequence_length}
    vae_encoder_input_shapes = {
        "batch_size": batch_size * num_images_per_prompt,
        "num_channels": vae_encoder_num_channels,
        "height": height,
        "width": width,
    }
    vae_decoder_input_shapes = {
        "batch_size": batch_size * num_images_per_prompt,
        "num_channels": vae_decoder_num_channels,
        "height": height,
        "width": width,
    }
    unet_input_shapes = {
        "batch_size": batch_size * num_images_per_prompt,
        "sequence_length": sequence_length,
        "num_channels": unet_num_channels,
        "height": height,
        "width": width,
    }

    components_shapes = {
        "text_encoder": text_encoder_input_shapes,
        "unet": unet_input_shapes,
        "vae_encoder": vae_encoder_input_shapes,
        "vae_decoder": vae_decoder_input_shapes,
    }

    return components_shapes


def get_stable_diffusion_models_for_export(
    pipeline: Union["StableDiffusionPipeline", "StableDiffusionXLPipeline"],
    text_encoder_input_shapes: Dict[str, int],
    unet_input_shapes: Dict[str, int],
    vae_encoder_input_shapes: Dict[str, int],
    vae_decoder_input_shapes: Dict[str, int],
    dynamic_batch_size: Optional[bool] = False,
    output_hidden_states: bool = False,
    lora_model_ids: Optional[List[str]] = None,
    lora_weight_names: Optional[List[str]] = None,
    lora_adapter_names: Optional[List[str]] = None,
    lora_scales: Optional[List[float]] = None,
    controlnet_ids: Optional[Union[str, List[str]]] = None,
    controlnet_input_shapes: Optional[Dict[str, int]] = None,
) -> Dict[str, Tuple[Union["PreTrainedModel", "ModelMixin"], "NeuronDefaultConfig"]]:
    """
    Returns the components of a Stable Diffusion model and their subsequent neuron configs.
    These components are chosen because they represent the bulk of the compute in the pipeline,
    and performance benchmarking has shown that running them on Neuron yields significant
    performance benefit (CLIP text encoder, VAE encoder, VAE decoder, Unet).

    Args:
        pipeline ([`Union["StableDiffusionPipeline", "StableDiffusionXLPipeline"]`]):
            The model to export.
        text_encoder_input_shapes (`Dict[str, int]`):
            Static shapes used for compiling text encoder.
        unet_input_shapes (`Dict[str, int]`):
            Static shapes used for compiling unet.
        vae_encoder_input_shapes (`Dict[str, int]`):
            Static shapes used for compiling vae encoder.
        vae_decoder_input_shapes (`Dict[str, int]`):
            Static shapes used for compiling vae decoder.
        dynamic_batch_size (`bool`, defaults to `False`):
            Whether the Neuron compiled model supports dynamic batch size.
        output_hidden_states (`bool`, defaults to `False`):
            Whether or not for the traced text encoders to return the hidden states of all layers.
        lora_model_ids (`Optional[List[str]]`, defaults to `None`):
            List of model ids (eg. `ostris/super-cereal-sdxl-lora`) of pretrained lora models hosted on the Hub or paths to local directories containing the lora weights.
        lora_weight_names (`Optional[List[str]]`, defaults to `None`):
            List of lora weights file names.
        lora_adapter_names (`Optional[List[str]]`, defaults to `None`):
            List of adapter names to be used for referencing the loaded adapter models.
        lora_scales (`Optional[List[float]]`, defaults to `None`):
            List of scaling factors for lora adapters.
        controlnet_ids (`Optional[Union[str, List[str]]]`, defaults to `None`):
            Model ID of one or multiple ControlNets providing additional conditioning to the `unet` during the denoising process. If you set multiple
            ControlNets as a list, the outputs from each ControlNet are added together to create one combined additional conditioning.
        controlnet_input_shapes (`Optional[Dict[str, int]]`, defaults to `None`):
            Static shapes used for compiling ControlNets.

    Returns:
        `Dict[str, Tuple[Union[`PreTrainedModel`, `ModelMixin`], `NeuronDefaultConfig`]`: A Dict containing the model and
        Neuron configs for the different components of the model.
    """
    models_for_export = get_submodels_for_export_stable_diffusion(
        pipeline=pipeline,
        lora_model_ids=lora_model_ids,
        lora_weight_names=lora_weight_names,
        lora_adapter_names=lora_adapter_names,
        lora_scales=lora_scales,
        controlnet_ids=controlnet_ids,
    )
    library_name = "diffusers"

    # Text encoders
    if DIFFUSION_MODEL_TEXT_ENCODER_NAME in models_for_export:
        text_encoder = models_for_export[DIFFUSION_MODEL_TEXT_ENCODER_NAME]
        text_encoder_config_constructor = TasksManager.get_exporter_config_constructor(
            model=text_encoder,
            exporter="neuron",
            task="feature-extraction",
            library_name=library_name,
        )
        text_encoder_neuron_config = text_encoder_config_constructor(
            text_encoder.config,
            task="feature-extraction",
            dynamic_batch_size=dynamic_batch_size,
            output_hidden_states=output_hidden_states,
            **text_encoder_input_shapes,
        )
        models_for_export[DIFFUSION_MODEL_TEXT_ENCODER_NAME] = (text_encoder, text_encoder_neuron_config)

    if DIFFUSION_MODEL_TEXT_ENCODER_2_NAME in models_for_export:
        text_encoder_2 = models_for_export[DIFFUSION_MODEL_TEXT_ENCODER_2_NAME]
        text_encoder_config_constructor_2 = TasksManager.get_exporter_config_constructor(
            model=text_encoder_2,
            exporter="neuron",
            task="feature-extraction",
            model_type="clip-text-with-projection",
            library_name=library_name,
        )
        text_encoder_neuron_config_2 = text_encoder_config_constructor_2(
            text_encoder_2.config,
            task="feature-extraction",
            dynamic_batch_size=dynamic_batch_size,
            output_hidden_states=output_hidden_states,
            **text_encoder_input_shapes,
        )
        models_for_export[DIFFUSION_MODEL_TEXT_ENCODER_2_NAME] = (text_encoder_2, text_encoder_neuron_config_2)

    # U-NET
    unet = models_for_export[DIFFUSION_MODEL_UNET_NAME]
    unet_neuron_config_constructor = TasksManager.get_exporter_config_constructor(
        model=unet,
        exporter="neuron",
        task="semantic-segmentation",
        model_type="unet",
        library_name=library_name,
    )
    unet_neuron_config = unet_neuron_config_constructor(
        unet.config,
        task="semantic-segmentation",
        dynamic_batch_size=dynamic_batch_size,
        **unet_input_shapes,
    )
    is_stable_diffusion_xl = isinstance(
        pipeline, (StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline, StableDiffusionXLPipeline)
    )
    unet_neuron_config.is_sdxl = is_stable_diffusion_xl

    unet_neuron_config.with_controlnet = True if controlnet_ids else False

    models_for_export[DIFFUSION_MODEL_UNET_NAME] = (unet, unet_neuron_config)

    # VAE Encoder
    vae_encoder = models_for_export[DIFFUSION_MODEL_VAE_ENCODER_NAME]
    vae_encoder_config_constructor = TasksManager.get_exporter_config_constructor(
        model=vae_encoder,
        exporter="neuron",
        task="semantic-segmentation",
        model_type="vae-encoder",
        library_name=library_name,
    )
    vae_encoder_neuron_config = vae_encoder_config_constructor(
        vae_encoder.config,
        task="semantic-segmentation",
        dynamic_batch_size=dynamic_batch_size,
        **vae_encoder_input_shapes,
    )
    models_for_export[DIFFUSION_MODEL_VAE_ENCODER_NAME] = (vae_encoder, vae_encoder_neuron_config)

    # VAE Decoder
    vae_decoder = models_for_export[DIFFUSION_MODEL_VAE_DECODER_NAME]
    vae_decoder_config_constructor = TasksManager.get_exporter_config_constructor(
        model=vae_decoder,
        exporter="neuron",
        task="semantic-segmentation",
        model_type="vae-decoder",
        library_name=library_name,
    )
    vae_decoder_neuron_config = vae_decoder_config_constructor(
        vae_decoder.config,
        task="semantic-segmentation",
        dynamic_batch_size=dynamic_batch_size,
        **vae_decoder_input_shapes,
    )
    models_for_export[DIFFUSION_MODEL_VAE_DECODER_NAME] = (vae_decoder, vae_decoder_neuron_config)

    # ControlNet
    if controlnet_ids:
        if isinstance(controlnet_ids, str):
            controlnet_ids = [controlnet_ids]
        for idx in range(len(controlnet_ids)):
            controlnet_name = DIFFUSION_MODEL_CONTROLNET_NAME + "_" + str(idx)
            controlnet = models_for_export[controlnet_name]
            controlnet_config_constructor = TasksManager.get_exporter_config_constructor(
                model=controlnet,
                exporter="neuron",
                task="semantic-segmentation",
                model_type="controlnet",
                library_name=library_name,
            )
            controlnet_neuron_config = controlnet_config_constructor(
                controlnet.config,
                task="semantic-segmentation",
                dynamic_batch_size=dynamic_batch_size,
                **controlnet_input_shapes,
            )
            models_for_export[controlnet_name] = (
                controlnet,
                controlnet_neuron_config,
            )

    return models_for_export


def _load_lora_weights_to_pipeline(
    pipeline: Union["StableDiffusionPipeline", "StableDiffusionXLPipeline"],
    lora_model_ids: Optional[Union[str, List[str]]] = None,
    weight_names: Optional[Union[str, List[str]]] = None,
    adapter_names: Optional[Union[str, List[str]]] = None,
    lora_scales: Optional[Union[float, List[float]]] = None,
):
    if isinstance(lora_model_ids, str):
        lora_model_ids = [
            lora_model_ids,
        ]
    if isinstance(weight_names, str):
        weight_names = [
            weight_names,
        ]
    if isinstance(adapter_names, str):
        adapter_names = [
            adapter_names,
        ]
    if isinstance(lora_scales, float):
        lora_scales = [
            lora_scales,
        ]
    if lora_model_ids and weight_names:
        if len(lora_model_ids) == 1:
            pipeline.load_lora_weights(lora_model_ids[0], weight_name=weight_names[0])
            # For tracing the lora weights, we need to use PEFT to fuse adapters directly into the model weights. It won't work by passing the lora scale to the Neuron pipeline during the inference.
            pipeline.fuse_lora(lora_scale=lora_scales[0] if lora_scales else 1.0)
        elif len(lora_model_ids) > 1:
            if not len(lora_model_ids) == len(weight_names) == len(adapter_names):
                raise ValueError(
                    f"weight_name and lora_scale are required to fuse more than one lora. You have {len(lora_model_ids)} lora models to fuse, but you have {len(weight_names)} lora weight names and {len(adapter_names)} adapter names."
                )
            for model_id, weight_name, adapter_name in zip(lora_model_ids, weight_names, adapter_names):
                pipeline.load_lora_weights(model_id, weight_name=weight_name, adapter_name=adapter_name)

            if lora_scales:
                pipeline.set_adapters(adapter_names, adapter_weights=lora_scales)
            pipeline.fuse_lora()

    return pipeline


def load_controlnets(controlnet_ids: Optional[Union[str, List[str]]] = None):
    contronets = []
    if controlnet_ids:
        if isinstance(controlnet_ids, str):
            controlnet_ids = [controlnet_ids]
        for model_id in controlnet_ids:
            model = ControlNetModel.from_pretrained(model_id)
            contronets.append(model)
    return contronets


def get_submodels_for_export_stable_diffusion(
    pipeline: Union["StableDiffusionPipeline", "StableDiffusionXLPipeline"],
    output_hidden_states: bool = False,
    lora_model_ids: Optional[Union[str, List[str]]] = None,
    lora_weight_names: Optional[Union[str, List[str]]] = None,
    lora_adapter_names: Optional[Union[str, List[str]]] = None,
    lora_scales: Optional[List[float]] = None,
    controlnet_ids: Optional[Union[str, List[str]]] = None,
) -> Dict[str, Union["PreTrainedModel", "ModelMixin"]]:
    """
    Returns the components of a Stable Diffusion model.
    """
    is_stable_diffusion_xl = isinstance(
        pipeline, (StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline, StableDiffusionXLPipeline)
    )

    # Lora
    pipeline = _load_lora_weights_to_pipeline(
        pipeline=pipeline,
        lora_model_ids=lora_model_ids,
        weight_names=lora_weight_names,
        adapter_names=lora_adapter_names,
        lora_scales=lora_scales,
    )

    models_for_export = []
    if hasattr(pipeline, "text_encoder_2"):
        projection_dim = pipeline.text_encoder_2.config.projection_dim
    else:
        projection_dim = pipeline.text_encoder.config.projection_dim

    # Text encoders
    if pipeline.text_encoder is not None:
        if is_stable_diffusion_xl or output_hidden_states:
            pipeline.text_encoder.config.output_hidden_states = True
        models_for_export.append((DIFFUSION_MODEL_TEXT_ENCODER_NAME, copy.deepcopy(pipeline.text_encoder)))

    text_encoder_2 = getattr(pipeline, "text_encoder_2", None)
    if text_encoder_2 is not None:
        text_encoder_2.config.output_hidden_states = True
        text_encoder_2.text_model.config.output_hidden_states = True
        models_for_export.append((DIFFUSION_MODEL_TEXT_ENCODER_2_NAME, copy.deepcopy(text_encoder_2)))

    # U-NET
    pipeline.unet.config.text_encoder_projection_dim = projection_dim
    # The U-NET time_ids inputs shapes depends on the value of `requires_aesthetics_score`
    # https://github.com/huggingface/diffusers/blob/v0.18.2/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_img2img.py#L571
    pipeline.unet.config.requires_aesthetics_score = getattr(pipeline.config, "requires_aesthetics_score", False)

    # Replace original cross-attention module with custom cross-attention module for better performance
    # For applying optimized attention score, we need to set env variable  `NEURON_FUSE_SOFTMAX=1`
    if os.environ.get("NEURON_FUSE_SOFTMAX") == "1":
        if is_stable_diffusion_xl:
            logger.info("Applying optimized attention score computation for sdxl.")
            Attention.get_attention_scores = get_attention_scores_sdxl
        else:
            logger.info("Applying optimized attention score computation for stable diffusion.")
            Attention.get_attention_scores = get_attention_scores_sd
    else:
        logger.warning(
            "You are not applying optimized attention score computation. If you want better performance, please"
            " set the environment variable with `export NEURON_FUSE_SOFTMAX=1` and recompile the unet model."
        )
    models_for_export.append((DIFFUSION_MODEL_UNET_NAME, copy.deepcopy(pipeline.unet)))

    if pipeline.vae.config.get("force_upcast", None) is True:
        pipeline.vae.to(dtype=torch.float32)

    # VAE Encoder
    vae_encoder = copy.deepcopy(pipeline.vae)
    vae_encoder.forward = lambda sample: {"latent_parameters": vae_encoder.encode(x=sample)["latent_dist"].parameters}
    models_for_export.append((DIFFUSION_MODEL_VAE_ENCODER_NAME, vae_encoder))

    # VAE Decoder
    vae_decoder = copy.deepcopy(pipeline.vae)
    vae_decoder.forward = lambda latent_sample: vae_decoder.decode(z=latent_sample)
    models_for_export.append((DIFFUSION_MODEL_VAE_DECODER_NAME, vae_decoder))

    # ControlNets
    controlnets = load_controlnets(controlnet_ids)
    if controlnets:
        for idx, controlnet in enumerate(controlnets):
            controlnet.config.text_encoder_projection_dim = pipeline.unet.config.text_encoder_projection_dim
            controlnet.config.requires_aesthetics_score = pipeline.unet.config.requires_aesthetics_score
            controlnet.config.time_cond_proj_dim = pipeline.unet.config.time_cond_proj_dim
            models_for_export.append((DIFFUSION_MODEL_CONTROLNET_NAME + "_" + str(idx), controlnet))

    return OrderedDict(models_for_export)


def check_mandatory_input_shapes(neuron_config_constructor, task, input_shapes):
    mandatory_shapes = neuron_config_constructor.func.get_mandatory_axes_for_task(task)
    for name in mandatory_shapes:
        if input_shapes.get(name, None) is None:
            raise AttributeError(
                f"Cannot find the value of `{name}` which is mandatory for exporting the model to the neuron format, please set the value explicitly."
            )
    input_shapes = {axis: input_shapes[axis] for axis in mandatory_shapes}
    return input_shapes


def replace_stable_diffusion_submodels(pipeline, submodels):
    if submodels is not None:
        unet_id = submodels.pop("unet", None)
        if unet_id is not None:
            unet = UNet2DConditionModel.from_pretrained(unet_id)
            pipeline.unet = unet

    return pipeline


def get_encoder_decoder_models_for_export(
    model: "PreTrainedModel",
    task: str,
    tensor_parallel_size: int,
    input_shapes: Dict[str, int],
    dynamic_batch_size: Optional[bool] = False,
    output_attentions: bool = False,
    output_hidden_states: bool = False,
    model_name_or_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Tuple["PreTrainedModel", "NeuronDefaultConfig"]]:
    """
    Returns the components of an encoder-decoder model and their subsequent neuron configs.
    The encoder includes the compute of encoder hidden states and the initialization of KV
    cache. The decoder the autoprogressive process of generating tokens, which takes past
    key values as inputs to save the compute.

    Args:
        model ("PreTrainedModel"):
            The model to export.
        task (`str`):
            The task to export the model for. If not specified, the task will be auto-inferred based on the model.
        tensor_parallel_size (`int`):
            Tensor parallelism size, the number of Neuron cores on which to shard the model.
        input_shapes (`Dict[str, int]`):
            Static shapes used for compiling the encoder and the decoder.
        dynamic_batch_size (`bool`, defaults to `False`):
            Whether the Neuron compiled model supports dynamic batch size.
        output_attentions (`bool`, defaults to `False`):
            Whether or not for the traced model to return the attentions tensors of all attention layers.
        output_hidden_states (`bool`, defaults to `False`):
            Whether or not for the traced model to return the hidden states of all layers.
        model_name_or_path (`Optional[Union[str, Path]]`, defaults to `None`):
            The location from where the model is loaded, this is needed in the case of tensor parallelism, since we need to load the model within the tracing API.

    Returns:
        `Dict[str, Tuple["PreTrainedModel", "NeuronDefaultConfig"]]`: A Dict containing the model and
        Neuron configs for the different components of the model.
    """
    models_for_export = {}

    # Encoder
    model_type = getattr(model.config, "model_type") + "-encoder"
    encoder_config_constructor = TasksManager.get_exporter_config_constructor(
        exporter="neuron",
        model_type=model_type,
        task=task,
        library_name="transformers",
    )
    check_mandatory_input_shapes(encoder_config_constructor, task, input_shapes)
    encoder_neuron_config = encoder_config_constructor(
        config=model.config,
        task=task,
        dynamic_batch_size=dynamic_batch_size,
        tensor_parallel_size=tensor_parallel_size,
        **input_shapes,
    )
    if not tensor_parallel_size > 1:
        models_for_export[ENCODER_NAME] = (model, encoder_neuron_config)
    else:
        if model_name_or_path:
            models_for_export[ENCODER_NAME] = (model_name_or_path, encoder_neuron_config)
        else:
            raise ValueError(
                f"you need to precise `model_name_or_path` when the parallelism is on, but now it's {model_name_or_path}."
            )

    # Decoder
    model_type = getattr(model.config, "model_type") + "-decoder"
    decoder_config_constructor = TasksManager.get_exporter_config_constructor(
        exporter="neuron",
        model_type=model_type,
        task=task,
        library_name="transformers",
    )
    check_mandatory_input_shapes(encoder_config_constructor, task, input_shapes)
    decoder_neuron_config = decoder_config_constructor(
        config=model.config,
        task=task,
        dynamic_batch_size=dynamic_batch_size,
        tensor_parallel_size=tensor_parallel_size,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        **input_shapes,
    )
    if not tensor_parallel_size > 1:
        models_for_export[DECODER_NAME] = (model, decoder_neuron_config)
    else:
        if model_name_or_path:
            models_for_export[DECODER_NAME] = (model_name_or_path, decoder_neuron_config)
        else:
            raise ValueError(
                f"you need to precise `model_name_or_path` when the parallelism is on, but now it's {model_name_or_path}."
            )

    return models_for_export
