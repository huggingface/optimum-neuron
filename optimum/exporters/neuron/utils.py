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
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from ...neuron.utils import (
    DECODER_NAME,
    DIFFUSION_MODEL_CONTROLNET_NAME,
    DIFFUSION_MODEL_TEXT_ENCODER_2_NAME,
    DIFFUSION_MODEL_TEXT_ENCODER_NAME,
    DIFFUSION_MODEL_TRANSFORMER_NAME,
    DIFFUSION_MODEL_UNET_NAME,
    DIFFUSION_MODEL_VAE_DECODER_NAME,
    DIFFUSION_MODEL_VAE_ENCODER_NAME,
    ENCODER_NAME,
    InputShapesArguments,
    LoRAAdapterArguments,
    get_attention_scores_sd,
    get_attention_scores_sdxl,
    neuron_scaled_dot_product_attention,
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
        DiffusionPipeline,
        FluxImg2ImgPipeline,
        FluxInpaintPipeline,
        FluxKontextPipeline,
        FluxPipeline,
        ModelMixin,
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLInpaintPipeline,
        StableDiffusionXLPipeline,
        UNet2DConditionModel,
    )
    from diffusers.models import ImageProjection
    from diffusers.models.attention_processor import Attention


if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel

    from .base import NeuronDefaultConfig


def build_stable_diffusion_components_mandatory_shapes(
    batch_size: int | None = None,
    sequence_length: int | None = None,
    unet_or_transformer_num_channels: int | None = None,
    vae_encoder_num_channels: int | None = None,
    vae_decoder_num_channels: int | None = None,
    height: int | None = None,
    width: int | None = None,
    num_images_per_prompt: int | None = 1,
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
    unet_or_transformer_input_shapes = {
        "batch_size": batch_size * num_images_per_prompt,
        "sequence_length": sequence_length,
        "num_channels": unet_or_transformer_num_channels,
        "height": height,
        "width": width,
    }

    components_shapes = {
        "text_encoder": text_encoder_input_shapes,
        "unet_or_transformer": unet_or_transformer_input_shapes,
        "vae_encoder": vae_encoder_input_shapes,
        "vae_decoder": vae_decoder_input_shapes,
    }

    return components_shapes


def get_diffusion_models_for_export(
    pipeline: "DiffusionPipeline",
    tensor_parallel_size: int,
    text_encoder_input_shapes: dict[str, Any],
    unet_input_shapes: dict[str, Any],
    transformer_input_shapes: dict[str, Any],
    vae_encoder_input_shapes: dict[str, Any],
    vae_decoder_input_shapes: dict[str, Any],
    lora_args: LoRAAdapterArguments,
    dynamic_batch_size: bool | None = False,
    output_hidden_states: bool = False,
    controlnet_ids: str | list[str] | None = None,
    controlnet_input_shapes: dict[str, Any] | None = None,
    image_encoder_input_shapes: dict[str, Any] | None = None,
    text_encoder_2_input_shapes: dict[str, Any] | None = None,
    model_name_or_path: str | Path | None = None,
) -> dict[str, tuple["PreTrainedModel | ModelMixin", "NeuronDefaultConfig"]]:
    """
    Returns the components of a Stable Diffusion / Diffusion Transformer(eg. Pixart) / Flux model and their subsequent neuron configs.
    These components are chosen because they represent the bulk of the compute in the pipeline,
    and performance benchmarking has shown that running them on Neuron yields significant
    performance benefit (CLIP text encoder, VAE encoder, VAE decoder, Unet).

    Args:
        pipeline (`"DiffusionPipeline"`):
            The model to export.
        tensor_parallel_size (`int`):
            Tensor parallelism size, the number of Neuron cores on which to shard the model.
        text_encoder_input_shapes (`dict[str, Any]`):
            Static shapes used for compiling text encoder.
        unet_input_shapes (`dict[str, Any]`):
            Static shapes used for compiling unet.
        transformer_input_shapes (`dict[str, Any]`):
            Static shapes used for compiling diffusion transformer.
        vae_encoder_input_shapes (`dict[str, Any]`):
            Static shapes used for compiling vae encoder.
        vae_decoder_input_shapes (`dict[str, Any]`):
            Static shapes used for compiling vae decoder.
        lora_args (`LoRAAdapterArguments`):
            Arguments for fetching the lora adapters, including `model_ids`, `weight_names`, `adapter_names` and `scales`.
        dynamic_batch_size (`bool`, defaults to `False`):
            Whether the Neuron compiled model supports dynamic batch size.
        output_hidden_states (`bool`, defaults to `False`):
            Whether or not for the traced text encoders to return the hidden states of all layers.
        controlnet_ids (`str | list[str] | None`, defaults to `None`):
            Model ID of one or multiple ControlNets providing additional conditioning to the `unet` during the denoising process. If you set multiple
            ControlNets as a list, the outputs from each ControlNet are added together to create one combined additional conditioning.
        controlnet_input_shapes (`dict[str, Any] | None`, defaults to `None`):
            Static shapes used for compiling ControlNets.
        image_encoder_input_shapes (`dict[str, Any] | None`, defaults to `None`):
            Static shapes used for compiling the image encoder.
        text_encoder_2_input_shapes (`dict[str, Any] | None`, defaults to `None`):
            Static shapes used for compiling text encoder 2.
        model_name_or_path (`str | Path | None`, defaults to `None`):
            Path to pretrained model or model identifier from the Hugging Face Hub.

    Returns:
        `dict[str, tuple[`PreTrainedModel` | `ModelMixin`, `NeuronDefaultConfig`]`: A dict containing the model and
        Neuron configs for the different components of the model.
    """
    models_for_export = get_submodels_for_export_diffusion(
        pipeline=pipeline,
        lora_args=lora_args,
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
            input_shapes=text_encoder_input_shapes,
        )
        models_for_export[DIFFUSION_MODEL_TEXT_ENCODER_NAME] = (text_encoder, text_encoder_neuron_config)

    if DIFFUSION_MODEL_TEXT_ENCODER_2_NAME in models_for_export:
        text_encoder_2 = models_for_export[DIFFUSION_MODEL_TEXT_ENCODER_2_NAME]
        text_encoder_config_constructor_2 = TasksManager.get_exporter_config_constructor(
            model=text_encoder_2,
            exporter="neuron",
            task="feature-extraction",
            library_name=library_name,
        )
        text_encoder_neuron_config_2 = text_encoder_config_constructor_2(
            text_encoder_2.config,
            task="feature-extraction",
            tensor_parallel_size=tensor_parallel_size,
            dynamic_batch_size=dynamic_batch_size,
            output_hidden_states=output_hidden_states,
            input_shapes=text_encoder_2_input_shapes,
        )
        if not tensor_parallel_size > 1:
            models_for_export[DIFFUSION_MODEL_TEXT_ENCODER_2_NAME] = (text_encoder_2, text_encoder_neuron_config_2)
        else:
            if model_name_or_path:
                models_for_export[DIFFUSION_MODEL_TEXT_ENCODER_2_NAME] = (
                    model_name_or_path,
                    text_encoder_neuron_config_2,
                )
            else:
                raise ValueError(
                    f"you need to precise `model_name_or_path` when the parallelism is on, but now it's {model_name_or_path}."
                )

    # U-NET
    if DIFFUSION_MODEL_UNET_NAME in models_for_export:
        unet = models_for_export[DIFFUSION_MODEL_UNET_NAME]
        unet_neuron_config_constructor = TasksManager.get_exporter_config_constructor(
            model=unet,
            exporter="neuron",
            task="semantic-segmentation",
            library_name=library_name,
        )
        unet_neuron_config = unet_neuron_config_constructor(
            unet.config,
            task="semantic-segmentation",
            dynamic_batch_size=dynamic_batch_size,
            float_dtype=unet.dtype,
            input_shapes=unet_input_shapes,
        )
        is_stable_diffusion_xl = isinstance(
            pipeline, (StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline, StableDiffusionXLPipeline)
        )
        unet_neuron_config.is_sdxl = is_stable_diffusion_xl
        unet_neuron_config.with_controlnet = True if controlnet_ids else False
        unet_neuron_config.with_ip_adapter = getattr(unet.config, "encoder_hid_dim_type", None) == "ip_image_proj"

        models_for_export[DIFFUSION_MODEL_UNET_NAME] = (unet, unet_neuron_config)

    # Diffusion Transformer
    transformer = None
    if DIFFUSION_MODEL_TRANSFORMER_NAME in models_for_export:
        transformer = models_for_export[DIFFUSION_MODEL_TRANSFORMER_NAME]
        transformer_neuron_config_constructor = TasksManager.get_exporter_config_constructor(
            model=transformer,
            exporter="neuron",
            task="semantic-segmentation",
            library_name=library_name,
        )
        transformer_neuron_config = transformer_neuron_config_constructor(
            transformer.config,
            task="semantic-segmentation",
            tensor_parallel_size=tensor_parallel_size,
            dynamic_batch_size=dynamic_batch_size,
            float_dtype=transformer.dtype,
            input_shapes=transformer_input_shapes,
        )
        if not tensor_parallel_size > 1:
            models_for_export[DIFFUSION_MODEL_TRANSFORMER_NAME] = (transformer, transformer_neuron_config)
        else:
            if model_name_or_path:
                transformer_neuron_config.pretrained_model_name_or_path = model_name_or_path
                models_for_export[DIFFUSION_MODEL_TRANSFORMER_NAME] = (model_name_or_path, transformer_neuron_config)

            else:
                raise ValueError(
                    f"you need to precise `model_name_or_path` when the parallelism is on, but now it's {model_name_or_path}."
                )

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
        float_dtype=vae_encoder.dtype,
        input_shapes=vae_encoder_input_shapes,
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
        float_dtype=transformer.dtype if transformer else vae_decoder.dtype,
        input_shapes=vae_decoder_input_shapes,
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
                float_dtype=controlnet.dtype,
                input_shapes=controlnet_input_shapes,
            )
            models_for_export[controlnet_name] = (
                controlnet,
                controlnet_neuron_config,
            )

    # IP-Adapter: need to compile the image encoder
    if "image_encoder" in models_for_export:
        image_encoder = models_for_export["image_encoder"]
        output_hidden_states = not isinstance(unet.encoder_hid_proj.image_projection_layers[0], ImageProjection)
        image_encoder_config_constructor = TasksManager.get_exporter_config_constructor(
            model=image_encoder,
            exporter="neuron",
            task="feature-extraction",
            model_type="clip-vision-with-projection",
            library_name=library_name,
        )
        image_encoder_neuron_config = image_encoder_config_constructor(
            image_encoder.config,
            task="feature-extraction",
            dynamic_batch_size=dynamic_batch_size,
            output_hidden_states=output_hidden_states,
            input_shapes=image_encoder_input_shapes,
        )
        models_for_export["image_encoder"] = (image_encoder, image_encoder_neuron_config)
        models_for_export[DIFFUSION_MODEL_UNET_NAME][1].image_encoder_output_hidden_states = output_hidden_states

    return models_for_export


def _load_lora_weights_to_pipeline(pipeline: "DiffusionPipeline", lora_args: LoRAAdapterArguments | None):
    if lora_args is None:
        lora_args = LoRAAdapterArguments()
    if lora_args.model_ids and lora_args.weight_names:
        if len(lora_args.model_ids) == 1:
            pipeline.load_lora_weights(lora_args.model_ids[0], weight_name=lora_args.weight_names[0])
            # For tracing the lora weights, we need to use PEFT to fuse adapters directly into the model weights. It won't work by passing the lora scale to the Neuron pipeline during the inference.
            pipeline.fuse_lora(lora_scale=lora_args.scales[0] if lora_args.scales else 1.0)
        elif len(lora_args.model_ids) > 1:
            if not len(lora_args.model_ids) == len(lora_args.weight_names) == len(lora_args.adapter_names):
                raise ValueError(
                    f"weight_name and lora_scale are required to fuse more than one lora. You have {len(lora_args.model_ids)} lora models to fuse, but you have {len(lora_args.weight_names)} lora weight names and {len(lora_args.adapter_names)} adapter names."
                )
            for model_id, weight_name, adapter_name in zip(
                lora_args.model_ids, lora_args.weight_names, lora_args.adapter_names
            ):
                pipeline.load_lora_weights(model_id, weight_name=weight_name, adapter_name=adapter_name)

            if lora_args.scales:
                pipeline.set_adapters(lora_args.adapter_names, adapter_weights=lora_args.scales)
            pipeline.fuse_lora()

    return pipeline


def load_controlnets(controlnet_ids: str | list[str] | None = None):
    contronets = []
    if controlnet_ids:
        if isinstance(controlnet_ids, str):
            controlnet_ids = [controlnet_ids]
        for model_id in controlnet_ids:
            model = ControlNetModel.from_pretrained(model_id)
            contronets.append(model)
    return contronets


def get_submodels_for_export_diffusion(
    pipeline: "DiffusionPipeline",
    lora_args: LoRAAdapterArguments,
    output_hidden_states: bool = False,
    controlnet_ids: str | list[str] | None = None,
) -> dict[str, "PreTrainedModel | ModelMixin"]:
    """
    Stable Diffusion / Diffusion Transformer(eg. Pixart) / Flux
    Returns the components of a  model.
    """
    is_stable_diffusion_xl = isinstance(
        pipeline, (StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline, StableDiffusionXLPipeline)
    )
    is_flux = isinstance(pipeline, (FluxPipeline, FluxImg2ImgPipeline, FluxInpaintPipeline, FluxKontextPipeline))

    # Lora
    pipeline = _load_lora_weights_to_pipeline(pipeline=pipeline, lora_args=lora_args)

    models_for_export = {}

    # Text encoders
    text_encoder = getattr(pipeline, "text_encoder", None)
    if text_encoder is not None:
        if is_stable_diffusion_xl or output_hidden_states:
            pipeline.text_encoder.config.output_hidden_states = True
        text_encoder.config.export_model_type = _get_diffusers_submodel_type(text_encoder)
        models_for_export[DIFFUSION_MODEL_TEXT_ENCODER_NAME] = text_encoder

    text_encoder_2 = getattr(pipeline, "text_encoder_2", None)
    if text_encoder_2 is not None:
        if text_encoder_2.config.model_type == "clip_text_model":
            text_encoder_2.config.output_hidden_states = True
            text_encoder_2.text_model.config.output_hidden_states = True
        text_encoder_2.config.export_model_type = _get_diffusers_submodel_type(text_encoder_2)
        models_for_export[DIFFUSION_MODEL_TEXT_ENCODER_2_NAME] = text_encoder_2
        projection_dim = getattr(pipeline.text_encoder_2.config, "projection_dim", None)
    else:
        projection_dim = getattr(pipeline.text_encoder.config, "projection_dim", None)

    # U-NET
    unet = getattr(pipeline, "unet", None)
    if unet is not None:
        # The U-NET time_ids inputs shapes depends on the value of `requires_aesthetics_score`
        # https://github.com/huggingface/diffusers/blob/v0.18.2/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_img2img.py#L571
        unet.config.requires_aesthetics_score = getattr(pipeline.config, "requires_aesthetics_score", False)
        unet.config.text_encoder_projection_dim = projection_dim

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
        unet.config.export_model_type = _get_diffusers_submodel_type(unet)
        models_for_export[DIFFUSION_MODEL_UNET_NAME] = unet

    # Diffusion transformer
    transformer = getattr(pipeline, "transformer", None)
    if transformer is not None:
        if not is_flux:  # The following will be handled by `ModelBuilder` if `is_flux`.
            transformer.config.requires_aesthetics_score = getattr(pipeline.config, "requires_aesthetics_score", False)
            transformer.config.text_encoder_projection_dim = projection_dim
            # apply optimized scaled_dot_product_attention
            sdpa_original = torch.nn.functional.scaled_dot_product_attention

            def attention_wrapper(query, key, value, attn_mask=None, dropout_p=None, is_causal=None):
                if attn_mask is not None:
                    return sdpa_original(
                        query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
                    )
                else:
                    return neuron_scaled_dot_product_attention(
                        query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
                    )

            torch.nn.functional.scaled_dot_product_attention = attention_wrapper
        transformer.config.export_model_type = _get_diffusers_submodel_type(transformer)
        models_for_export[DIFFUSION_MODEL_TRANSFORMER_NAME] = transformer

    if pipeline.vae.config.get("force_upcast", None) is True:
        pipeline.vae.to(dtype=torch.float32)

    # VAE Encoder
    vae_encoder = copy.deepcopy(pipeline.vae)
    vae_encoder.forward = lambda sample: {"latent_parameters": vae_encoder.encode(x=sample)["latent_dist"].parameters}
    models_for_export[DIFFUSION_MODEL_VAE_ENCODER_NAME] = vae_encoder

    # VAE Decoder
    vae_decoder = copy.deepcopy(pipeline.vae)
    unet_or_transformer = unet or transformer
    vae_decoder.forward = lambda latent_sample: vae_decoder.decode(z=latent_sample)
    if vae_decoder.dtype is torch.float32 and unet_or_transformer.dtype is not torch.float32:
        vae_decoder = f32Wrapper(vae_decoder)
    models_for_export[DIFFUSION_MODEL_VAE_DECODER_NAME] = vae_decoder

    # ControlNets
    controlnets = load_controlnets(controlnet_ids)
    if controlnets:
        for idx, controlnet in enumerate(controlnets):
            controlnet.config.text_encoder_projection_dim = pipeline.unet.config.text_encoder_projection_dim
            controlnet.config.requires_aesthetics_score = pipeline.unet.config.requires_aesthetics_score
            controlnet.config.time_cond_proj_dim = pipeline.unet.config.time_cond_proj_dim
            models_for_export[DIFFUSION_MODEL_CONTROLNET_NAME + "_" + str(idx)] = controlnet

    # Image Encoder
    image_encoder = getattr(pipeline, "image_encoder", None)
    if image_encoder is not None:
        models_for_export["image_encoder"] = image_encoder

    return models_for_export


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


class f32Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.original = model

    def forward(self, x):
        y = x.to(torch.float32)
        output = self.original(y)
        return output

    def __getattr__(self, name):
        # Delegate attribute/method lookup to the wrapped model if not found in this wrapper
        if name == "original":
            return super().__getattr__(name)
        return getattr(self.original, name)


_DIFFUSERS_CLASS_NAME_TO_SUBMODEL_TYPE = {
    "CLIPTextModel": "clip-text-model",
    "CLIPTextModelWithProjection": "clip-text-with-projection",
    "FluxTransformer2DModel": "flux-transformer-2d",
    "SD3Transformer2DModel": "sd3-transformer-2d",
    "UNet2DConditionModel": "unet",
    "PixArtTransformer2DModel": "pixart-transformer-2d",
    "T5EncoderModel": "t5",
}


def _get_diffusers_submodel_type(submodel):
    export_model_type = _DIFFUSERS_CLASS_NAME_TO_SUBMODEL_TYPE.get(submodel.__class__.__name__)
    if "t5" in export_model_type:
        export_model_type = "t5-encoder"
    return export_model_type


def get_encoder_decoder_models_for_export(
    model: "PreTrainedModel",
    task: str,
    tensor_parallel_size: int,
    input_shapes: dict[str, int],
    dynamic_batch_size: bool | None = False,
    output_attentions: bool = False,
    output_hidden_states: bool = False,
    model_name_or_path: str | Path | None = None,
    preprocessors: list | None = None,
) -> dict[str, tuple["PreTrainedModel", "NeuronDefaultConfig"]]:
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
        input_shapes (`dict[str, int]`):
            Static shapes used for compiling the encoder and the decoder.
        dynamic_batch_size (`bool`, defaults to `False`):
            Whether the Neuron compiled model supports dynamic batch size.
        output_attentions (`bool`, defaults to `False`):
            Whether or not for the traced model to return the attentions tensors of all attention layers.
        output_hidden_states (`bool`, defaults to `False`):
            Whether or not for the traced model to return the hidden states of all layers.
        model_name_or_path (`str | Path | None`, defaults to `None`):
            The location from where the model is loaded, this is needed in the case of tensor parallelism, since we need to load the model within the tracing API.

    Returns:
        `dict[str, tuple["PreTrainedModel", "NeuronDefaultConfig"]]`: A dict containing the model and
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
    input_shape_args = InputShapesArguments(**input_shapes)
    # Whisper specific
    if getattr(model.config, "model_type", None) == "whisper":
        setattr(model.config, "stride", [model.model.encoder.conv1.stride[0], model.model.encoder.conv2.stride[0]])
    encoder_neuron_config = encoder_config_constructor(
        config=model.config,
        task=task,
        dynamic_batch_size=dynamic_batch_size,
        tensor_parallel_size=tensor_parallel_size,
        input_shapes=input_shape_args,
        preprocessors=preprocessors,
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
    check_mandatory_input_shapes(decoder_config_constructor, task, input_shapes)
    decoder_neuron_config = decoder_config_constructor(
        config=model.config,
        task=task,
        dynamic_batch_size=dynamic_batch_size,
        tensor_parallel_size=tensor_parallel_size,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        input_shapes=input_shape_args,
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
