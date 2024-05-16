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
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch

from ...neuron.utils import (
    DECODER_NAME,
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
    from diffusers import UNet2DConditionModel
    from diffusers.models.attention_processor import (
        Attention,
        AttnAddedKVProcessor,
        AttnAddedKVProcessor2_0,
        AttnProcessor,
        AttnProcessor2_0,
        LoRAAttnProcessor,
        LoRAAttnProcessor2_0,
    )


if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel

    from .base import NeuronDefaultConfig

    if is_diffusers_available():
        from diffusers import ModelMixin, StableDiffusionPipeline, StableDiffusionXLImg2ImgPipeline


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
    pipeline: Union["StableDiffusionPipeline", "StableDiffusionXLImg2ImgPipeline"],
    task: str,
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
) -> Dict[str, Tuple[Union["PreTrainedModel", "ModelMixin"], "NeuronDefaultConfig"]]:
    """
    Returns the components of a Stable Diffusion model and their subsequent neuron configs.
    These components are chosen because they represent the bulk of the compute in the pipeline,
    and performance benchmarking has shown that running them on Neuron yields significant
    performance benefit (CLIP text encoder, VAE encoder, VAE decoder, Unet).

    Args:
        pipeline ([`Union["StableDiffusionPipeline", "StableDiffusionXLImg2ImgPipeline"]`]):
            The model to export.
        task (`str`):
            Task name, should be either "stable-diffusion" or "stable-diffusion-xl".
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

    Returns:
        `Dict[str, Tuple[Union[`PreTrainedModel`, `ModelMixin`], `NeuronDefaultConfig`]`: A Dict containing the model and
        Neuron configs for the different components of the model.
    """
    models_for_export = get_submodels_for_export_stable_diffusion(
        pipeline=pipeline,
        task=task,
        lora_model_ids=lora_model_ids,
        lora_weight_names=lora_weight_names,
        lora_adapter_names=lora_adapter_names,
        lora_scales=lora_scales,
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
    if task == "stable-diffusion-xl":
        unet_neuron_config.is_sdxl = True
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

    return models_for_export


def _load_lora_weights_to_pipeline(
    pipeline: Union["StableDiffusionPipeline", "StableDiffusionXLImg2ImgPipeline"],
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


def get_submodels_for_export_stable_diffusion(
    pipeline: Union["StableDiffusionPipeline", "StableDiffusionXLImg2ImgPipeline"],
    task: str,
    output_hidden_states: bool = False,
    lora_model_ids: Optional[Union[str, List[str]]] = None,
    lora_weight_names: Optional[Union[str, List[str]]] = None,
    lora_adapter_names: Optional[Union[str, List[str]]] = None,
    lora_scales: Optional[List[float]] = None,
) -> Dict[str, Union["PreTrainedModel", "ModelMixin"]]:
    """
    Returns the components of a Stable Diffusion model.
    """
    is_sdxl = "xl" in task

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
        if is_sdxl or output_hidden_states:
            pipeline.text_encoder.config.output_hidden_states = True
        models_for_export.append((DIFFUSION_MODEL_TEXT_ENCODER_NAME, copy.deepcopy(pipeline.text_encoder)))

    text_encoder_2 = getattr(pipeline, "text_encoder_2", None)
    if text_encoder_2 is not None:
        text_encoder_2.config.output_hidden_states = True
        models_for_export.append((DIFFUSION_MODEL_TEXT_ENCODER_2_NAME, copy.deepcopy(text_encoder_2)))

    # U-NET
    pipeline.unet.set_attn_processor(AttnProcessor())
    pipeline.unet.config.text_encoder_projection_dim = projection_dim
    # The U-NET time_ids inputs shapes depends on the value of `requires_aesthetics_score`
    # https://github.com/huggingface/diffusers/blob/v0.18.2/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_img2img.py#L571
    pipeline.unet.config.requires_aesthetics_score = getattr(pipeline.config, "requires_aesthetics_score", False)

    # Replace original cross-attention module with custom cross-attention module for better performance
    # For applying optimized attention score, we need to set env variable  `NEURON_FUSE_SOFTMAX=1`
    if os.environ.get("NEURON_FUSE_SOFTMAX") == "1":
        logger.info("Applying optimized attention score computation.")
        Attention.get_attention_scores = get_attention_scores_sdxl if is_sdxl else get_attention_scores_sd
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
    vae_encoder.forward = lambda sample: {"latent_sample": vae_encoder.encode(x=sample)["latent_dist"].sample()}
    models_for_export.append((DIFFUSION_MODEL_VAE_ENCODER_NAME, vae_encoder))

    # VAE Decoder
    vae_decoder = copy.deepcopy(pipeline.vae)
    vae_decoder.forward = lambda latent_sample: vae_decoder.decode(z=latent_sample)
    models_for_export.append((DIFFUSION_MODEL_VAE_DECODER_NAME, vae_decoder))

    return OrderedDict(models_for_export)


# Using xformers or torch_2_0 can avoid overflow on float16, do not apply this unless compilation error.
def override_diffusers_2_0_attn_processors(model):
    for _, submodule in model.named_modules():
        if isinstance(submodule, Attention):
            if isinstance(submodule.processor, AttnProcessor2_0):
                submodule.set_processor(AttnProcessor())
            elif isinstance(submodule.processor, LoRAAttnProcessor2_0):
                lora_attn_processor = LoRAAttnProcessor(
                    hidden_size=submodule.processor.hidden_size,
                    cross_attention_dim=submodule.processor.cross_attention_dim,
                    rank=submodule.processor.rank,
                    network_alpha=submodule.processor.to_q_lora.network_alpha,
                )
                lora_attn_processor.to_q_lora = copy.deepcopy(submodule.processor.to_q_lora)
                lora_attn_processor.to_k_lora = copy.deepcopy(submodule.processor.to_k_lora)
                lora_attn_processor.to_v_lora = copy.deepcopy(submodule.processor.to_v_lora)
                lora_attn_processor.to_out_lora = copy.deepcopy(submodule.processor.to_out_lora)
                submodule.set_processor(lora_attn_processor)
            elif isinstance(submodule.processor, AttnAddedKVProcessor2_0):
                submodule.set_processor(AttnAddedKVProcessor())
    return model


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
    input_shapes: Dict[str, int],
    dynamic_batch_size: Optional[bool] = False,
    output_attentions: bool = False,
    output_hidden_states: bool = False,
) -> Dict[str, Tuple["PreTrainedModel", "NeuronDefaultConfig"]]:
    """
    Returns the components of an encoder-decoder model and their subsequent neuron configs.
    The encoder includes the compute of encoder hidden states and the initialization of KV
    cache. The decoder the autoprogressive process of generating tokens, which takes past
    key values as inputs to save the compute.

    Args:
        model ("PreTrainedModel"):
            The model to export.
        input_shapes (`Dict[str, int]`):
            Static shapes used for compiling the encoder and the decoder.
        dynamic_batch_size (`bool`, defaults to `False`):
            Whether the Neuron compiled model supports dynamic batch size.
        output_attentions (`bool`, defaults to `False`):
            Whether or not for the traced model to return the attentions tensors of all attention layers.
        output_hidden_states (`bool`, defaults to `False`):
            Whether or not for the traced model to return the hidden states of all layers.

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
        **input_shapes,
    )
    models_for_export[ENCODER_NAME] = (model, encoder_neuron_config)

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
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        **input_shapes,
    )
    models_for_export[DECODER_NAME] = (model, decoder_neuron_config)

    return models_for_export
