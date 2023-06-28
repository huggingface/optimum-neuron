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
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import packaging
import torch
from ...utils import (
    is_diffusers_available,
    logging,
)
from ..tasks import TasksManager

logger = logging.get_logger()

if TYPE_CHECKING:
    from .base import NeuronConfig

    from transformers.modeling_utils import PreTrainedModel

    if is_diffusers_available():
        from diffusers import ModelMixin, StableDiffusionPipeline
        from diffusers.models.attention_processor import Attention

def get_stable_diffusion_models_for_export(
    pipeline: "StableDiffusionPipeline",
    text_encoder_input_shapes: Dict[str, int],
    vae_decoder_input_shapes: Dict[str, int],
    unet_input_shapes: Dict[str, int],
    vae_post_quant_conv_input_shapes: Dict[str, int],
    dynamic_batch_size: Optional[bool]=False,
) -> Dict[str, Tuple[Union["PreTrainedModel", "ModelMixin"], "NeuronConfig"]]:
    """
    Returns the components of a Stable Diffusion model and their subsequent neuron configs. 
    These components are chosen because they represent the bulk of the compute in the pipeline, 
    and performance benchmarking has shown that running them on Neuron yields significant 
    performance benefit (CLIP text encoder, VAE decoder, Unet, VAE post quant conv). 

    Args:
        pipeline ([`StableDiffusionPipeline`]):
            The model to export.

    Returns:
        `Dict[str, Tuple[Union[`PreTrainedModel`, `ModelMixin`], `NeuronConfig`]: A Dict containing the model and
        Neuron configs for the different components of the model.
    """
    models_for_export = {}

    # Text encoder
    text_encoder = copy.deepcopy(pipeline.text_encoder)
    text_encoder_config_constructor = TasksManager.get_exporter_config_constructor(
        model=text_encoder, exporter="neuron", task="feature-extraction"
    )
    text_encoder_neuron_config = text_encoder_config_constructor(
        text_encoder.config, 
        task="feature-extraction", 
        dynamic_batch_size=dynamic_batch_size,
        **text_encoder_input_shapes,
    )
    models_for_export["text_encoder"] = (text_encoder, text_encoder_neuron_config)

    # VAE Decoder
    vae_config = pipeline.vae.config
    vae_decoder = copy.deepcopy(pipeline.vae.decoder)
    vae_decoder_config_constructor = TasksManager.get_exporter_config_constructor(
        model=vae_decoder, 
        exporter="neuron", 
        task="semantic-segmentation", 
        model_type="vae-decoder",
    )
    vae_decoder_neuron_config = vae_decoder_config_constructor(
        vae_config, 
        task="semantic-segmentation",
        dynamic_batch_size=dynamic_batch_size,
        **vae_decoder_input_shapes,
    )
    models_for_export["vae_decoder"] = (vae_decoder, vae_decoder_neuron_config)

    # U-NET
    unet = copy.deepcopy(pipeline.unet)
    unet_neuron_config_constructor = TasksManager.get_exporter_config_constructor(
        model=unet, exporter="neuron", task="semantic-segmentation", model_type="unet"
    )
    unet_neuron_config = unet_neuron_config_constructor(
        unet.config,
        task="semantic-segmentation", 
        dynamic_batch_size=dynamic_batch_size,
        **unet_input_shapes,
    )
    models_for_export["unet"] = (unet, unet_neuron_config)

    # VAE Decoder
    post_quant_conv = copy.deepcopy(pipeline.vae.post_quant_conv)
    post_quant_conv_neuron_config_constructor = TasksManager.get_exporter_config_constructor(
        model=post_quant_conv, exporter="neuron", task="semantic-segmentation", model_type="conv2d"
    )
    post_quant_conv_neuron_config = post_quant_conv_neuron_config_constructor(
        vae_config,
        task="semantic-segmentation", 
        dynamic_batch_size=dynamic_batch_size,
        **vae_post_quant_conv_input_shapes,
    )
    models_for_export["vae_conv"] = (post_quant_conv, post_quant_conv_neuron_config)

    del pipeline

    return models_for_export