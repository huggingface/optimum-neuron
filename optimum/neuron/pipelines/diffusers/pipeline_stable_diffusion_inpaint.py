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
"""Override some diffusers API for NeuroStableDiffusionInpaintPipeline"""

import logging
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL
import torch
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import deprecate
from diffusers.utils.torch_utils import randn_tensor
from diffusers.image_processor import PipelineImageInput
from .pipeline_utils import DiffusionPipelineMixin


logger = logging.getLogger(__name__)


class StableDiffusionInpaintPipelineMixin(StableDiffusionInpaintPipeline, DiffusionPipelineMixin):
    run_safety_checker = DiffusionPipelineMixin.run_safety_checker
    
    # Adapted from https://github.com/huggingface/diffusers/blob/v0.21.2/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py#L329
    def encode_prompt():
        pass
    
    # Adapted from https://github.com/huggingface/diffusers/blob/v0.21.2/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py#L572
    def prepare_latents():
        pass

    # Adapted from https://github.com/huggingface/diffusers/blob/v0.21.2/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py#L699
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        masked_image_latents: torch.FloatTensor = None,
        strength: float = 1.0,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: int = None,
    ):
        # 0. Height and width to unet (static shapes)
        height = self.unet.config.neuron["static_height"] * self.vae_scale_factor
        width = self.unet.config.neuron["static_width"] * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        if self.num_images_per_prompt != num_images_per_prompt and not self.dynamic_batch_size:
            logger.warning(
                f"Overriding `num_images_per_prompt({num_images_per_prompt})` to {self.num_images_per_prompt} used for the compilation. Please recompile the models with your "
                f"custom `num_images_per_prompt` or turn on `dynamic_batch_size`, if you wish generating {num_images_per_prompt} per prompt."
            )
            num_images_per_prompt = self.num_images_per_prompt
        
        # 1. Check inputs
        self.check_inputs(
            prompt,
            height,
            width,
            strength,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )
        
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]