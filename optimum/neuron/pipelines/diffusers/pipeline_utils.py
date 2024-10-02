#  Copyright 2023 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import logging
from typing import List, Optional, Union

import torch
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.utils.torch_utils import randn_tensor


logger = logging.getLogger(__name__)


class NeuronDiffusionPipelineMixin:
    def check_num_images_per_prompt(self, prompt_batch_size: int, neuron_batch_size: int, num_images_per_prompt: int):
        if (
            not self.data_parallel_mode == "all"
            and not self.dynamic_batch_size
            and neuron_batch_size != prompt_batch_size * num_images_per_prompt
        ):
            raise ValueError(
                f"Models in the pipeline were compiled with `batch_size` {neuron_batch_size} which does not equal the number of"
                f" prompt({prompt_batch_size}) multiplied by `num_images_per_prompt`({num_images_per_prompt}). You need to enable"
                " `dynamic_batch_size` or precisely configure `num_images_per_prompt` during the compilation."
            )
    
    # Adapted from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(
        self, 
        batch_size, 
        num_channels_latents, 
        height, 
        width, 
        dtype, 
        device,
        generator, 
        latents=None
    ):
        import pdb
        pdb.set_trace()
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, dtype=dtype)
        elif latents.shape != shape:
            raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
