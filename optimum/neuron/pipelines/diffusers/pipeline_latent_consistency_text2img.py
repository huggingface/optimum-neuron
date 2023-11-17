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
"""Override some diffusers API for NeuronLatentConsistencyModelPipeline"""

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import torch
from diffusers import LatentConsistencyModelPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

from .pipeline_utils import StableDiffusionPipelineMixin


if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


class NeuronLatentConsistencyPipelineMixin(StableDiffusionPipelineMixin, LatentConsistencyModelPipeline):
    # Adapted from https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/latent_consistency_models/pipeline_latent_consistency_text2img.py#L470
    def check_inputs(
        self,
        prompt: Union[str, List[str]],
        height: int,
        width: int,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        callback_on_step_end_tensor_inputs: List[str] = None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not set(callback_on_step_end_tensor_inputs) <= set(
            self._callback_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

    # Adapted from https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/latent_consistency_models/pipeline_latent_consistency_text2img.py#L525
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 50,
        original_inference_steps: Optional[int] = None,
        guidance_scale: float = 8.5,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`Optional[Union[str, List[str]]]`, defaults to `None`):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            num_inference_steps (`int`, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            original_inference_steps (`Optional[int]`, defaults to `None`):
                The original number of inference steps use to generate a linearly-spaced timestep schedule, from which
                we will draw `num_inference_steps` evenly spaced timesteps from as our final timestep schedule,
                following the Skipping-Step method in the paper (see Section 4.3). If not set this will default to the
                scheduler's `original_inference_steps` attribute.
            guidance_scale (`float`, defaults to 8.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
                Note that the original latent consistency models paper uses a different CFG formulation where the
                guidance scales are decreased by 1 (so in the paper formulation CFG is enabled when `guidance_scale >
                0`).
            num_images_per_prompt (`int`, defaults to 1):
                The number of images to generate per prompt.
            generator (`Optional[Union[torch.Generator, List[torch.Generator]]]`, defaults to `None`):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`Optional[torch.FloatTensor]`, defaults to `None`):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`Optional[torch.FloatTensor]`, defaults to `None`):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            output_type (`str`, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`Optional[Dict[str, Any]]`, defaults to `None`):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            clip_skip (`Optional[int]`, defaults to `None`):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Optional[Callable]`, defaults to `None`):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List[str]`, defaults to `["latents"]`):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeine class.

        Examples:

        Returns:
            [`diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        # -1. Check `num_images_per_prompt`
        if self.num_images_per_prompt != num_images_per_prompt and not self.dynamic_batch_size:
            logger.warning(
                f"Overriding `num_images_per_prompt({num_images_per_prompt})` to {self.num_images_per_prompt} used for the compilation. Please recompile the models with your "
                f"custom `num_images_per_prompt` or turn on `dynamic_batch_size`, if you wish generating {num_images_per_prompt} per prompt."
            )
            num_images_per_prompt = self.num_images_per_prompt

        # 0. Height and width to unet (static shapes)
        height = self.unet.config.neuron["static_height"] * self.vae_scale_factor
        width = self.unet.config.neuron["static_width"] * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, prompt_embeds, callback_on_step_end_tensor_inputs)
        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        neuron_batch_size = self.unet.config.neuron["static_batch_size"]
        self.check_num_images_per_prompt(batch_size, neuron_batch_size, num_images_per_prompt)

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        # NOTE: when a LCM is distilled from an LDM via latent consistency distillation (Algorithm 1) with guided
        # distillation, the forward pass of the LCM learns to approximate sampling from the LDM using CFG with the
        # unconditional prompt "" (the empty string). Due to this, LCMs currently do not support negative prompts.
        prompt_embeds, _ = self.encode_prompt(
            prompt,
            num_images_per_prompt,
            False,  # do_classifier_free_guidance set to False
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=None,
            lora_scale=lora_scale,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, original_inference_steps=original_inference_steps)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variable
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            generator,
            latents,
        )
        bs = batch_size * num_images_per_prompt

        # 6. Get Guidance Scale Embedding
        # NOTE: We use the Imagen CFG formulation that StableDiffusionPipeline uses rather than the original LCM paper
        # CFG formulation, so we need to subtract 1 from the input guidance_scale.
        # LCM CFG formulation:  cfg_noise = noise_cond + cfg_scale * (noise_cond - noise_uncond), (cfg_scale > 0.0 using CFG)
        w = torch.tensor(self.guidance_scale - 1).repeat(bs)
        w_embedding = self.get_guidance_scale_embedding(w, embedding_dim=self.unet.config.time_cond_proj_dim).to(
            dtype=latents.dtype
        )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, None)

        # 8. LCM MultiStep Sampling Loop:
        self._num_timesteps = len(timesteps)
        num_warmup_steps = self._num_timesteps - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latents = latents.to(prompt_embeds.dtype)

                # model prediction (v-prediction, eps, x)
                model_pred = self.unet(
                    sample=latents,
                    timestep=t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=w_embedding,
                )[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents, denoised = self.scheduler.step(model_pred, t, latents, **extra_step_kwargs, return_dict=False)
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    w_embedding = callback_outputs.pop("w_embedding", w_embedding)
                    denoised = callback_outputs.pop("denoised", denoised)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        denoised = denoised.to(prompt_embeds.dtype)
        if not output_type == "latent":
            image = self.vae_decoder(denoised / getattr(self.vae_decoder.config, "scaling_factor", 0.18215))[0]
            image, has_nsfw_concept = self.run_safety_checker(image, prompt_embeds.dtype)
        else:
            image = denoised
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
