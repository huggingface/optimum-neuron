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
"""Override some diffusers API for NeuroStableDiffusionImg2ImgPipeline"""

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import PIL
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils import deprecate
from diffusers.utils.torch_utils import randn_tensor

from .pipeline_utils import StableDiffusionPipelineMixin


if TYPE_CHECKING:
    from diffusers.image_processor import PipelineImageInput


logger = logging.getLogger(__name__)


class NeuronStableDiffusionImg2ImgPipelineMixin(StableDiffusionPipelineMixin, StableDiffusionImg2ImgPipeline):
    # Adapted from diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.prepare_latents
    def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, generator=None):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            init_latents = image
        else:
            # [Modified] Replace with pre-compiled vae encoder, encode the init image into latents and scale the latents
            init_latents = self.vae_encoder(sample=image)[0]
            scaling_factor = self.vae_encoder.config.scaling_factor or 0.18215
            init_latents = scaling_factor * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, dtype=dtype)

        # get latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        latents = init_latents

        return latents

    # Adapted from diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.__call__
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        image: Optional["PipelineImageInput"] = None,
        strength: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`Optional[Union[str, List[str]]]`, defaults to `None`):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`Optional["PipelineImageInput"]`, defaults to `None`):
                `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
                numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
                or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
                list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image
                latents as `image`, but if passing latents directly it is not encoded again.
            strength (`float`, defaults to 0.8):
                Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a
                starting point and more noise is added the higher the `strength`. The number of denoising steps depends
                on the amount of noise initially added. When `strength` is 1, added noise is maximum and the denoising
                process runs for the full number of iterations specified in `num_inference_steps`. A value of 1
                essentially ignores `image`.
            num_inference_steps (`int`, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.
            guidance_scale (`float`, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`Optional[Union[str, List[str]`, defaults to `None`):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, defaults to 1):
                The number of images to generate per prompt. If it is different from the batch size used for the compiltaion,
                it will be overriden by the static batch size of neuron (except for dynamic batching).
            eta (`float`, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`diffusers.schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`Optional[Union[torch.Generator, List[torch.Generator]]]`, defaults to `None`):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            prompt_embeds (`Optional[torch.FloatTensor]`, defaults to `None`):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`Optional[torch.FloatTensor]`, defaults to `None`):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`Optional[str]`, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, defaults to `True`):
                Whether or not to return a [`diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Optional[Callable]`, defaults to `None`):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, defaults to `None`):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

        Examples:

        ```py
        >>> from optimum.neuron import NeuronStableDiffusionImg2ImgPipeline
        >>> from diffusers.utils import load_image

        >>> url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
        >>> init_image = load_image(url).convert("RGB")

        >>> compiler_args = {"auto_cast": "matmul", "auto_cast_type": "bf16"}
        >>> input_shapes = {"batch_size": 1, "height": 512, "width": 512}
        >>> pipeline = NeuronStableDiffusionImg2ImgPipeline.from_pretrained(
        ...     "nitrosocke/Ghibli-Diffusion", export=True, **compiler_args, **input_shapes,
        ... )
        >>> pipeline.save_pretrained("sd_img2img/")

        >>> prompt = "ghibli style, a fantasy landscape with snowcapped mountains, trees, lake with detailed reflection."
        >>> image = pipeline(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images[0]
        ```

        Returns:
            [`diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        # 0. Check `num_images_per_prompt`
        if self.num_images_per_prompt != num_images_per_prompt and not self.dynamic_batch_size:
            logger.warning(
                f"Overriding `num_images_per_prompt({num_images_per_prompt})` to {self.num_images_per_prompt} used for the compilation. Please recompile the models with your "
                f"custom `num_images_per_prompt` or turn on `dynamic_batch_size`, if you wish generating {num_images_per_prompt} image per prompt."
            )
            num_images_per_prompt = self.num_images_per_prompt

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, strength, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        neuron_batch_size = self.unet.config.neuron["static_batch_size"]
        self.check_num_images_per_prompt(batch_size, neuron_batch_size, num_images_per_prompt)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0 and (
            self.dynamic_batch_size or self.data_parallel_mode == "unet"
        )

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Preprocess image
        height = self.vae_encoder.config.neuron["static_height"]
        width = self.vae_encoder.config.neuron["static_width"]
        image = self.image_processor.preprocess(image, height=height, width=width)

        # 5. set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device=None)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 6. Prepare latent variables
        latents = self.prepare_latents(
            image, latent_timestep, batch_size, num_images_per_prompt, prompt_embeds.dtype, generator
        )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                # [modified for neuron] Remove not traced inputs: cross_attention_kwargs, return_dict
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if not output_type == "latent":
            # [Modified] Replace with pre-compiled vae decoder
            image = self.vae_decoder(latents / getattr(self.vae_decoder.config, "scaling_factor", 0.18215))[0]
            image, has_nsfw_concept = self.run_safety_checker(image, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
