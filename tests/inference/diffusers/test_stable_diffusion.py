# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from io import BytesIO

import cv2
import numpy as np
import PIL
import pytest
import requests
from compel import Compel, ReturnedEmbeddingsType
from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image
from PIL import Image

from optimum.neuron import (
    NeuronLatentConsistencyModelPipeline,
    NeuronStableDiffusionControlNetPipeline,
    NeuronStableDiffusionImg2ImgPipeline,
    NeuronStableDiffusionInpaintPipeline,
    NeuronStableDiffusionInstructPix2PixPipeline,
    NeuronStableDiffusionPipeline,
    NeuronStableDiffusionXLImg2ImgPipeline,
    NeuronStableDiffusionXLInpaintPipeline,
    NeuronStableDiffusionXLPipeline,
)
from optimum.neuron.modeling_diffusion import (
    NeuronControlNetModel,
    NeuronModelTextEncoder,
    NeuronModelUnet,
    NeuronModelVaeDecoder,
    NeuronModelVaeEncoder,
    NeuronMultiControlNetModel,
)
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx
from optimum.utils import logging
from optimum.utils.testing_utils import require_diffusers


logger = logging.get_logger()


def _download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")


def _prepare_canny_image(image_url=None):
    if image_url is None:
        image_url = (
            "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
        )
    original_image = load_image(image_url)
    image = np.array(original_image)
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = PIL.Image.fromarray(image)

    return canny_image


# [STABLE DIFFUSION]
@is_inferentia_test
@requires_neuronx
@require_diffusers
def test_sd_export_and_inference_non_dyn(neuron_stable_diffusion_num_img_per_prompt_4_non_dyn_path):
    neuron_pipeline = NeuronStableDiffusionPipeline.from_pretrained(
        neuron_stable_diffusion_num_img_per_prompt_4_non_dyn_path
    )
    assert isinstance(neuron_pipeline.text_encoder, NeuronModelTextEncoder)
    assert isinstance(neuron_pipeline.unet, NeuronModelUnet)
    assert isinstance(neuron_pipeline.vae_encoder, NeuronModelVaeEncoder)
    assert isinstance(neuron_pipeline.vae_decoder, NeuronModelVaeDecoder)

    prompts = ["sailing ship in storm by Leonardo da Vinci"]
    image = neuron_pipeline(prompts, num_images_per_prompt=4).images[0]
    assert isinstance(image, PIL.Image.Image)


@is_inferentia_test
@requires_neuronx
@require_diffusers
@pytest.mark.skip(reason="Dynamic batching broken.")
def test_sd_export_and_inference_dyn(neuron_stable_diffusion_dyn_path):
    neuron_pipeline = NeuronStableDiffusionPipeline(neuron_stable_diffusion_dyn_path)

    prompts = ["sailing ship in storm by Leonardo da Vinci"] * 2
    image = neuron_pipeline(prompts, num_images_per_prompt=2).images[0]
    assert isinstance(image, PIL.Image.Image)


@is_inferentia_test
@requires_neuronx
@require_diffusers
def test_sd_img2img_export_and_inference(neuron_stable_diffusion_num_img_per_prompt_1_non_dyn_path):
    neuron_pipeline = NeuronStableDiffusionImg2ImgPipeline.from_pretrained(
        neuron_stable_diffusion_num_img_per_prompt_1_non_dyn_path
    )

    url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
    init_image = _download_image(url)
    prompt = "ghibli style, a fantasy landscape with mountain, trees and lake, reflection"
    image = neuron_pipeline(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images[0]
    assert isinstance(image, PIL.Image.Image)


@is_inferentia_test
@requires_neuronx
@require_diffusers
def test_sd_inpaint_export_and_inference(neuron_stable_diffusion_num_img_per_prompt_1_non_dyn_path):
    neuron_pipeline = NeuronStableDiffusionInpaintPipeline.from_pretrained(
        neuron_stable_diffusion_num_img_per_prompt_1_non_dyn_path
    )

    img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
    init_image = _download_image(img_url).resize((512, 512))
    mask_image = _download_image(mask_url).resize((512, 512))
    prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
    image = neuron_pipeline(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
    assert isinstance(image, PIL.Image.Image)


@is_inferentia_test
@requires_neuronx
@require_diffusers
def test_sd_instruct_pix2pix_export_and_inference(neuron_stable_diffusion_ip2p_path):
    neuron_pipeline = NeuronStableDiffusionInstructPix2PixPipeline.from_pretrained(neuron_stable_diffusion_ip2p_path)

    img_url = "https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png"
    init_image = _download_image(img_url).resize((512, 512))
    prompt = "Add a beautiful sunset"
    image = neuron_pipeline(prompt=prompt, image=init_image).images[0]
    assert isinstance(image, PIL.Image.Image)


@is_inferentia_test
@requires_neuronx
@require_diffusers
def test_sd_lcm_export_and_inference(neuron_lcm_path):
    neuron_pipeline = NeuronLatentConsistencyModelPipeline.from_pretrained(neuron_lcm_path)

    prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
    image = neuron_pipeline(prompt, num_inference_steps=4, guidance_scale=8.0).images[0]
    assert isinstance(image, PIL.Image.Image)


@is_inferentia_test
@requires_neuronx
@require_diffusers
def test_sd_export_and_inference_with_fused_lora(neuron_stable_diffusion_with_fused_lora_path):
    neuron_pipeline = NeuronStableDiffusionPipeline.from_pretrained(neuron_stable_diffusion_with_fused_lora_path)
    assert isinstance(neuron_pipeline.text_encoder, NeuronModelTextEncoder)
    assert isinstance(neuron_pipeline.unet, NeuronModelUnet)
    assert isinstance(neuron_pipeline.vae_encoder, NeuronModelVaeEncoder)
    assert isinstance(neuron_pipeline.vae_decoder, NeuronModelVaeDecoder)

    prompts = ["A cute brown bear eating a slice of pizza"]
    image = neuron_pipeline(prompts, num_images_per_prompt=1).images[0]
    assert isinstance(image, PIL.Image.Image)


@is_inferentia_test
@requires_neuronx
@require_diffusers
def test_sd_compatibility_with_compel(neuron_stable_diffusion_with_hidden_states_output_path):
    pipe = NeuronStableDiffusionPipeline.from_pretrained(neuron_stable_diffusion_with_hidden_states_output_path)

    prompt = "a red cat playing with a ball++"
    compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
    prompt_embeds = compel_proc(prompt)
    image = pipe(prompt_embeds=prompt_embeds, num_inference_steps=2).images[0]
    assert isinstance(image, PIL.Image.Image)


@is_inferentia_test
@requires_neuronx
@require_diffusers
def test_sd_export_and_inference_with_single_controlnet(neuron_stable_diffusion_single_controlnet_path):
    neuron_pipeline = NeuronStableDiffusionControlNetPipeline.from_pretrained(
        neuron_stable_diffusion_single_controlnet_path
    )
    assert isinstance(neuron_pipeline.text_encoder, NeuronModelTextEncoder)
    assert isinstance(neuron_pipeline.unet, NeuronModelUnet)
    assert isinstance(neuron_pipeline.vae_encoder, NeuronModelVaeEncoder)
    assert isinstance(neuron_pipeline.vae_decoder, NeuronModelVaeDecoder)
    assert isinstance(neuron_pipeline.controlnet, NeuronControlNetModel)

    prompt = "the mona lisa"
    canny_image = _prepare_canny_image()
    image = neuron_pipeline(prompt, image=canny_image).images[0]
    neuron_pipeline.scheduler = UniPCMultistepScheduler.from_config(neuron_pipeline.scheduler.config)
    assert isinstance(image, PIL.Image.Image)


@is_inferentia_test
@requires_neuronx
@require_diffusers
def test_sd_export_and_inference_with_multiple_controlnet(neuron_stable_diffusion_multiple_controlnets_path):
    neuron_pipeline = NeuronStableDiffusionControlNetPipeline.from_pretrained(
        neuron_stable_diffusion_multiple_controlnets_path
    )
    assert isinstance(neuron_pipeline.text_encoder, NeuronModelTextEncoder)
    assert isinstance(neuron_pipeline.unet, NeuronModelUnet)
    assert isinstance(neuron_pipeline.vae_encoder, NeuronModelVaeEncoder)
    assert isinstance(neuron_pipeline.vae_decoder, NeuronModelVaeDecoder)
    assert isinstance(neuron_pipeline.controlnet, NeuronMultiControlNetModel)

    prompt = "the mona lisa"
    canny_image = _prepare_canny_image()
    image = neuron_pipeline(prompt, image=[canny_image, canny_image]).images[0]
    neuron_pipeline.scheduler = UniPCMultistepScheduler.from_config(neuron_pipeline.scheduler.config)
    assert isinstance(image, PIL.Image.Image)


# [STABLE DIFFUSION XL]
@is_inferentia_test
@requires_neuronx
@require_diffusers
def test_sdxl_export_and_inference_non_dyn(neuron_sdxl_num_img_per_prompt_4_non_dyn_path):
    neuron_pipeline = NeuronStableDiffusionXLPipeline.from_pretrained(neuron_sdxl_num_img_per_prompt_4_non_dyn_path)
    assert isinstance(neuron_pipeline.text_encoder, NeuronModelTextEncoder)
    assert isinstance(neuron_pipeline.text_encoder_2, NeuronModelTextEncoder)
    assert isinstance(neuron_pipeline.unet, NeuronModelUnet)
    assert isinstance(neuron_pipeline.vae_encoder, NeuronModelVaeEncoder)
    assert isinstance(neuron_pipeline.vae_decoder, NeuronModelVaeDecoder)

    prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    prompt_2 = "Van Gogh painting"
    negative_prompt_1 = "low quality, low resolution"
    negative_prompt_2 = "low quality, low resolution"

    image = neuron_pipeline(
        prompt=prompt,
        prompt_2=prompt_2,
        negative_prompt=negative_prompt_1,
        negative_prompt_2=negative_prompt_2,
        num_images_per_prompt=4,
    ).images[0]
    assert isinstance(image, PIL.Image.Image)


@is_inferentia_test
@requires_neuronx
@require_diffusers
def test_sdxl_export_and_inference_dyn(neuron_sdxl_dyn_path):
    neuron_pipeline = NeuronStableDiffusionXLPipeline.from_pretrained(neuron_sdxl_dyn_path)

    prompt = ["Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"] * 2
    prompt_2 = ["Van Gogh painting"] * 2
    negative_prompt_1 = ["low quality, low resolution"] * 2
    negative_prompt_2 = ["low quality, low resolution"] * 2
    image = neuron_pipeline(
        prompt=prompt,
        prompt_2=prompt_2,
        negative_prompt=negative_prompt_1,
        negative_prompt_2=negative_prompt_2,
        num_images_per_prompt=2,
    ).images[0]
    assert isinstance(image, PIL.Image.Image)


@is_inferentia_test
@requires_neuronx
@require_diffusers
def test_sdxl_img2img_export_and_inference(neuron_sdxl_num_img_per_prompt_1_non_dyn_path):
    neuron_pipeline = NeuronStableDiffusionXLImg2ImgPipeline.from_pretrained(
        neuron_sdxl_num_img_per_prompt_1_non_dyn_path
    )

    url = "https://huggingface.co/datasets/optimum/documentation-images/resolve/main/intel/openvino/sd_xl/castle_friedrich.png"
    init_image = _download_image(url)
    prompt = "a dog running, lake, moat"
    image = neuron_pipeline(prompt=prompt, image=init_image).images[0]
    assert isinstance(image, PIL.Image.Image)


@is_inferentia_test
@requires_neuronx
@require_diffusers
def test_sdxl_inpaint_export_and_inference(neuron_sdxl_num_img_per_prompt_1_non_dyn_path):
    neuron_pipeline = NeuronStableDiffusionXLInpaintPipeline.from_pretrained(
        neuron_sdxl_num_img_per_prompt_1_non_dyn_path
    )

    img_url = (
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-text2img.png"
    )
    mask_url = (
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-inpaint-mask.png"
    )
    init_image = _download_image(img_url).resize((64, 64))
    mask_image = _download_image(mask_url).resize((64, 64))
    prompt = "A deep sea diver floating"
    image = neuron_pipeline(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
    assert isinstance(image, PIL.Image.Image)


@is_inferentia_test
@requires_neuronx
@require_diffusers
def test_sdxl_compatibility_with_compel(neuron_sdxl_with_hidden_states_output_path):
    pipe = NeuronStableDiffusionXLPipeline.from_pretrained(neuron_sdxl_with_hidden_states_output_path)

    prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    negative_prompt = "low quality, low resolution"

    compel = Compel(
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True],
    )
    prompt_embeds, pooled = compel(prompt)
    neg_prompt_embeds, neg_pooled = compel(negative_prompt)
    positive_prompt_embeds, negative_prompt_embeds = compel.pad_conditioning_tensors_to_same_length(
        [prompt_embeds, neg_prompt_embeds]
    )

    image = pipe(
        prompt_embeds=positive_prompt_embeds,
        pooled_prompt_embeds=pooled,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_pooled_prompt_embeds=neg_pooled,
        output_type="pil",
        num_inference_steps=1,
    ).images[0]
    assert isinstance(image, PIL.Image.Image)


@is_inferentia_test
@requires_neuronx
@require_diffusers
def test_sdxl_from_pipe(neuron_sdxl_num_img_per_prompt_1_non_dyn_path):
    txt2img_pipeline = NeuronStableDiffusionXLPipeline.from_pretrained(neuron_sdxl_num_img_per_prompt_1_non_dyn_path)
    img2img_pipeline = NeuronStableDiffusionXLImg2ImgPipeline.from_pipe(txt2img_pipeline)
    url = "https://huggingface.co/datasets/optimum/documentation-images/resolve/main/intel/openvino/sd_xl/castle_friedrich.png"
    init_image = _download_image(url)
    prompt = "a dog running, lake, moat"
    image = img2img_pipeline(prompt=prompt, image=init_image).images[0]
    assert isinstance(image, PIL.Image.Image)
