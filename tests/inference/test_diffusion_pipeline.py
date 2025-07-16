# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import copy
import unittest

import cv2
import numpy as np
import PIL
import torch
from compel import Compel, ReturnedEmbeddingsType
from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image
from parameterized import parameterized

from optimum.neuron import (
    NeuronFluxPipeline,
    NeuronLatentConsistencyModelPipeline,
    NeuronPixArtAlphaPipeline,
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
    NeuronModelTransformer,
    NeuronModelUnet,
    NeuronModelVaeDecoder,
    NeuronModelVaeEncoder,
    NeuronMultiControlNetModel,
)
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx
from optimum.utils import logging
from optimum.utils.testing_utils import require_diffusers

from .inference_utils import LORA_WEIGHTS_TINY, MODEL_NAMES, download_image


logger = logging.get_logger()


@is_inferentia_test
@requires_neuronx
@require_diffusers
class NeuronStableDiffusionPipelineIntegrationTest(unittest.TestCase):
    NEURON_MODEL_CLASS = NeuronStableDiffusionPipeline
    STATIC_INPUTS_SHAPES = {"batch_size": 1, "height": 64, "width": 64}
    COMPILER_ARGS = {"auto_cast": "matmul", "auto_cast_type": "bf16"}
    SUPPORTED_ARCHITECTURES = [
        "stable-diffusion",
    ]
    ATOL_FOR_VALIDATION = 1e-3

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_export_and_inference_non_dyn(self, model_arch):
        num_images_per_prompt = 4
        input_shapes = copy.deepcopy(self.STATIC_INPUTS_SHAPES)
        input_shapes.update({"num_images_per_prompt": num_images_per_prompt})
        neuron_pipeline = self.NEURON_MODEL_CLASS.from_pretrained(
            MODEL_NAMES[model_arch],
            export=True,
            dynamic_batch_size=False,
            disable_neuron_cache=True,
            **input_shapes,
            **self.COMPILER_ARGS,
        )
        self.assertIsInstance(neuron_pipeline.text_encoder, NeuronModelTextEncoder)
        self.assertIsInstance(neuron_pipeline.unet, NeuronModelUnet)
        self.assertIsInstance(neuron_pipeline.vae_encoder, NeuronModelVaeEncoder)
        self.assertIsInstance(neuron_pipeline.vae_decoder, NeuronModelVaeDecoder)

        prompts = ["sailing ship in storm by Leonardo da Vinci"]
        image = neuron_pipeline(prompts, num_images_per_prompt=num_images_per_prompt).images[0]
        self.assertIsInstance(image, PIL.Image.Image)

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_export_and_inference_dyn(self, model_arch):
        neuron_pipeline = self.NEURON_MODEL_CLASS.from_pretrained(
            MODEL_NAMES[model_arch],
            export=True,
            dynamic_batch_size=True,
            **self.STATIC_INPUTS_SHAPES,
            **self.COMPILER_ARGS,
        )

        prompts = ["sailing ship in storm by Leonardo da Vinci"] * 2
        image = neuron_pipeline(prompts, num_images_per_prompt=2).images[0]
        self.assertIsInstance(image, PIL.Image.Image)

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_img2img_export_and_inference(self, model_arch):
        neuron_pipeline = NeuronStableDiffusionImg2ImgPipeline.from_pretrained(
            MODEL_NAMES[model_arch],
            export=True,
            dynamic_batch_size=False,
            **self.STATIC_INPUTS_SHAPES,
            **self.COMPILER_ARGS,
        )

        url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
        init_image = download_image(url)
        prompt = "ghibli style, a fantasy landscape with mountain, trees and lake, reflection"
        image = neuron_pipeline(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images[0]
        self.assertIsInstance(image, PIL.Image.Image)

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_inpaint_export_and_inference(self, model_arch):
        neuron_pipeline = NeuronStableDiffusionInpaintPipeline.from_pretrained(
            MODEL_NAMES[model_arch],
            export=True,
            dynamic_batch_size=False,
            **self.STATIC_INPUTS_SHAPES,
            **self.COMPILER_ARGS,
        )

        img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
        mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
        init_image = download_image(img_url).resize((512, 512))
        mask_image = download_image(mask_url).resize((512, 512))
        prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
        image = neuron_pipeline(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
        self.assertIsInstance(image, PIL.Image.Image)

    @parameterized.expand(["stable-diffusion-ip2p"], skip_on_empty=True)
    def test_instruct_pix2pix_export_and_inference(self, model_arch):
        neuron_pipeline = NeuronStableDiffusionInstructPix2PixPipeline.from_pretrained(
            MODEL_NAMES[model_arch],
            export=True,
            dynamic_batch_size=True,
            **self.STATIC_INPUTS_SHAPES,
            **self.COMPILER_ARGS,
        )

        img_url = "https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png"
        init_image = download_image(img_url).resize((512, 512))
        prompt = "Add a beautiful sunset"
        image = neuron_pipeline(prompt=prompt, image=init_image).images[0]
        self.assertIsInstance(image, PIL.Image.Image)

    @parameterized.expand(["latent-consistency"], skip_on_empty=True)
    def test_lcm_export_and_inference(self, model_arch):
        neuron_pipeline = NeuronLatentConsistencyModelPipeline.from_pretrained(
            MODEL_NAMES[model_arch],
            export=True,
            dynamic_batch_size=False,
            **self.STATIC_INPUTS_SHAPES,
            **self.COMPILER_ARGS,
        )

        prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
        image = neuron_pipeline(prompt, num_inference_steps=4, guidance_scale=8.0).images[0]
        self.assertIsInstance(image, PIL.Image.Image)

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_export_and_inference_with_fused_lora(self, model_arch):
        num_images_per_prompt = 1
        input_shapes = copy.deepcopy(self.STATIC_INPUTS_SHAPES)
        input_shapes.update({"num_images_per_prompt": num_images_per_prompt})
        lora_params = LORA_WEIGHTS_TINY[model_arch]
        neuron_pipeline = self.NEURON_MODEL_CLASS.from_pretrained(
            MODEL_NAMES[model_arch],
            export=True,
            dynamic_batch_size=False,
            lora_model_ids=lora_params[0],
            lora_weight_names=lora_params[1],
            lora_adapter_names=lora_params[2],
            lora_scales=0.9,
            **input_shapes,
            **self.COMPILER_ARGS,
        )
        self.assertIsInstance(neuron_pipeline.text_encoder, NeuronModelTextEncoder)
        self.assertIsInstance(neuron_pipeline.unet, NeuronModelUnet)
        self.assertIsInstance(neuron_pipeline.vae_encoder, NeuronModelVaeEncoder)
        self.assertIsInstance(neuron_pipeline.vae_decoder, NeuronModelVaeDecoder)

        prompts = ["A cute brown bear eating a slice of pizza"]
        image = neuron_pipeline(prompts, num_images_per_prompt=num_images_per_prompt).images[0]
        self.assertIsInstance(image, PIL.Image.Image)

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_compatibility_with_compel(self, model_arch):
        num_images_per_prompt = 1
        input_shapes = copy.deepcopy(self.STATIC_INPUTS_SHAPES)
        input_shapes.update({"num_images_per_prompt": num_images_per_prompt})
        pipe = self.NEURON_MODEL_CLASS.from_pretrained(
            MODEL_NAMES[model_arch],
            export=True,
            disable_neuron_cache=True,
            inline_weights_to_neff=True,
            output_hidden_states=True,
            **input_shapes,
            **self.COMPILER_ARGS,
        )

        prompt = "a red cat playing with a ball++"
        compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)

        prompt_embeds = compel_proc(prompt)

        image = pipe(prompt_embeds=prompt_embeds, num_inference_steps=2).images[0]
        self.assertIsInstance(image, PIL.Image.Image)

    @staticmethod
    def prepare_canny_image(image_url=None):
        if image_url is None:
            image_url = "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
        original_image = load_image(image_url)
        image = np.array(original_image)
        low_threshold = 100
        high_threshold = 200
        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_image = PIL.Image.fromarray(image)

        return canny_image

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_export_and_inference_with_single_controlnet(self, model_arch):
        input_shapes = copy.deepcopy(self.STATIC_INPUTS_SHAPES)
        input_shapes.update({"num_images_per_prompt": 1})
        controlnet_id = "hf-internal-testing/tiny-controlnet"
        neuron_pipeline = NeuronStableDiffusionControlNetPipeline.from_pretrained(
            MODEL_NAMES[model_arch],
            controlnet_ids=controlnet_id,
            export=True,
            **input_shapes,
            **self.COMPILER_ARGS,
        )
        self.assertIsInstance(neuron_pipeline.text_encoder, NeuronModelTextEncoder)
        self.assertIsInstance(neuron_pipeline.unet, NeuronModelUnet)
        self.assertIsInstance(neuron_pipeline.vae_encoder, NeuronModelVaeEncoder)
        self.assertIsInstance(neuron_pipeline.vae_decoder, NeuronModelVaeDecoder)
        self.assertIsInstance(neuron_pipeline.controlnet, NeuronControlNetModel)

        prompt = "the mona lisa"
        canny_image = NeuronStableDiffusionPipelineIntegrationTest.prepare_canny_image()
        image = neuron_pipeline(prompt, image=canny_image).images[0]
        neuron_pipeline.scheduler = UniPCMultistepScheduler.from_config(neuron_pipeline.scheduler.config)
        self.assertIsInstance(image, PIL.Image.Image)

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_export_and_inference_with_multiple_controlnet(self, model_arch):
        input_shapes = copy.deepcopy(self.STATIC_INPUTS_SHAPES)
        input_shapes.update({"num_images_per_prompt": 1})
        controlnet_id = "hf-internal-testing/tiny-controlnet"

        neuron_pipeline = NeuronStableDiffusionControlNetPipeline.from_pretrained(
            MODEL_NAMES[model_arch],
            controlnet_ids=[controlnet_id, controlnet_id],
            export=True,
            **input_shapes,
            **self.COMPILER_ARGS,
        )
        self.assertIsInstance(neuron_pipeline.text_encoder, NeuronModelTextEncoder)
        self.assertIsInstance(neuron_pipeline.unet, NeuronModelUnet)
        self.assertIsInstance(neuron_pipeline.vae_encoder, NeuronModelVaeEncoder)
        self.assertIsInstance(neuron_pipeline.vae_decoder, NeuronModelVaeDecoder)
        self.assertIsInstance(neuron_pipeline.controlnet, NeuronMultiControlNetModel)

        prompt = "the mona lisa"
        canny_image = NeuronStableDiffusionPipelineIntegrationTest.prepare_canny_image()
        image = neuron_pipeline(prompt, image=[canny_image, canny_image]).images[0]
        neuron_pipeline.scheduler = UniPCMultistepScheduler.from_config(neuron_pipeline.scheduler.config)
        self.assertIsInstance(image, PIL.Image.Image)


@is_inferentia_test
@requires_neuronx
@require_diffusers
class NeuronStableDiffusionXLPipelineIntegrationTest(unittest.TestCase):
    NEURON_MODEL_CLASS = NeuronStableDiffusionXLPipeline
    STATIC_INPUTS_SHAPES = {"batch_size": 1, "height": 64, "width": 64}
    COMPILER_ARGS = {"auto_cast": "all", "auto_cast_type": "bf16"}
    SUPPORTED_ARCHITECTURES = [
        "stable-diffusion-xl",
    ]
    ATOL_FOR_VALIDATION = 1e-3

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_export_and_inference_non_dyn(self, model_arch):
        num_images_per_prompt = 4
        input_shapes = copy.deepcopy(self.STATIC_INPUTS_SHAPES)
        input_shapes.update({"num_images_per_prompt": num_images_per_prompt})
        neuron_pipeline = self.NEURON_MODEL_CLASS.from_pretrained(
            MODEL_NAMES[model_arch],
            export=True,
            dynamic_batch_size=False,
            **input_shapes,
            **self.COMPILER_ARGS,
        )
        self.assertIsInstance(neuron_pipeline.text_encoder, NeuronModelTextEncoder)
        self.assertIsInstance(neuron_pipeline.text_encoder_2, NeuronModelTextEncoder)
        self.assertIsInstance(neuron_pipeline.unet, NeuronModelUnet)
        self.assertIsInstance(neuron_pipeline.vae_encoder, NeuronModelVaeEncoder)
        self.assertIsInstance(neuron_pipeline.vae_decoder, NeuronModelVaeDecoder)

        prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
        prompt_2 = "Van Gogh painting"
        negative_prompt_1 = "low quality, low resolution"
        negative_prompt_2 = "low quality, low resolution"

        image = neuron_pipeline(
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt_1,
            negative_prompt_2=negative_prompt_2,
            num_images_per_prompt=num_images_per_prompt,
        ).images[0]
        self.assertIsInstance(image, PIL.Image.Image)

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_export_and_inference_dyn(self, model_arch):
        neuron_pipeline = self.NEURON_MODEL_CLASS.from_pretrained(
            MODEL_NAMES[model_arch],
            export=True,
            dynamic_batch_size=True,
            **self.STATIC_INPUTS_SHAPES,
            **self.COMPILER_ARGS,
        )

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
        self.assertIsInstance(image, PIL.Image.Image)

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_img2img_export_and_inference(self, model_arch):
        neuron_pipeline = NeuronStableDiffusionXLImg2ImgPipeline.from_pretrained(
            MODEL_NAMES[model_arch],
            export=True,
            dynamic_batch_size=False,
            **self.STATIC_INPUTS_SHAPES,
            **self.COMPILER_ARGS,
        )

        url = "https://huggingface.co/datasets/optimum/documentation-images/resolve/main/intel/openvino/sd_xl/castle_friedrich.png"
        init_image = download_image(url)
        prompt = "a dog running, lake, moat"
        image = neuron_pipeline(prompt=prompt, image=init_image).images[0]
        self.assertIsInstance(image, PIL.Image.Image)

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_inpaint_export_and_inference(self, model_arch):
        neuron_pipeline = NeuronStableDiffusionXLInpaintPipeline.from_pretrained(
            MODEL_NAMES[model_arch],
            export=True,
            dynamic_batch_size=False,
            **self.STATIC_INPUTS_SHAPES,
            **self.COMPILER_ARGS,
        )

        img_url = (
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-text2img.png"
        )
        mask_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-inpaint-mask.png"
        init_image = download_image(img_url).resize((64, 64))
        mask_image = download_image(mask_url).resize((64, 64))
        prompt = "A deep sea diver floating"
        image = neuron_pipeline(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
        self.assertIsInstance(image, PIL.Image.Image)

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_compatibility_with_compel(self, model_arch):
        num_images_per_prompt = 1
        input_shapes = copy.deepcopy(self.STATIC_INPUTS_SHAPES)
        input_shapes.update({"num_images_per_prompt": num_images_per_prompt})
        pipe = self.NEURON_MODEL_CLASS.from_pretrained(
            MODEL_NAMES[model_arch],
            export=True,
            inline_weights_to_neff=True,
            output_hidden_states=True,
            **input_shapes,
            **self.COMPILER_ARGS,
        )

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
        self.assertIsInstance(image, PIL.Image.Image)

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_from_pipe(self, model_arch):
        txt2img_pipeline = NeuronStableDiffusionXLPipeline.from_pretrained(
            MODEL_NAMES[model_arch],
            export=True,
            dynamic_batch_size=False,
            **self.STATIC_INPUTS_SHAPES,
            **self.COMPILER_ARGS,
        )
        img2img_pipeline = NeuronStableDiffusionXLImg2ImgPipeline.from_pipe(txt2img_pipeline)
        url = "https://huggingface.co/datasets/optimum/documentation-images/resolve/main/intel/openvino/sd_xl/castle_friedrich.png"
        init_image = download_image(url)
        prompt = "a dog running, lake, moat"
        image = img2img_pipeline(prompt=prompt, image=init_image).images[0]
        self.assertIsInstance(image, PIL.Image.Image)


@is_inferentia_test
@requires_neuronx
@require_diffusers
class NeuronPixArtAlphaPipelineIntegrationTest(unittest.TestCase):
    ATOL_FOR_VALIDATION = 1e-3

    def test_export_and_inference_non_dyn(self):
        model_id = "hf-internal-testing/tiny-pixart-alpha-pipe"
        compiler_args = {"auto_cast": "none"}
        input_shapes = {"batch_size": 1, "height": 64, "width": 64, "sequence_length": 32}
        neuron_pipeline = NeuronPixArtAlphaPipeline.from_pretrained(
            model_id,
            export=True,
            torch_dtype=torch.bfloat16,
            dynamic_batch_size=False,
            disable_neuron_cache=True,
            **input_shapes,
            **compiler_args,
        )
        self.assertIsInstance(neuron_pipeline.text_encoder, NeuronModelTextEncoder)
        self.assertIsInstance(neuron_pipeline.transformer, NeuronModelTransformer)
        self.assertIsInstance(neuron_pipeline.vae_encoder, NeuronModelVaeEncoder)
        self.assertIsInstance(neuron_pipeline.vae_decoder, NeuronModelVaeDecoder)

        prompt = "Mario eating hamburgers."

        neuron_pipeline.transformer.config.sample_size = (
            32  # Skip the sample size check because the dummy model uses a smaller sample size (8).
        )
        image = neuron_pipeline(prompt=prompt, use_resolution_binning=False).images[
            0
        ]  # Set `use_resolution_binning=False` to prevent resizing.
        self.assertIsInstance(image, PIL.Image.Image)


@is_inferentia_test
@requires_neuronx
@require_diffusers
class NeuronFluxPipelineIntegrationTest(unittest.TestCase):
    ATOL_FOR_VALIDATION = 1e-3

    def test_export_and_inference(self):
        model_id = "hf-internal-testing/tiny-flux-pipe-gated-silu"
        compiler_args = {"auto_cast": "none"}
        input_shapes = {"batch_size": 1, "height": 8, "width": 8, "num_images_per_prompt": 1, "sequence_length": 256}
        neuron_pipeline = NeuronFluxPipeline.from_pretrained(
            model_id,
            export=True,
            torch_dtype=torch.bfloat16,
            tensor_parallel_size=2,
            dynamic_batch_size=False,
            disable_neuron_cache=True,
            **input_shapes,
            **compiler_args,
        )
        self.assertIsInstance(neuron_pipeline.text_encoder, NeuronModelTextEncoder)
        self.assertIsInstance(neuron_pipeline.text_encoder_2, NeuronModelTextEncoder)
        self.assertIsInstance(neuron_pipeline.transformer, NeuronModelTransformer)
        self.assertIsInstance(neuron_pipeline.vae_encoder, NeuronModelVaeEncoder)
        self.assertIsInstance(neuron_pipeline.vae_decoder, NeuronModelVaeDecoder)

        prompt = "Mario eating hamburgers."
        image = neuron_pipeline(
            prompt, num_inference_steps=4, max_sequence_length=256, generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]
        self.assertIsInstance(image, PIL.Image.Image)


if __name__ == "__main__":
    unittest.main()
