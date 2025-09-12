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

import PIL
import torch
from diffusers.utils import load_image
from optimum.utils.testing_utils import require_diffusers

from optimum.neuron import NeuronFluxInpaintPipeline, NeuronFluxKontextPipeline, NeuronFluxPipeline
from optimum.neuron.modeling_diffusion import (
    NeuronModelTextEncoder,
    NeuronModelTransformer,
    NeuronModelVaeDecoder,
    NeuronModelVaeEncoder,
)
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


@is_inferentia_test
@requires_neuronx
@require_diffusers
def test_flux_txt2img(neuron_flux_tp2_path):
    neuron_pipeline = NeuronFluxPipeline.from_pretrained(neuron_flux_tp2_path)

    assert isinstance(neuron_pipeline.text_encoder, NeuronModelTextEncoder)
    assert isinstance(neuron_pipeline.text_encoder_2, NeuronModelTextEncoder)
    assert isinstance(neuron_pipeline.transformer, NeuronModelTransformer)
    assert isinstance(neuron_pipeline.vae_encoder, NeuronModelVaeEncoder)
    assert isinstance(neuron_pipeline.vae_decoder, NeuronModelVaeDecoder)

    prompt = "Mario eating hamburgers."
    image = neuron_pipeline(
        prompt, num_inference_steps=4, max_sequence_length=256, generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    assert isinstance(image, PIL.Image.Image)


@is_inferentia_test
@requires_neuronx
@require_diffusers
def test_flux_inpaint(neuron_flux_tp2_path):
    neuron_pipeline = NeuronFluxInpaintPipeline.from_pretrained(neuron_flux_tp2_path)

    assert isinstance(neuron_pipeline.text_encoder, NeuronModelTextEncoder)
    assert isinstance(neuron_pipeline.text_encoder_2, NeuronModelTextEncoder)
    assert isinstance(neuron_pipeline.transformer, NeuronModelTransformer)
    assert isinstance(neuron_pipeline.vae_encoder, NeuronModelVaeEncoder)
    assert isinstance(neuron_pipeline.vae_decoder, NeuronModelVaeDecoder)

    prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
    img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
    source = load_image(img_url)
    mask = load_image(mask_url)
    image = neuron_pipeline(prompt=prompt, image=source, mask_image=mask, max_sequence_length=256).images[0]
    assert isinstance(image, PIL.Image.Image)


@is_inferentia_test
@requires_neuronx
@require_diffusers
def test_flux_kontext_img_edit(neuron_flux_kontext_tp2_path):
    neuron_pipeline = NeuronFluxKontextPipeline.from_pretrained(neuron_flux_kontext_tp2_path)

    assert isinstance(neuron_pipeline.text_encoder, NeuronModelTextEncoder)
    assert isinstance(neuron_pipeline.text_encoder_2, NeuronModelTextEncoder)
    assert isinstance(neuron_pipeline.transformer, NeuronModelTransformer)
    assert isinstance(neuron_pipeline.vae_encoder, NeuronModelVaeEncoder)
    assert isinstance(neuron_pipeline.vae_decoder, NeuronModelVaeDecoder)

    prompt = "Change the dog on the bench into a labrador"
    img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    source = load_image(img_url)

    image = neuron_pipeline(
        prompt=prompt,
        image=source,
        max_sequence_length=256,
        _auto_resize=False,
        max_area=64,
    ).images[0]
    assert isinstance(image, PIL.Image.Image)
