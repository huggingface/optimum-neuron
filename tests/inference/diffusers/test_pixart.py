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
from optimum.utils.testing_utils import require_diffusers

from optimum.neuron import NeuronPixArtAlphaPipeline
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
def test_export_and_inference_non_dyn(neuron_pixart_alpha_path):
    neuron_pipeline = NeuronPixArtAlphaPipeline.from_pretrained(neuron_pixart_alpha_path)
    assert isinstance(neuron_pipeline.text_encoder, NeuronModelTextEncoder)
    assert isinstance(neuron_pipeline.transformer, NeuronModelTransformer)
    assert isinstance(neuron_pipeline.vae_encoder, NeuronModelVaeEncoder)
    assert isinstance(neuron_pipeline.vae_decoder, NeuronModelVaeDecoder)

    prompt = "Mario eating hamburgers."

    neuron_pipeline.transformer.config.sample_size = (
        32  # Skip the sample size check because the dummy model uses a smaller sample size (8).
    )
    image = neuron_pipeline(prompt=prompt, use_resolution_binning=False).images[
        0
    ]  # Set `use_resolution_binning=False` to prevent resizing.
    assert isinstance(image, PIL.Image.Image)
