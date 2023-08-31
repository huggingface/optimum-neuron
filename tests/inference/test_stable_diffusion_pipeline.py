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

import PIL
from parameterized import parameterized

from optimum.neuron import NeuronStableDiffusionPipeline
from optimum.neuron.modeling_diffusion import (
    NeuronModelTextEncoder,
    NeuronModelUnet,
    NeuronModelVaeDecoder,
    NeuronModelVaeEncoder,  # noqa
)
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx
from optimum.utils import logging
from optimum.utils.testing_utils import require_diffusers

from .inference_utils import MODEL_NAMES


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
            **input_shapes,
            **self.COMPILER_ARGS,
            device_ids=[0, 1],
        )
        self.assertIsInstance(neuron_pipeline.text_encoder, NeuronModelTextEncoder)
        self.assertIsInstance(neuron_pipeline.unet, NeuronModelUnet)
        self.assertIsInstance(neuron_pipeline.vae_encoder, NeuronModelVaeEncoder)
        self.assertIsInstance(neuron_pipeline.vae_decoder, NeuronModelVaeDecoder)

        prompt = "sailing ship in storm by Leonardo da Vinci"
        with self.assertRaises(Exception) as context:
            image = neuron_pipeline(prompt).images[0]
        self.assertIn("pipeline were compiled with", str(context.exception))

        prompts = ["sailing ship in storm by Leonardo da Vinci"] * num_images_per_prompt
        image = neuron_pipeline(prompts).images[0]
        self.assertIsInstance(image, PIL.Image.Image)

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_export_and_inference_dyn(self, model_arch):
        neuron_pipeline = self.NEURON_MODEL_CLASS.from_pretrained(
            MODEL_NAMES[model_arch],
            export=True,
            dynamic_batch_size=True,
            **self.STATIC_INPUTS_SHAPES,
            **self.COMPILER_ARGS,
            device_ids=[0, 1],
        )

        prompts = ["sailing ship in storm by Leonardo da Vinci"] * 2
        image = neuron_pipeline(prompts, num_images_per_prompt=2).images[0]
        self.assertIsInstance(image, PIL.Image.Image)
