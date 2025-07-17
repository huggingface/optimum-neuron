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

import unittest

import PIL
import torch

from optimum.neuron import NeuronFluxPipeline
from optimum.neuron.modeling_diffusion import (
    NeuronModelTextEncoder,
    NeuronModelTransformer,
    NeuronModelVaeDecoder,
    NeuronModelVaeEncoder,
)
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx
from optimum.utils.testing_utils import require_diffusers


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
