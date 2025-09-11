# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import subprocess
import tempfile
import unittest

from optimum.exporters.neuron.model_configs import *  # noqa: F403
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx
from optimum.utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@is_inferentia_test
class TestExportCLI(unittest.TestCase):
    @requires_neuronx
    def test_stable_diffusion(self):
        model_ids = ["hf-internal-testing/tiny-stable-diffusion-torch", "echarlaix/tiny-random-latent-consistency"]
        for model_id in model_ids:
            with tempfile.TemporaryDirectory() as tempdir:
                subprocess.run(
                    [
                        "optimum-cli",
                        "export",
                        "neuron",
                        "--model",
                        model_id,
                        "--batch_size",
                        "1",
                        "--height",
                        "64",
                        "--width",
                        "64",
                        "--num_images_per_prompt",
                        "1",
                        "--auto_cast",
                        "matmul",
                        "--auto_cast_type",
                        "bf16",
                        tempdir,
                    ],
                    shell=False,
                    check=True,
                )

    @requires_neuronx
    def test_pixart(self):
        model_ids = ["hf-internal-testing/tiny-pixart-alpha-pipe"]
        for model_id in model_ids:
            with tempfile.TemporaryDirectory() as tempdir:
                subprocess.run(
                    [
                        "optimum-cli",
                        "export",
                        "neuron",
                        "--model",
                        model_id,
                        "--batch_size",
                        "1",
                        "--height",
                        "8",
                        "--width",
                        "8",
                        "--sequence_length",
                        "16",
                        "--num_images_per_prompt",
                        "1",
                        "--torch_dtype",
                        "bfloat16",
                        tempdir,
                    ],
                    shell=False,
                    check=True,
                )

    @requires_neuronx
    def test_flux_tp2(self):
        model_ids = ["hf-internal-testing/tiny-flux-pipe-gated-silu"]
        for model_id in model_ids:
            with tempfile.TemporaryDirectory() as tempdir:
                subprocess.run(
                    [
                        "optimum-cli",
                        "export",
                        "neuron",
                        "--model",
                        model_id,
                        "--tensor_parallel_size",
                        "2",
                        "--batch_size",
                        "1",
                        "--height",
                        "8",
                        "--width",
                        "8",
                        "--num_images_per_prompt",
                        "1",
                        "--torch_dtype",
                        "bfloat16",
                        tempdir,
                    ],
                    shell=False,
                    check=True,
                )

    @requires_neuronx
    def test_stable_diffusion_multi_lora(self):
        model_id = "hf-internal-testing/tiny-stable-diffusion-torch"
        lora_model_id = "Jingya/tiny-stable-diffusion-lora-64"
        lora_weight_name = "pytorch_lora_weights.safetensors"
        adpater_name = "pokemon"
        with tempfile.TemporaryDirectory() as tempdir:
            subprocess.run(
                [
                    "optimum-cli",
                    "export",
                    "neuron",
                    "--model",
                    model_id,
                    "--batch_size",
                    "1",
                    "--height",
                    "64",
                    "--width",
                    "64",
                    "--num_images_per_prompt",
                    "4",
                    "--lora_model_ids",
                    lora_model_id,
                    "--lora_weight_names",
                    lora_weight_name,
                    "--lora_adapter_names",
                    adpater_name,
                    "--lora_scales",
                    "0.9",
                    "--auto_cast",
                    "matmul",
                    "--auto_cast_type",
                    "bf16",
                    tempdir,
                ],
                shell=False,
                check=True,
            )

    @requires_neuronx
    def test_stable_diffusion_single_controlnet(self):
        model_id = "hf-internal-testing/tiny-stable-diffusion-torch"
        controlnet_id = "hf-internal-testing/tiny-controlnet"
        with tempfile.TemporaryDirectory() as tempdir:
            subprocess.run(
                [
                    "optimum-cli",
                    "export",
                    "neuron",
                    "--model",
                    model_id,
                    "--batch_size",
                    "1",
                    "--height",
                    "64",
                    "--width",
                    "64",
                    "--controlnet_ids",
                    controlnet_id,
                    "--num_images_per_prompt",
                    "1",
                    "--auto_cast",
                    "matmul",
                    "--auto_cast_type",
                    "bf16",
                    tempdir,
                ],
                shell=False,
                check=True,
            )

    @requires_neuronx
    def test_stable_diffusion_xl(self):
        model_id = "echarlaix/tiny-random-stable-diffusion-xl"
        with tempfile.TemporaryDirectory() as tempdir:
            subprocess.run(
                [
                    "optimum-cli",
                    "export",
                    "neuron",
                    "--model",
                    model_id,
                    "--batch_size",
                    "1",
                    "--height",
                    "64",
                    "--width",
                    "64",
                    "--num_images_per_prompt",
                    "4",
                    "--auto_cast",
                    "matmul",
                    "--auto_cast_type",
                    "bf16",
                    tempdir,
                ],
                shell=False,
                check=True,
            )

    @requires_neuronx
    def test_replace_unet(self):
        model_id = "echarlaix/tiny-random-stable-diffusion-xl"
        unet_id = "Jingya/tiny-random-sdxl-unet"
        with tempfile.TemporaryDirectory() as tempdir:
            subprocess.run(
                [
                    "optimum-cli",
                    "export",
                    "neuron",
                    "--model",
                    model_id,
                    "--unet",
                    unet_id,
                    "--batch_size",
                    "1",
                    "--height",
                    "64",
                    "--width",
                    "64",
                    "--num_images_per_prompt",
                    "4",
                    "--auto_cast",
                    "matmul",
                    "--auto_cast_type",
                    "bf16",
                    tempdir,
                ],
                shell=False,
                check=True,
            )
