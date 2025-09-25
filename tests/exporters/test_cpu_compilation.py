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

import subprocess
import tempfile
import unittest
from pathlib import Path

from parameterized import parameterized

from optimum.neuron import (
    NeuronModelForFeatureExtraction,
    NeuronModelForSeq2SeqLM,
    NeuronStableDiffusionPipeline,
)
from optimum.neuron.utils.testing_utils import requires_neuronx


# Models to test for CPU backend compilation
CPU_BACKEND_ENCODER_MODELS = {
    "bert": "hf-internal-testing/tiny-random-BertModel",
}

CPU_BACKEND_SEQ2SEQ_MODELS = {
    "t5": "hf-internal-testing/tiny-random-t5",
}

CPU_BACKEND_DIFFUSION_MODELS = {
    "stable-diffusion": "hf-internal-testing/tiny-stable-diffusion-torch",
}

INSTANCE_TYPES = ["inf2", "trn1", "trn1n", "trn2"]


@requires_neuronx
class NeuronCPUBackendIntegrationTest(unittest.TestCase):
    def test_no_instance_type_cli():
        """
        Raise when `--instance_type` is not specified.
        """
        pass

    def test_no_instance_type_modeling():
        """
        Raise when `--instance_type` is not specified.
        """
        pass


@requires_neuronx
class NeuronEncoderCPUBackendTestCase(unittest.TestCase):
    """
    Tests for CPU backend compilation of encoder models.
    This class tests both the export functionality and integration aspects
    for encoder models using cpu_backend=True.
    """

    @parameterized.expand(INSTANCE_TYPES)
    def test_cpu_backend_export_cli(self, instance_type):
        """
        Test CPU backend compilation for encoder models using Optimum CLI and verify artifacts creation.
        """
        model_id = CPU_BACKEND_ENCODER_MODELS["bert"]
        with tempfile.TemporaryDirectory() as tempdir:
            subprocess.run(
                [
                    "optimum-cli",
                    "export",
                    "neuron",
                    "--model",
                    model_id,
                    "--sequence_length",
                    "16",
                    "--batch_size",
                    "1",
                    "--task",
                    "text-classification",
                    "--instance_type",
                    instance_type,
                    tempdir,
                ],
                shell=False,
                check=True,
            )
            # Verify artifacts were created
            self._verify_encoder_artifacts(tempdir)

    def test_cpu_backend_with_modeling(self):
        """
        Test CPU backend compilation using NeuronModel API.
        """
        model_id = CPU_BACKEND_ENCODER_MODELS["bert"]
        instance_type = "inf2"

        compiler_configs = {"auto_cast": "matmul", "auto_cast_type": "bf16", "instance_type": instance_type}
        input_shapes = {
            "batch_size": 1,
            "sequence_length": 16,
        }
        neuron_model = NeuronModelForFeatureExtraction.from_pretrained(
            model_id,
            export=True,
            **compiler_configs,
            **input_shapes,
        )
        self.assertIsNone(neuron_model, "CPU backend export should return None.")

    def _verify_encoder_artifacts(self, save_dir: Path):
        """Verify that encoder compilation artifacts are created properly."""
        # Check main config file
        config_file = save_dir / "config.json"
        self.assertTrue(config_file.exists(), f"Main config file {config_file} not found")

        # Check for neuron files
        neuron_files = list(save_dir.glob("*.neuron"))
        self.assertGreaterEqual(len(neuron_files), 1, f"No .neuron files found in {save_dir}")

        # Verify neuron files are not empty
        for neuron_file in neuron_files:
            self.assertGreater(neuron_file.stat().st_size, 0, f"Neuron file {neuron_file} is empty")


@requires_neuronx
class NeuronCPUBackendSeq2SeqTestCase(unittest.TestCase):
    """
    Tests for CPU backend compilation of seq2seq models.
    This class tests both the export functionality and integration aspects
    for seq2seq models (T5) using cpu_backend=True.
    """

    @parameterized.expand(INSTANCE_TYPES)
    def test_cpu_backend_export_cli(self, instance_type):
        """
        Test CPU backend compilation for seq2seq models using Optimum CLI and verify artifacts creation.
        """
        model_id = CPU_BACKEND_SEQ2SEQ_MODELS["t5"]
        with tempfile.TemporaryDirectory() as tempdir:
            subprocess.run(
                [
                    "optimum-cli",
                    "export",
                    "neuron",
                    "--model",
                    model_id,
                    "--task",
                    "text2text-generation",
                    "--batch_size",
                    "1",
                    "--sequence_length",
                    "16",
                    "--num_beams",
                    "4",
                    "--auto_cast",
                    "matmul",
                    "--auto_cast_type",
                    "bf16",
                    "--output_hidden_states",
                    "--output_attentions",
                    "--instance_type",
                    instance_type,
                    tempdir,
                ],
                shell=False,
                check=True,
            )
            # Verify artifacts were created
            self._verify_seq2seq_artifacts(tempdir)

    def test_cpu_backend_with_modeling(self):
        """
        Test CPU backend compilation using NeuronModel API for seq2seq models.
        """
        model_id = CPU_BACKEND_SEQ2SEQ_MODELS["t5"]
        instance_type = "inf2"

        compiler_configs = {"auto_cast": "matmul", "auto_cast_type": "bf16", "instance_type": instance_type}
        input_shapes = {
            "batch_size": 1,
            "sequence_length": 16,
            "num_beams": 4,
        }

        neuron_model = NeuronModelForSeq2SeqLM.from_pretrained(
            model_id,
            export=True,
            **compiler_configs,
            **input_shapes,
        )
        self.assertIsNone(neuron_model, "CPU backend export should return None.")

    def _verify_seq2seq_artifacts(self, save_dir: Path):
        """Verify that seq2seq compilation artifacts are created properly."""
        # Check main config file
        config_file = save_dir / "config.json"
        self.assertTrue(config_file.exists(), f"Main config file {config_file} not found")

        # Check encoder directory and artifacts
        encoder_dir = save_dir / "encoder"
        self.assertTrue(encoder_dir.exists(), f"Encoder directory {encoder_dir} not found")

        encoder_config = encoder_dir / "config.json"
        self.assertTrue(encoder_config.exists(), f"Encoder config {encoder_config} not found")

        encoder_neuron_files = list(encoder_dir.glob("*.neuron"))
        self.assertGreaterEqual(len(encoder_neuron_files), 1, f"No encoder .neuron files found in {encoder_dir}")

        # Check decoder directory and artifacts
        decoder_dir = save_dir / "decoder"
        self.assertTrue(decoder_dir.exists(), f"Decoder directory {decoder_dir} not found")

        decoder_config = decoder_dir / "config.json"
        self.assertTrue(decoder_config.exists(), f"Decoder config {decoder_config} not found")

        decoder_neuron_files = list(decoder_dir.glob("*.neuron"))
        self.assertGreaterEqual(len(decoder_neuron_files), 1, f"No decoder .neuron files found in {decoder_dir}")

        # Verify files are not empty
        for neuron_file in encoder_neuron_files + decoder_neuron_files:
            self.assertGreater(neuron_file.stat().st_size, 0, f"Neuron file {neuron_file} is empty")


@requires_neuronx
class NeuronCPUBackendDiffusionTestCase(unittest.TestCase):
    """
    Tests for CPU backend compilation of diffusion models.
    This class tests both the export functionality and integration aspects
    for diffusion models (Stable Diffusion) using cpu_backend=True.
    """

    @parameterized.expand(INSTANCE_TYPES)
    def test_cpu_backend_export_cli(self, instance_type):
        """
        Test CPU backend compilation for diffusion models using Optimum CLI and verify artifacts creation.
        """
        model_id = CPU_BACKEND_DIFFUSION_MODELS["stable-diffusion"]
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
                    "--instance_type",
                    instance_type,
                    tempdir,
                ],
                shell=False,
                check=True,
            )
            # Verify artifacts were created
            self._verify_diffusion_artifacts(tempdir)

    def test_cpu_backend_with_modeling(self):
        """
        Test CPU backend compilation using NeuronModel API for diffusion models.
        """
        model_id = CPU_BACKEND_DIFFUSION_MODELS["stable-diffusion"]
        instance_type = "inf2"

        compiler_configs = {"auto_cast": "matmul", "auto_cast_type": "bf16", "instance_type": instance_type}
        input_shapes = {
            "batch_size": 1,
            "height": 64,
            "width": 64,
            "num_images_per_prompt": 1,
        }

        neuron_model = NeuronStableDiffusionPipeline.from_pretrained(
            model_id,
            export=True,
            **compiler_configs,
            **input_shapes,
        )
        self.assertIsNone(neuron_model, "CPU backend export should return None.")

    def _verify_diffusion_artifacts(self, save_dir: Path):
        """Verify that diffusion compilation artifacts are created properly."""
        # Check for typical diffusion model components directories
        expected_components = ["text_encoder", "unet", "vae_encoder", "vae_decoder"]

        for component in expected_components:
            component_dir = save_dir / component
            if component_dir.exists():
                component_config = component_dir / "config.json"
                self.assertTrue(component_config.exists(), f"Component config {component_config} not found")

                neuron_files = list(component_dir.glob("*.neuron"))
                for neuron_file in neuron_files:
                    self.assertGreater(neuron_file.stat().st_size, 0, f"Neuron file {neuron_file} is empty")

        # Check main model config file
        main_config = save_dir / "model_index.json"
        if main_config.exists():
            self.assertTrue(main_config.exists(), f"Main model config {main_config} not found")
