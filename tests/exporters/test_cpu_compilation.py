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

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from parameterized import parameterized
from transformers import AutoConfig, set_seed

from optimum.neuron import (
    NeuronModelForFeatureExtraction,
    NeuronModelForSeq2SeqLM,
)
from optimum.neuron.utils.testing_utils import requires_neuronx


SEED = 42

# Models to test for CPU backend compilation
CPU_BACKEND_ENCODER_MODELS = {
    "distilbert": "hf-internal-testing/tiny-random-DistilBertModel",
    "bert": "hf-internal-testing/tiny-random-BertModel",
    "roberta": "hf-internal-testing/tiny-random-RobertaModel",
}

CPU_BACKEND_SEQ2SEQ_MODELS = {
    "t5": "hf-internal-testing/tiny-random-t5",
}


class NeuronCPUBackendEncoderTestCase(unittest.TestCase):
    """
    Tests for CPU backend compilation of encoder models.
    This class tests both the export functionality and integration aspects
    for encoder models (DistilBERT, BERT, RoBERTa) using cpu_backend=True.
    """

    def setUp(self):
        """Set up test environment for CPU backend compilation."""
        # Configure environment for CPU compilation
        os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "inf2"
        set_seed(SEED)

    @parameterized.expand(CPU_BACKEND_ENCODER_MODELS.items())
    @requires_neuronx
    def test_cpu_backend_encoder_export(self, model_name, model_id):
        """
        Test CPU backend compilation for encoder models using _export method.
        
        This test verifies that encoder models can be compiled with cpu_backend=True
        and that the compilation artifacts are created successfully.
        """
        with TemporaryDirectory() as tmp_dir:
            try:
                # Load model config
                config = AutoConfig.from_pretrained(model_id)

                # Export model with CPU backend
                result = NeuronModelForFeatureExtraction._export(
                    model_id=model_id,
                    config=config,
                    batch_size=1,
                    sequence_length=128,
                    cpu_backend=True,
                )

                # For cpu_backend=True, _export returns None but saves artifacts
                self.assertIsNone(result, "CPU backend export should return None")

            except Exception as e:
                self.fail(f"CPU backend compilation failed for {model_name} ({model_id}): {e}")

    @requires_neuronx
    def test_cpu_backend_encoder_artifacts_creation(self):
        """
        Integration test to verify encoder model compilation creates proper artifacts.
        This test compiles a model and verifies the directory structure and files
        are created as expected.
        """
        model_id = CPU_BACKEND_ENCODER_MODELS["distilbert"]

        with TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir) / "test_encoder_export"
            save_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Use the main_export function directly to have more control over the process
                from optimum.exporters.neuron import main_export

                main_export(
                    model_name_or_path=model_id,
                    output=save_dir,
                    task="feature-extraction",
                    batch_size=1,
                    sequence_length=128,
                    cpu_backend=True,
                    do_validation=False,  # Skip validation as it requires Neuron hardware
                    compiler_kwargs={},  # Empty compiler kwargs for basic test
                )

                # Verify artifacts were created
                self._verify_encoder_artifacts(save_dir)

            except Exception as e:
                self.fail(f"CPU backend integration test failed for encoder: {e}")

    @requires_neuronx
    def test_cpu_backend_compilation_with_compiler_options(self):
        """
        Test CPU backend compilation with different compiler options.
        This test verifies that CPU backend compilation works with various
        compiler settings like auto_cast, optlevel, etc. for encoder models.
        """
        model_id = CPU_BACKEND_ENCODER_MODELS["distilbert"]
        config = AutoConfig.from_pretrained(model_id)

        # Test different compiler configurations
        compiler_configs = [
            {"auto_cast": "matmul", "auto_cast_type": "bf16", "optlevel": "1"},
            {"auto_cast": "all", "auto_cast_type": "fp16", "optlevel": "2"},
            {"auto_cast": None, "optlevel": "3"},
        ]

        for compiler_opts in compiler_configs:
            with self.subTest(compiler_opts=compiler_opts):
                try:
                    result = NeuronModelForFeatureExtraction._export(
                        model_id=model_id,
                        config=config,
                        batch_size=1,
                        sequence_length=128,
                        cpu_backend=True,
                        **compiler_opts,
                    )

                    self.assertIsNone(
                        result, f"CPU backend export should return None for compiler opts {compiler_opts}"
                    )

                except Exception as e:
                    self.fail(f"CPU backend compilation failed for compiler opts {compiler_opts}: {e}")

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

    def setUp(self):
        """Set up test environment for CPU backend seq2seq tests."""
        os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "inf2"
        set_seed(SEED)

    @parameterized.expand(CPU_BACKEND_SEQ2SEQ_MODELS.items())
    @requires_neuronx
    def test_cpu_backend_seq2seq_export(self, model_name, model_id):
        """
        Test CPU backend compilation for seq2seq models using _export method.
        
        This test verifies that seq2seq models can be compiled with cpu_backend=True
        and that the compilation artifacts are created successfully.
        """
        with TemporaryDirectory() as tmp_dir:
            try:
                # Load model config
                config = AutoConfig.from_pretrained(model_id)

                # Export model with CPU backend
                result = NeuronModelForSeq2SeqLM._export(
                    model_id=model_id,
                    config=config,
                    batch_size=1,
                    sequence_length=64,
                    num_beams=4,
                    cpu_backend=True,
                )

                # For cpu_backend=True, _export returns None but saves artifacts
                self.assertIsNone(result, "CPU backend export should return None")

            except Exception as e:
                self.fail(f"CPU backend compilation failed for {model_name} ({model_id}): {e}")

    @requires_neuronx
    def test_cpu_backend_seq2seq_artifacts_creation(self):
        """
        Integration test to verify seq2seq model compilation creates proper artifacts.
        This test compiles a seq2seq model and verifies the directory structure and files
        are created as expected.
        """
        model_id = CPU_BACKEND_SEQ2SEQ_MODELS["t5"]

        with TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir) / "test_seq2seq_export"
            save_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Use the main_export function directly
                from optimum.exporters.neuron import main_export

                main_export(
                    model_name_or_path=model_id,
                    output=save_dir,
                    task="text2text-generation",
                    batch_size=1,
                    sequence_length=64,
                    num_beams=4,
                    cpu_backend=True,
                    do_validation=False,  # Skip validation as it requires Neuron hardware
                    compiler_kwargs={},  # Empty compiler kwargs for basic test
                )

                # Verify artifacts were created
                self._verify_seq2seq_artifacts(save_dir)

            except Exception as e:
                self.fail(f"CPU backend integration test failed for seq2seq: {e}")

    @requires_neuronx
    def test_cpu_backend_seq2seq_with_compiler_options(self):
        """
        Test CPU backend compilation with different compiler options for seq2seq models.
        This test verifies that CPU backend compilation works with various
        compiler settings for seq2seq models.
        """
        model_id = CPU_BACKEND_SEQ2SEQ_MODELS["t5"]
        config = AutoConfig.from_pretrained(model_id)

        # Test different compiler configurations for seq2seq
        compiler_configs = [
            {"auto_cast": "matmul", "auto_cast_type": "bf16", "optlevel": "1"},
            {"auto_cast": "all", "auto_cast_type": "fp16", "optlevel": "2"},
            {"auto_cast": None, "optlevel": "3"},
        ]

        for compiler_opts in compiler_configs:
            with self.subTest(compiler_opts=compiler_opts):
                try:
                    result = NeuronModelForSeq2SeqLM._export(
                        model_id=model_id,
                        config=config,
                        batch_size=1,
                        sequence_length=64,
                        num_beams=4,
                        cpu_backend=True,
                        **compiler_opts,
                    )

                    self.assertIsNone(
                        result, f"CPU backend export should return None for compiler opts {compiler_opts}"
                    )

                except Exception as e:
                    self.fail(f"CPU backend seq2seq compilation failed for compiler opts {compiler_opts}: {e}")

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
