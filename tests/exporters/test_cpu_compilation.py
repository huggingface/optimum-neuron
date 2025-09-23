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

from diffusers import StableDiffusionPipeline
from parameterized import parameterized
from transformers import AutoConfig, set_seed

from optimum.exporters.neuron import (
    build_stable_diffusion_components_mandatory_shapes,
    export_models,
)
from optimum.exporters.neuron.__main__ import get_submodels_and_neuron_configs
from optimum.neuron import (
    NeuronModelForCausalLM,
    NeuronModelForFeatureExtraction,
    NeuronModelForSeq2SeqLM,
)
from optimum.neuron.utils.testing_utils import requires_neuronx


SEED = 42

# Models to test for CPU backend compilation
CPU_BACKEND_ENCODER_MODELS = {
    "bert": "hf-internal-testing/tiny-random-BertModel",
}

CPU_BACKEND_SEQ2SEQ_MODELS = {
    "t5": "hf-internal-testing/tiny-random-t5",
}

CPU_BACKEND_DECODER_MODELS = {
    "llama": "hf-internal-testing/tiny-random-LlamaForCausalLM",
}

CPU_BACKEND_DIFFUSION_MODELS = {
    "stable-diffusion": "hf-internal-testing/tiny-stable-diffusion-torch",
}


class NeuronCPUBackendEncoderTestCase(unittest.TestCase):
    """
    Tests for CPU backend compilation of encoder models.
    This class tests both the export functionality and integration aspects
    for encoder models using cpu_backend=True.
    """

    def setUp(self):
        """Set up test environment for CPU backend compilation."""
        # Configure environment for CPU compilation
        os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "inf2"
        set_seed(SEED)

    @parameterized.expand(CPU_BACKEND_ENCODER_MODELS.items())
    @requires_neuronx
    def test_cpu_backend_encoder_export_and_artifacts(self, model_name, model_id):
        """
        Test CPU backend compilation for encoder models and verify artifacts creation.
        This test verifies that encoder models can be compiled with cpu_backend=True
        and that the compilation artifacts are created successfully.
        """
        with TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir) / "test_encoder_export"
            save_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Use the main_export function to export and create artifacts
                from optimum.exporters.neuron import main_export

                result = main_export(
                    model_name_or_path=model_id,
                    output=save_dir,
                    task="feature-extraction",
                    batch_size=1,
                    sequence_length=128,
                    cpu_backend=True,
                    do_validation=False,  # Skip validation as it requires Neuron hardware
                    compiler_kwargs={},  # Empty compiler kwargs for basic test
                )

                # For cpu_backend=True, main_export returns None but saves artifacts
                self.assertIsNone(result, "CPU backend export should return None")

                # Verify artifacts were created
                self._verify_encoder_artifacts(save_dir)

            except Exception as e:
                self.fail(f"CPU backend compilation and artifacts test failed for {model_name} ({model_id}): {e}")

    @requires_neuronx
    def test_cpu_backend_compilation_with_compiler_options(self):
        """
        Test CPU backend compilation with different compiler options.
        This test verifies that CPU backend compilation works with various
        compiler settings like auto_cast, optlevel, etc. for encoder models.
        """
        model_id = CPU_BACKEND_ENCODER_MODELS["bert"]
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
    def test_cpu_backend_seq2seq_export_and_artifacts(self, model_name, model_id):
        """
        Test CPU backend compilation for seq2seq models and verify artifacts creation.
        This test verifies that seq2seq models can be compiled with cpu_backend=True
        and that the compilation artifacts are created successfully.
        """
        with TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir) / "test_seq2seq_export"
            save_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Use the main_export function to export and create artifacts
                from optimum.exporters.neuron import main_export

                result = main_export(
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

                # For cpu_backend=True, main_export returns None but saves artifacts
                self.assertIsNone(result, "CPU backend export should return None")

                # Verify artifacts were created
                self._verify_seq2seq_artifacts(save_dir)

            except Exception as e:
                self.fail(f"CPU backend compilation and artifacts test failed for {model_name} ({model_id}): {e}")

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


@requires_neuronx
class NeuronCPUBackendDecoderTestCase(unittest.TestCase):
    """
    Tests for CPU backend compilation of decoder models.
    This class tests both the export functionality and integration aspects
    for decoder models using cpu_backend=True.
    """

    def setUp(self):
        """Set up test environment for CPU backend decoder tests."""
        os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "inf2"
        set_seed(SEED)

    @parameterized.expand(CPU_BACKEND_DECODER_MODELS.items())
    @requires_neuronx
    def test_cpu_backend_decoder_export_and_artifacts(self, model_name, model_id):
        """
        Test CPU backend compilation for decoder models and verify artifacts creation.
        This test verifies that decoder models can be compiled with cpu_backend=True
        and that the compilation artifacts are created successfully.
        """
        with TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir) / "test_decoder_export"
            save_dir.mkdir(parents=True, exist_ok=True)

            try:
                export_kwargs = {
                    "batch_size": 1,
                    "sequence_length": 128,
                    "tensor_parallel_size": 1,
                    "auto_cast_type": "bf16",
                }
                neuron_config = NeuronModelForCausalLM.get_neuron_config(model_name_or_path=model_id, **export_kwargs)
                model = NeuronModelForCausalLM.export(model_id=model_id, neuron_config=neuron_config)
                self.assertIsNotNone(model, "CPU backend export should return a model instance")

                # Save model and verify artifacts
                model.save_pretrained(save_dir)
                self._verify_decoder_artifacts(save_dir)

            except Exception as e:
                self.fail(f"CPU backend compilation and artifacts test failed for {model_name} ({model_id}): {e}")

    @requires_neuronx
    def test_cpu_backend_decoder_with_compiler_options(self):
        """
        Test CPU backend compilation with different compiler options for decoder models.
        This test verifies that CPU backend compilation works with various
        compiler settings for decoder models.
        """
        model_id = CPU_BACKEND_DECODER_MODELS["llama"]

        # Test different compiler configurations for decoder
        compiler_configs = [
            {"auto_cast_type": "bf16", "tensor_parallel_size": 2},
            {"auto_cast_type": "fp16", "tensor_parallel_size": 2},
        ]

        for compiler_opts in compiler_configs:
            with self.subTest(compiler_opts=compiler_opts):
                try:
                    export_kwargs = {"batch_size": 1, "sequence_length": 128, **compiler_opts}
                    neuron_config = NeuronModelForCausalLM.get_neuron_config(
                        model_name_or_path=model_id, **export_kwargs
                    )
                    model = NeuronModelForCausalLM.export(model_id=model_id, neuron_config=neuron_config)
                    self.assertIsNotNone(
                        model, f"CPU backend export should return model instance for compiler opts {compiler_opts}"
                    )

                except Exception as e:
                    self.fail(f"CPU backend decoder compilation failed for compiler opts {compiler_opts}: {e}")

    def _verify_decoder_artifacts(self, save_dir: Path):
        """Verify that decoder compilation artifacts are created properly."""
        # Check main config file
        config_file = save_dir / "config.json"
        self.assertTrue(config_file.exists(), f"Main config file {config_file} not found")

        # Check for pt files
        pt_files = list(save_dir.glob("*.pt"))
        self.assertGreaterEqual(len(pt_files), 1, f"No .pt files found in {save_dir}")

        # Verify .pt files are not empty
        for pt_file in pt_files:
            self.assertGreater(pt_file.stat().st_size, 0, f"Neuron file {pt_file} is empty")


@requires_neuronx
class NeuronCPUBackendDiffusionTestCase(unittest.TestCase):
    """
    Tests for CPU backend compilation of diffusion models.
    This class tests both the export functionality and integration aspects
    for diffusion models (Stable Diffusion) using cpu_backend=True.
    """

    def setUp(self):
        """Set up test environment for CPU backend diffusion tests."""
        os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "inf2"
        set_seed(SEED)

    @parameterized.expand(CPU_BACKEND_DIFFUSION_MODELS.items())
    @requires_neuronx
    def test_cpu_backend_diffusion_export_and_artifacts(self, model_name, model_id):
        """
        Test CPU backend compilation for diffusion models and verify artifacts creation.
        This test verifies that diffusion models can be compiled with cpu_backend=True
        and that the compilation artifacts are created successfully.
        """
        with TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir) / "test_diffusion_export"
            save_dir.mkdir(parents=True, exist_ok=True)

            try:
                model = StableDiffusionPipeline.from_pretrained(model_id)

                input_shapes = build_stable_diffusion_components_mandatory_shapes(
                    **{"batch_size": 1, "height": 64, "width": 64, "num_images_per_prompt": 1}
                )

                compiler_kwargs = {"auto_cast": "matmul", "auto_cast_type": "bf16"}

                models_and_neuron_configs, output_model_names = get_submodels_and_neuron_configs(
                    model=model,
                    input_shapes=input_shapes,
                    task="text-to-image",
                    library_name="diffusers",
                    output=save_dir,
                    model_name_or_path=model_id,
                )

                _, neuron_outputs = export_models(
                    models_and_neuron_configs=models_and_neuron_configs,
                    task="text-to-image",
                    output_dir=save_dir,
                    output_file_names=output_model_names,
                    compiler_kwargs=compiler_kwargs,
                    cpu_backend=True,
                )

                self.assertIsNotNone(neuron_outputs, "CPU backend diffusion export should return neuron outputs")

                # Verify artifacts were created
                self._verify_diffusion_artifacts(save_dir)

            except Exception as e:
                self.fail(f"CPU backend compilation and artifacts test failed for {model_name} ({model_id}): {e}")

    @requires_neuronx
    def test_cpu_backend_diffusion_with_compiler_options(self):
        """
        Test CPU backend compilation with different compiler options for diffusion models.
        This test verifies that CPU backend compilation works with various
        compiler settings for diffusion models.
        """
        model_id = CPU_BACKEND_DIFFUSION_MODELS["stable-diffusion"]

        # Test different compiler configurations for diffusion
        compiler_configs = [
            {"auto_cast": "matmul", "auto_cast_type": "bf16"},
            {"auto_cast": "all", "auto_cast_type": "fp16"},
            {"auto_cast": None},
        ]

        for compiler_opts in compiler_configs:
            with self.subTest(compiler_opts=compiler_opts):
                with TemporaryDirectory() as tmpdirname:
                    try:
                        model = StableDiffusionPipeline.from_pretrained(model_id)

                        input_shapes = build_stable_diffusion_components_mandatory_shapes(
                            **{"batch_size": 1, "height": 64, "width": 64, "num_images_per_prompt": 1}
                        )

                        models_and_neuron_configs, output_model_names = get_submodels_and_neuron_configs(
                            model=model,
                            input_shapes=input_shapes,
                            task="text-to-image",
                            library_name="diffusers",
                            output=Path(tmpdirname),
                            model_name_or_path=model_id,
                        )

                        _, neuron_outputs = export_models(
                            models_and_neuron_configs=models_and_neuron_configs,
                            task="text-to-image",
                            output_dir=Path(tmpdirname),
                            output_file_names=output_model_names,
                            compiler_kwargs=compiler_opts,
                            cpu_backend=True,
                        )

                        self.assertIsNotNone(neuron_outputs, f"CPU backend export should return neuron outputs for compiler opts {compiler_opts}")

                    except Exception as e:
                        self.fail(f"CPU backend diffusion compilation failed for compiler opts {compiler_opts}: {e}")

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
