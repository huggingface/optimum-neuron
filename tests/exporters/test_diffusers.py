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
from pathlib import Path
from tempfile import TemporaryDirectory

from optimum.utils import is_diffusers_available
from optimum.utils.testing_utils import require_diffusers
from parameterized import parameterized
from transformers import set_seed
from transformers.testing_utils import require_vision

from optimum.exporters.neuron import (
    build_stable_diffusion_components_mandatory_shapes,
    export_models,
    validate_models_outputs,
)
from optimum.exporters.neuron.__main__ import get_submodels_and_neuron_configs
from optimum.exporters.neuron.model_configs import *  # noqa: F403
from optimum.neuron.utils import LoRAAdapterArguments
from optimum.neuron.utils.testing_utils import requires_neuronx

from .exporters_utils import (
    LORA_WEIGHTS_TINY,
    STABLE_DIFFUSION_MODELS_TINY,
)


if is_diffusers_available():
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

SEED = 42


@requires_neuronx
@require_vision
@require_diffusers
class NeuronStableDiffusionExportTestCase(unittest.TestCase):
    """
    Integration tests ensuring stable diffusion models are correctly exported.
    """

    @parameterized.expand(
        [STABLE_DIFFUSION_MODELS_TINY["stable-diffusion"], STABLE_DIFFUSION_MODELS_TINY["latent-consistency"]]
    )
    def test_export_for_stable_diffusion_models(self, model_id):
        set_seed(SEED)

        # prepare neuron config / models
        model = StableDiffusionPipeline.from_pretrained(model_id)
        input_shapes = build_stable_diffusion_components_mandatory_shapes(
            **{"batch_size": 1, "height": 64, "width": 64, "num_images_per_prompt": 1}
        )
        compiler_kwargs = {"auto_cast": "matmul", "auto_cast_type": "bf16", "instance_type": "inf2"}

        with TemporaryDirectory() as tmpdirname:
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
                compiler_kwargs=compiler_kwargs,
            )
            validate_models_outputs(
                models_and_neuron_configs=models_and_neuron_configs,
                neuron_named_outputs=neuron_outputs,
                output_dir=Path(tmpdirname),
                neuron_files_subpaths=output_model_names,
            )

    @parameterized.expand([STABLE_DIFFUSION_MODELS_TINY["stable-diffusion-xl"]])
    def test_export_for_stable_diffusion_xl_models(self, model_id):
        set_seed(SEED)

        # prepare neuron config / models
        model = StableDiffusionXLPipeline.from_pretrained(model_id)
        input_shapes = build_stable_diffusion_components_mandatory_shapes(
            **{"batch_size": 1, "height": 64, "width": 64, "num_images_per_prompt": 1}
        )
        compiler_kwargs = {"auto_cast": "matmul", "auto_cast_type": "bf16", "instance_type": "inf2"}

        with TemporaryDirectory() as tmpdirname:
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
                compiler_kwargs=compiler_kwargs,
            )
            validate_models_outputs(
                models_and_neuron_configs=models_and_neuron_configs,
                neuron_named_outputs=neuron_outputs,
                output_dir=Path(tmpdirname),
                neuron_files_subpaths=output_model_names,
            )

    def test_export_sd_with_fused_lora_weights(self):
        model_id = STABLE_DIFFUSION_MODELS_TINY["stable-diffusion"]
        lora_params = LORA_WEIGHTS_TINY["stable-diffusion"]
        set_seed(SEED)

        # prepare neuron config / models
        model = StableDiffusionPipeline.from_pretrained(model_id)
        input_shapes = build_stable_diffusion_components_mandatory_shapes(
            **{"batch_size": 1, "height": 64, "width": 64, "num_images_per_prompt": 1}
        )
        lora_args = LoRAAdapterArguments(
            model_ids=lora_params[0],
            weight_names=lora_params[1],
            adapter_names=lora_params[2],
            scales=0.9,
        )
        compiler_kwargs = {"auto_cast": "matmul", "auto_cast_type": "bf16", "instance_type": "inf2"}

        with TemporaryDirectory() as tmpdirname:
            models_and_neuron_configs, output_model_names = get_submodels_and_neuron_configs(
                model=model,
                input_shapes=input_shapes,
                task="text-to-image",
                library_name="diffusers",
                output=Path(tmpdirname),
                model_name_or_path=model_id,
                lora_args=lora_args,
            )
            _, neuron_outputs = export_models(
                models_and_neuron_configs=models_and_neuron_configs,
                task="text-to-image",
                output_dir=Path(tmpdirname),
                output_file_names=output_model_names,
                compiler_kwargs=compiler_kwargs,
            )
            validate_models_outputs(
                models_and_neuron_configs=models_and_neuron_configs,
                neuron_named_outputs=neuron_outputs,
                output_dir=Path(tmpdirname),
                neuron_files_subpaths=output_model_names,
            )
