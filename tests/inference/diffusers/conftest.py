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
from tempfile import TemporaryDirectory

import pytest
import torch

from optimum.neuron import (
    NeuronFluxKontextPipeline,
    NeuronFluxPipeline,
    NeuronLatentConsistencyModelPipeline,
    NeuronPixArtAlphaPipeline,
    NeuronStableDiffusionControlNetPipeline,
    NeuronStableDiffusionInstructPix2PixPipeline,
    NeuronStableDiffusionPipeline,
    NeuronStableDiffusionXLPipeline,
)


MODEL_NAMES = {
    "stable-diffusion": "hf-internal-testing/tiny-stable-diffusion-torch",
    "stable-diffusion-ip2p": "asntr/tiny-stable-diffusion-pix2pix-torch",
    "stable-diffusion-xl": "echarlaix/tiny-random-stable-diffusion-xl",
    "latent-consistency": "echarlaix/tiny-random-latent-consistency",
}
LORA_WEIGHTS_TINY = {
    "stable-diffusion": ("Jingya/tiny-stable-diffusion-lora-64", "pytorch_lora_weights.safetensors", "pokemon"),
}
SD_CONTROLNET_ID = "hf-internal-testing/tiny-controlnet"
DEFAULT_STATIC_INPUTS_SHAPES = {"batch_size": 1, "height": 64, "width": 64}
DEFAULT_COMPILER_ARGS = {"auto_cast": "matmul", "auto_cast_type": "bf16"}


# [Stable Diffusion]
@pytest.fixture(scope="module")
def neuron_stable_diffusion_num_img_per_prompt_1_non_dyn_path():
    input_shapes = DEFAULT_STATIC_INPUTS_SHAPES | {"num_images_per_prompt": 1}
    neuron_pipeline = NeuronStableDiffusionPipeline.from_pretrained(
        MODEL_NAMES["stable-diffusion"],
        export=True,
        dynamic_batch_size=False,
        disable_neuron_cache=True,
        **input_shapes,
        **DEFAULT_COMPILER_ARGS,
    )
    model_dir = TemporaryDirectory()
    model_path = model_dir.name
    neuron_pipeline.save_pretrained(model_path)
    del neuron_pipeline
    yield model_path


@pytest.fixture(scope="module")
def neuron_stable_diffusion_num_img_per_prompt_4_non_dyn_path():
    input_shapes = DEFAULT_STATIC_INPUTS_SHAPES | {"num_images_per_prompt": 4}
    neuron_pipeline = NeuronStableDiffusionPipeline.from_pretrained(
        MODEL_NAMES["stable-diffusion"],
        export=True,
        dynamic_batch_size=False,
        disable_neuron_cache=True,
        **input_shapes,
        **DEFAULT_COMPILER_ARGS,
    )
    model_dir = TemporaryDirectory()
    model_path = model_dir.name
    neuron_pipeline.save_pretrained(model_path)
    del neuron_pipeline
    yield model_path


@pytest.fixture(scope="module")
def neuron_stable_diffusion_dyn_path():
    neuron_pipeline = NeuronStableDiffusionPipeline.from_pretrained(
        MODEL_NAMES["stable-diffusion"],
        export=True,
        dynamic_batch_size=True,
        disable_neuron_cache=True,
        **DEFAULT_STATIC_INPUTS_SHAPES,
        **DEFAULT_COMPILER_ARGS,
    )
    model_dir = TemporaryDirectory()
    model_path = model_dir.name
    neuron_pipeline.save_pretrained(model_path)
    del neuron_pipeline
    yield model_path


@pytest.fixture(scope="module")
def neuron_stable_diffusion_ip2p_path():
    neuron_pipeline = NeuronStableDiffusionInstructPix2PixPipeline.from_pretrained(
        MODEL_NAMES["stable-diffusion-ip2p"],
        export=True,
        dynamic_batch_size=True,
        disable_neuron_cache=True,
        **DEFAULT_STATIC_INPUTS_SHAPES,
        **DEFAULT_COMPILER_ARGS,
    )
    model_dir = TemporaryDirectory()
    model_path = model_dir.name
    neuron_pipeline.save_pretrained(model_path)
    del neuron_pipeline
    yield model_path


@pytest.fixture(scope="module")
def neuron_lcm_path():
    neuron_pipeline = NeuronLatentConsistencyModelPipeline.from_pretrained(
        MODEL_NAMES["latent-consistency"],
        export=True,
        dynamic_batch_size=True,
        disable_neuron_cache=True,
        **DEFAULT_STATIC_INPUTS_SHAPES,
        **DEFAULT_COMPILER_ARGS,
    )
    model_dir = TemporaryDirectory()
    model_path = model_dir.name
    neuron_pipeline.save_pretrained(model_path)
    del neuron_pipeline
    yield model_path


@pytest.fixture(scope="module")
def neuron_stable_diffusion_with_fused_lora_path():
    input_shapes = DEFAULT_STATIC_INPUTS_SHAPES | {"num_images_per_prompt": 1}
    lora_params = LORA_WEIGHTS_TINY["stable-diffusion"]
    neuron_pipeline = NeuronStableDiffusionPipeline.from_pretrained(
        MODEL_NAMES["stable-diffusion"],
        export=True,
        dynamic_batch_size=False,
        disable_neuron_cache=True,
        lora_model_ids=lora_params[0],
        lora_weight_names=lora_params[1],
        lora_adapter_names=lora_params[2],
        lora_scales=0.9,
        **input_shapes,
        **DEFAULT_COMPILER_ARGS,
    )
    model_dir = TemporaryDirectory()
    model_path = model_dir.name
    neuron_pipeline.save_pretrained(model_path)
    del neuron_pipeline
    yield model_path


@pytest.fixture(scope="module")
def neuron_stable_diffusion_with_hidden_states_output_path():
    # We need hidden states as output to test compel
    input_shapes = DEFAULT_STATIC_INPUTS_SHAPES | {"num_images_per_prompt": 1}
    neuron_pipeline = NeuronStableDiffusionPipeline.from_pretrained(
        MODEL_NAMES["stable-diffusion"],
        export=True,
        disable_neuron_cache=True,
        inline_weights_to_neff=True,
        output_hidden_states=True,
        **input_shapes,
        **DEFAULT_COMPILER_ARGS,
    )
    model_dir = TemporaryDirectory()
    model_path = model_dir.name
    neuron_pipeline.save_pretrained(model_path)
    del neuron_pipeline
    yield model_path


@pytest.fixture(scope="module")
def neuron_stable_diffusion_single_controlnet_path():
    input_shapes = DEFAULT_STATIC_INPUTS_SHAPES | {"num_images_per_prompt": 1}
    neuron_pipeline = NeuronStableDiffusionControlNetPipeline.from_pretrained(
        MODEL_NAMES["stable-diffusion"],
        controlnet_ids=SD_CONTROLNET_ID,
        export=True,
        **input_shapes,
        **DEFAULT_COMPILER_ARGS,
    )
    model_dir = TemporaryDirectory()
    model_path = model_dir.name
    neuron_pipeline.save_pretrained(model_path)
    del neuron_pipeline
    yield model_path


@pytest.fixture(scope="module")
def neuron_stable_diffusion_multiple_controlnets_path():
    input_shapes = DEFAULT_STATIC_INPUTS_SHAPES | {"num_images_per_prompt": 1}
    neuron_pipeline = NeuronStableDiffusionControlNetPipeline.from_pretrained(
        MODEL_NAMES["stable-diffusion"],
        controlnet_ids=[SD_CONTROLNET_ID, SD_CONTROLNET_ID],
        export=True,
        **input_shapes,
        **DEFAULT_COMPILER_ARGS,
    )
    model_dir = TemporaryDirectory()
    model_path = model_dir.name
    neuron_pipeline.save_pretrained(model_path)
    del neuron_pipeline
    yield model_path


# [Stable Diffusion XL]
@pytest.fixture(scope="module")
def neuron_sdxl_num_img_per_prompt_1_non_dyn_path():
    input_shapes = DEFAULT_STATIC_INPUTS_SHAPES | {"num_images_per_prompt": 1}
    neuron_pipeline = NeuronStableDiffusionXLPipeline.from_pretrained(
        MODEL_NAMES["stable-diffusion-xl"],
        export=True,
        dynamic_batch_size=False,
        disable_neuron_cache=True,
        **input_shapes,
        **DEFAULT_COMPILER_ARGS,
    )
    model_dir = TemporaryDirectory()
    model_path = model_dir.name
    neuron_pipeline.save_pretrained(model_path)
    del neuron_pipeline
    yield model_path


@pytest.fixture(scope="module")
def neuron_sdxl_num_img_per_prompt_4_non_dyn_path():
    input_shapes = DEFAULT_STATIC_INPUTS_SHAPES | {"num_images_per_prompt": 4}
    neuron_pipeline = NeuronStableDiffusionXLPipeline.from_pretrained(
        MODEL_NAMES["stable-diffusion-xl"],
        export=True,
        dynamic_batch_size=False,
        disable_neuron_cache=True,
        **input_shapes,
        **DEFAULT_COMPILER_ARGS,
    )
    model_dir = TemporaryDirectory()
    model_path = model_dir.name
    neuron_pipeline.save_pretrained(model_path)
    del neuron_pipeline
    yield model_path


@pytest.fixture(scope="module")
def neuron_sdxl_dyn_path():
    neuron_pipeline = NeuronStableDiffusionXLPipeline.from_pretrained(
        MODEL_NAMES["stable-diffusion-xl"],
        export=True,
        dynamic_batch_size=True,
        disable_neuron_cache=True,
        **DEFAULT_STATIC_INPUTS_SHAPES,
        **DEFAULT_COMPILER_ARGS,
    )
    model_dir = TemporaryDirectory()
    model_path = model_dir.name
    neuron_pipeline.save_pretrained(model_path)
    del neuron_pipeline
    yield model_path


@pytest.fixture(scope="module")
def neuron_sdxl_with_hidden_states_output_path():
    # We need hidden states as output to test compel
    input_shapes = DEFAULT_STATIC_INPUTS_SHAPES | {"num_images_per_prompt": 1}
    neuron_pipeline = NeuronStableDiffusionXLPipeline.from_pretrained(
        MODEL_NAMES["stable-diffusion-xl"],
        export=True,
        disable_neuron_cache=True,
        inline_weights_to_neff=True,
        output_hidden_states=True,
        **input_shapes,
        **DEFAULT_COMPILER_ARGS,
    )
    model_dir = TemporaryDirectory()
    model_path = model_dir.name
    neuron_pipeline.save_pretrained(model_path)
    del neuron_pipeline
    yield model_path


# [Pixart]
@pytest.fixture(scope="module")
def neuron_pixart_alpha_path():
    compiler_args = {"auto_cast": "none"}
    input_shapes = {"batch_size": 1, "height": 64, "width": 64, "sequence_length": 32}

    neuron_pipeline = NeuronPixArtAlphaPipeline.from_pretrained(
        "hf-internal-testing/tiny-pixart-alpha-pipe",
        export=True,
        torch_dtype=torch.bfloat16,
        dynamic_batch_size=False,
        disable_neuron_cache=True,
        **input_shapes,
        **compiler_args,
    )
    model_dir = TemporaryDirectory()
    model_path = model_dir.name
    neuron_pipeline.save_pretrained(model_path)
    del neuron_pipeline
    yield model_path


# [Flux]
@pytest.fixture(scope="module")
def neuron_flux_tp2_path():
    compiler_args = {"auto_cast": "none"}
    input_shapes = {
        "batch_size": 1,
        "height": 8,
        "width": 8,
        "num_images_per_prompt": 1,
        "sequence_length": 256,
    }

    neuron_pipeline = NeuronFluxPipeline.from_pretrained(
        "hf-internal-testing/tiny-flux-pipe-gated-silu",
        export=True,
        torch_dtype=torch.bfloat16,
        tensor_parallel_size=2,
        dynamic_batch_size=False,
        disable_neuron_cache=True,
        **input_shapes,
        **compiler_args,
    )
    model_dir = TemporaryDirectory()
    model_path = model_dir.name
    neuron_pipeline.save_pretrained(model_path)
    del neuron_pipeline
    yield model_path


@pytest.fixture(scope="module")
def neuron_flux_kontext_tp2_path():
    compiler_args = {"auto_cast": "none"}
    input_shapes = {
        "batch_size": 1,
        "height": 8,
        "width": 8,
        "num_images_per_prompt": 1,
        "sequence_length": 256,
    }

    neuron_pipeline = NeuronFluxKontextPipeline.from_pretrained(
        "hf-internal-testing/tiny-flux-kontext-pipe-gated-silu",
        export=True,
        torch_dtype=torch.bfloat16,
        tensor_parallel_size=2,
        dynamic_batch_size=False,
        disable_neuron_cache=True,
        **input_shapes,
        **compiler_args,
    )
    model_dir = TemporaryDirectory()
    model_path = model_dir.name
    neuron_pipeline.save_pretrained(model_path)
    del neuron_pipeline
    yield model_path
