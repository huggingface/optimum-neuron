# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import shutil

import PIL
import torch

from optimum.neuron import (
    NeuronFluxPipeline,
    NeuronStableDiffusionPipeline,
    NeuronStableDiffusionXLPipeline,
)
from optimum.neuron.cache import get_hub_cached_entries, synchronize_hub_cache
from optimum.neuron.utils.testing_utils import requires_neuronx

from ..cache_utils import (
    assert_local_and_hub_cache_sync,
    check_traced_cache_entry,
    get_local_cached_files,
    local_cache_size,
)


def export_flux_model(model_id):
    compiler_args = {"auto_cast": "none"}
    input_shapes = {"batch_size": 1, "height": 8, "width": 8}
    return NeuronFluxPipeline.from_pretrained(
        model_id,
        export=True,
        torch_dtype=torch.bfloat16,
        tensor_parallel_size=2,
        **compiler_args,
        **input_shapes,
    )


def export_stable_diffusion_model(model_id):
    batch_size = 1
    height = 64
    width = 64
    num_images_per_prompt = 1
    return NeuronStableDiffusionPipeline.from_pretrained(
        model_id,
        export=True,
        batch_size=batch_size,
        height=height,
        width=width,
        num_images_per_prompt=num_images_per_prompt,
        inline_weights_to_neff=False,
    )


def export_stable_diffusion_xl_model(model_id):
    batch_size = 1
    height = 64
    width = 64
    num_images_per_prompt = 1
    return NeuronStableDiffusionXLPipeline.from_pretrained(
        model_id,
        export=True,
        batch_size=batch_size,
        height=height,
        width=width,
        num_images_per_prompt=num_images_per_prompt,
        inline_weights_to_neff=False,
    )


def check_stable_diffusion_inference(model):
    prompts = ["sailing ship in storm by Leonardo da Vinci"]
    image = model(prompts, num_images_per_prompt=1).images[0]
    assert isinstance(image, PIL.Image.Image)


@requires_neuronx
def test_stable_diffusion_cache(cache_repos):
    cache_path, cache_repo_id = cache_repos
    model_id = "hf-internal-testing/tiny-stable-diffusion-torch"
    # Export the model a first time to populate the local cache
    model = export_stable_diffusion_model(model_id)
    check_stable_diffusion_inference(model)
    # check registry
    check_traced_cache_entry(cache_path)
    # Synchronize the hub cache with the local cache
    synchronize_hub_cache(cache_repo_id=cache_repo_id)
    assert_local_and_hub_cache_sync(cache_path, cache_repo_id)
    # Verify we are able to fetch the cached entry for the model
    model_entries = get_hub_cached_entries(model_id, cache_repo_id=cache_repo_id)
    assert len(model_entries) == 1
    # Clear the local cache
    for root, dirs, files in os.walk(cache_path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    assert local_cache_size(cache_path) == 0
    # Export the model again: the compilation artifacts should be fetched from the Hub
    model = export_stable_diffusion_model(model_id)
    check_stable_diffusion_inference(model)
    # Verify the local cache directory has not been populated
    assert len(get_local_cached_files(cache_path, ".neuron")) == 0


@requires_neuronx
def test_stable_diffusion_xl_cache(cache_repos):
    cache_path, cache_repo_id = cache_repos
    model_id = "echarlaix/tiny-random-stable-diffusion-xl"
    # Export the model a first time to populate the local cache
    model = export_stable_diffusion_xl_model(model_id)
    check_stable_diffusion_inference(model)
    # check registry
    check_traced_cache_entry(cache_path)
    # Synchronize the hub cache with the local cache
    synchronize_hub_cache(cache_repo_id=cache_repo_id)
    assert_local_and_hub_cache_sync(cache_path, cache_repo_id)
    # Verify we are able to fetch the cached entry for the model
    model_entries = get_hub_cached_entries(model_id, cache_repo_id=cache_repo_id)
    assert len(model_entries) == 1
    # Clear the local cache
    for root, dirs, files in os.walk(cache_path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    assert local_cache_size(cache_path) == 0
    # Export the model again: the compilation artifacts should be fetched from the Hub
    model = export_stable_diffusion_xl_model(model_id)
    check_stable_diffusion_inference(model)
    # Verify the local cache directory has not been populated
    assert len(get_local_cached_files(cache_path, ".neuron")) == 0


def check_flux_inference(model):
    prompts = ["sailing ship in storm by Leonardo da Vinci"]
    image = model(prompts).images[0]
    assert isinstance(image, PIL.Image.Image)


@requires_neuronx
def test_flux_cache(cache_repos):
    cache_path, cache_repo_id = cache_repos
    model_id = "hf-internal-testing/tiny-flux-pipe-gated-silu"
    # Export the model a first time to populate the local cache
    model = export_flux_model(model_id)
    check_flux_inference(model)
    # check registry
    check_traced_cache_entry(cache_path)
    # Synchronize the hub cache with the local cache
    synchronize_hub_cache(cache_repo_id=cache_repo_id)
    assert_local_and_hub_cache_sync(cache_path, cache_repo_id)
    # Verify we are able to fetch the cached entry for the model
    model_entries = get_hub_cached_entries(model_id, cache_repo_id=cache_repo_id)
    assert len(model_entries) == 1
    # Clear the local cache
    for root, dirs, files in os.walk(cache_path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    assert local_cache_size(cache_path) == 0
    # Export the model again: the compilation artifacts should be fetched from the Hub
    model = export_flux_model(model_id)
    check_flux_inference(model)
    # Verify the local cache directory has not been populated
    assert len(get_local_cached_files(cache_path, ".neuron")) == 0
