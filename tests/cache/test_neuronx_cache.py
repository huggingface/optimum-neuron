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
import glob
import os
import shutil
import socket
import subprocess
from tempfile import TemporaryDirectory
from time import time

import PIL
import pytest
import torch
from huggingface_hub import HfApi
from transformers import AutoTokenizer

from optimum.neuron import (
    NeuronModelForCausalLM,
    NeuronModelForSequenceClassification,
    NeuronStableDiffusionPipeline,
    NeuronStableDiffusionXLPipeline,
)
from optimum.neuron.cache import get_hub_cached_entries, get_hub_cached_models, synchronize_hub_cache
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


@pytest.fixture
def cache_repos():
    # Setup: create temporary Hub repository and local cache directory
    api = HfApi()
    hostname = socket.gethostname()
    cache_repo_id = f"{hostname}-{time()}-optimum-neuron-cache"
    cache_repo_id = api.create_repo(cache_repo_id, private=True).repo_id
    cache_dir = TemporaryDirectory()
    cache_path = cache_dir.name
    # Modify environment to force neuronx cache to use temporary caches
    previous_env = {}
    env_vars = ["NEURON_COMPILE_CACHE_URL", "CUSTOM_CACHE_REPO"]
    for var in env_vars:
        previous_env[var] = os.environ.get(var)
    os.environ["NEURON_COMPILE_CACHE_URL"] = cache_path
    os.environ["CUSTOM_CACHE_REPO"] = cache_repo_id
    try:
        yield (cache_path, cache_repo_id)
    finally:
        # Teardown
        api.delete_repo(cache_repo_id)
        for var in env_vars:
            if previous_env[var] is None:
                os.environ.pop(var)
            else:
                os.environ[var] = previous_env[var]


def export_decoder_model(model_id):
    batch_size = 2
    sequence_length = 512
    num_cores = 2
    auto_cast_type = "bf16"
    return NeuronModelForCausalLM.from_pretrained(
        model_id,
        export=True,
        batch_size=batch_size,
        sequence_length=sequence_length,
        num_cores=num_cores,
        auto_cast_type=auto_cast_type,
    )


def export_encoder_model(model_id):
    batch_size = 1
    sequence_length = 64
    return NeuronModelForSequenceClassification.from_pretrained(
        model_id,
        export=True,
        dynamic_batch_size=False,
        batch_size=batch_size,
        sequence_length=sequence_length,
        inline_weights_to_neff=False,
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
        data_parallel_mode="none",  # TODO: Remove when weights separated makesits way to a neuron sdk release.
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
        data_parallel_mode="none",  # TODO: Remove when weights separated makesits way to a neuron sdk release.
    )


def check_decoder_generation(model):
    batch_size = model.neuron_config.batch_size
    input_ids = torch.ones((batch_size, 20), dtype=torch.int64)
    with torch.inference_mode():
        sample_output = model.generate(input_ids)
        assert sample_output.shape[0] == batch_size


def check_encoder_inference(model, tokenizer):
    text = ["This is a sample output"]
    tokens = tokenizer(text, return_tensors="pt")
    outputs = model(**tokens)
    assert "logits" in outputs


def check_stable_diffusion_inference(model):
    prompts = ["sailing ship in storm by Leonardo da Vinci"]
    image = model(prompts, num_images_per_prompt=1).images[0]
    assert isinstance(image, PIL.Image.Image)


def get_local_cached_files(cache_path, extension="*"):
    links = glob.glob(f"{cache_path}/**/*/*.{extension}", recursive=True)
    return [link for link in links if os.path.isfile(link)]


def check_decoder_cache_entry(model, cache_path):
    local_files = get_local_cached_files(cache_path, "json")
    model_id = model.neuron_config.checkpoint_id
    model_configurations = [path for path in local_files if model_id in path]
    assert len(model_configurations) > 0


def check_traced_cache_entry(cache_path):
    local_files = get_local_cached_files(cache_path, "json")
    registry_path = [path for path in local_files if "REGISTRY" in path][0]
    registry_key = registry_path.split("/")[-1].replace(".json", "")
    local_files.remove(registry_path)
    local_hash_keys = []
    for local_file in local_files:
        local_hash_keys.append(local_file.split("/")[-2].replace("MODULE_", ""))
    assert registry_key in local_hash_keys


def assert_local_and_hub_cache_sync(cache_path, cache_repo_id):
    api = HfApi()
    remote_files = api.list_repo_files(cache_repo_id)
    local_files = get_local_cached_files(cache_path)
    for file in local_files:
        assert os.path.isfile(file)
        path_in_repo = file[len(cache_path) :].lstrip("/")
        assert path_in_repo in remote_files


def local_cache_size(cache_path):
    return len(get_local_cached_files(cache_path))


@is_inferentia_test
@requires_neuronx
def test_decoder_cache(cache_repos):
    cache_path, cache_repo_id = cache_repos
    model_id = "llamafactory/tiny-random-Llama-3"
    # Export the model a first time to populate the local cache
    model = export_decoder_model(model_id)
    check_decoder_generation(model)
    check_decoder_cache_entry(model, cache_path)
    # Synchronize the hub cache with the local cache
    synchronize_hub_cache(cache_repo_id=cache_repo_id)
    assert_local_and_hub_cache_sync(cache_path, cache_repo_id)
    # Verify we are able to fetch the cached entry for the model
    model_entries = get_hub_cached_entries(model_id, cache_repo_id=cache_repo_id)
    assert len(model_entries) == 1
    assert model_entries[0] == model.neuron_config.to_dict()
    # Also verify that the model appears in the list of cached models
    cached_models = get_hub_cached_models()
    assert ("llama", "llamafactory", "tiny-random-Llama-3") in cached_models
    # Clear the local cache
    for root, dirs, files in os.walk(cache_path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    assert local_cache_size(cache_path) == 0
    # Export the model again: the compilation artifacts should be fetched from the Hub
    model = export_decoder_model(model_id)
    check_decoder_generation(model)
    # Verify the local cache directory has not been populated
    assert len(get_local_cached_files(cache_path, "neff")) == 0


@is_inferentia_test
@requires_neuronx
def test_encoder_cache(cache_repos):
    cache_path, cache_repo_id = cache_repos
    model_id = "hf-internal-testing/tiny-random-BertModel"
    # Export the model a first time to populate the local cache
    model = export_encoder_model(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    check_encoder_inference(model, tokenizer)
    # check registry
    check_traced_cache_entry(cache_path)
    # Synchronize the hub cache with the local cache
    synchronize_hub_cache(cache_repo_id=cache_repo_id)
    assert_local_and_hub_cache_sync(cache_path, cache_repo_id)
    # Verify we are able to fetch the cached entry for the model
    model_entries = get_hub_cached_entries(model_id, task="text-classification", cache_repo_id=cache_repo_id)
    assert len(model_entries) == 1
    # Clear the local cache
    for root, dirs, files in os.walk(cache_path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    assert local_cache_size(cache_path) == 0
    # Export the model again: the compilation artifacts should be fetched from the Hub
    model = export_encoder_model(model_id)
    check_encoder_inference(model, tokenizer)
    # Verify the local cache directory has not been populated
    assert len(get_local_cached_files(cache_path, ".neuron")) == 0


@is_inferentia_test
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


@is_inferentia_test
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


@is_inferentia_test
@requires_neuronx
@pytest.mark.parametrize(
    "var, value, match",
    [
        ("CUSTOM_CACHE_REPO", "foo/bar", "The foo/bar repository does not exist"),
        ("HF_ENDPOINT", "https://foo.bar.baz", "Name or service not known"),
        ("HF_TOKEN", "foo", "401 Client Error: Unauthorized"),
    ],
    ids=["invalid_repo", "invalid_endpoint", "invalid_token"],
)
def test_decoder_cache_unavailable(cache_repos, var, value, match):
    # Just exporting the model will only emit a warning
    export_decoder_model("llamafactory/tiny-random-Llama-3")
    prev_value = os.environ.get(var, None)
    # Modify the specified environment variable to trigger an error
    os.environ[var] = value
    try:
        with pytest.raises(ValueError, match=match):
            synchronize_hub_cache()
    finally:
        if prev_value is None:
            os.environ.pop(var)
        else:
            os.environ[var] = prev_value


@is_inferentia_test
@requires_neuronx
def test_optimum_neuron_cli_cache_synchronize(cache_repos):
    cache_path, cache_repo_id = cache_repos
    model_id = "llamafactory/tiny-random-Llama-3"
    # Export a model to populate the local cache
    export_decoder_model(model_id)
    # Synchronize the hub cache with the local cache
    command = "optimum-cli neuron cache synchronize".split()
    p = subprocess.Popen(command, stdout=subprocess.PIPE)
    p.communicate()
    assert p.returncode == 0
    assert_local_and_hub_cache_sync(cache_path, cache_repo_id)
    # Check the model entry in the hub
    command = f"optimum-cli neuron cache lookup {model_id}".split()
    p = subprocess.Popen(command, stdout=subprocess.PIPE)
    stdout, _ = p.communicate()
    stdout = stdout.decode("utf-8")
    assert p.returncode == 0
    assert f"1 entrie(s) found in cache for {model_id}" in stdout
