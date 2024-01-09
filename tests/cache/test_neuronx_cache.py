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
from tempfile import TemporaryDirectory

import pytest
import torch
from huggingface_hub import HfApi
from transformers.testing_utils import ENDPOINT_STAGING

from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.utils import hf_neuronx_cache
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx
from optimum.utils.testing_utils import TOKEN


@pytest.fixture
def cache_repos():
    # Setup
    cache_dir = TemporaryDirectory()
    cache_path = cache_dir.name
    previous_cache_dir = os.environ.get("NEURON_COMPILE_CACHE_URL", None)
    os.environ["NEURON_COMPILE_CACHE_URL"] = cache_path
    api = HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)
    user = api.whoami(token=TOKEN)["name"]
    hostname = socket.gethostname()
    cache_repo_id = f"{user}/{hostname}-optimum-neuron-cache"
    if api.repo_exists(cache_repo_id, token=TOKEN):
        api.delete_repo(cache_repo_id, token=TOKEN)
    cache_repo_id = api.create_repo(cache_repo_id, token=TOKEN).repo_id
    print(api.repo_info(cache_repo_id, token=TOKEN))
    yield (cache_path, cache_repo_id)
    # Teardown
    if previous_cache_dir is None:
        os.environ.pop("NEURON_COMPILE_CACHE_URL")
    else:
        os.environ["NEURON_COMPILE_CACHE_URL"] = previous_cache_dir
    api.delete_repo(cache_repo_id, token=TOKEN)


def export_decoder_model(model_id):
    batch_size = 2
    sequence_length = 512
    num_cores = 1
    auto_cast_type = "fp32"
    return NeuronModelForCausalLM.from_pretrained(
        model_id,
        export=True,
        batch_size=batch_size,
        sequence_length=sequence_length,
        num_cores=num_cores,
        auto_cast_type=auto_cast_type,
    )


def check_decoder_generation(model):
    batch_size = model.config.neuron["batch_size"]
    input_ids = torch.ones((batch_size, 20), dtype=torch.int64)
    with torch.inference_mode():
        sample_output = model.generate(input_ids)
        assert sample_output.shape[0] == batch_size


def assert_local_and_hub_cache_sync(cache_dir, cache_repo_id):
    api = HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)
    remote_files = api.list_repo_files(cache_repo_id, token=TOKEN)
    local_links = glob.glob(f"{cache_dir}/**", recursive=True)
    for link in local_links:
        if os.path.isfile(link):
            path_in_repo = link[len(cache_dir) :].lstrip("/")
            if path_in_repo.startswith("neuron"):
                assert path_in_repo in remote_files


@is_inferentia_test
@requires_neuronx
def test_decoder_cache(cache_repos):
    cache_path, cache_repo_id = cache_repos
    # Export the model a first time to populate both the local and hub caches
    with hf_neuronx_cache(hf_cache_id=cache_repo_id, endpoint=ENDPOINT_STAGING, token=TOKEN):
        model = export_decoder_model("hf-internal-testing/tiny-random-gpt2")
    check_decoder_generation(model)
    assert_local_and_hub_cache_sync(cache_path, cache_repo_id)
    # Clear the local cache
    for root, dirs, files in os.walk(cache_path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    assert len(os.listdir(cache_path)) == 0
    # Export the model again: the compilation artifacts should be fetched from the Hub
    with hf_neuronx_cache(hf_cache_id=cache_repo_id, endpoint=ENDPOINT_STAGING, token=TOKEN):
        model = export_decoder_model("hf-internal-testing/tiny-random-gpt2")
    check_decoder_generation(model)
    # Verify the local cache directory has not been populated
    assert len(os.listdir(cache_path)) == 0


@is_inferentia_test
@requires_neuronx
def test_decoder_cache_wrong_url():
    repo_id = "foo/bar"
    with pytest.raises(ValueError, match=f"The {repo_id} repository does not exist"):
        with hf_neuronx_cache(hf_cache_id=repo_id, endpoint=ENDPOINT_STAGING, token=TOKEN):
            export_decoder_model("hf-internal-testing/tiny-random-gpt2")
