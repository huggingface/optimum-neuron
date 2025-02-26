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
import os
import shutil
from pathlib import Path

import pytest
from huggingface_hub import HfApi, create_repo, delete_repo, get_token, login, logout

from optimum.neuron.utils.cache_utils import (
    delete_custom_cache_repo_name_from_hf_home,
    load_custom_cache_repo_name_from_hf_home,
    set_custom_cache_repo_name_in_hf_home,
    set_neuron_cache_path,
)

from .utils import OPTIMUM_INTERNAL_TESTING_CACHE_REPO, TOKEN_STAGING, USER_STAGING, get_random_string


# Inferentia fixtures
ENCODER_ARCHITECTURES = [
    "albert",
    "bert",
    "camembert",
    "convbert",
    "distilbert",
    "electra",
    "flaubert",
    "mobilebert",
    "mpnet",
    "roberta",
    "roformer",
    "xlm",
    "roberta",
]
DECODER_ARCHITECTURES = ["gpt2", "llama", "mixtral"]
DIFFUSER_ARCHITECTURES = ["stable-diffusion", "stable-diffusion-xl"]

INFERENTIA_MODEL_NAMES = {
    "albert": "hf-internal-testing/tiny-random-AlbertModel",
    "bert": "hf-internal-testing/tiny-random-BertModel",
    "camembert": "hf-internal-testing/tiny-random-camembert",
    "convbert": "hf-internal-testing/tiny-random-ConvBertModel",
    # "deberta": "hf-internal-testing/tiny-random-DebertaModel",  # Failed for INF1: 'XSoftmax'
    # "deberta-v2": "hf-internal-testing/tiny-random-DebertaV2Model",  # Failed for INF1: 'XSoftmax'
    "distilbert": "hf-internal-testing/tiny-random-DistilBertModel",
    "electra": "hf-internal-testing/tiny-random-ElectraModel",
    "flaubert": "flaubert/flaubert_small_cased",
    "gpt2": "hf-internal-testing/tiny-random-gpt2",
    "llama": "dacorvo/tiny-random-llama",
    "mixtral": "dacorvo/Mixtral-tiny",
    "mobilebert": "hf-internal-testing/tiny-random-MobileBertModel",
    "mpnet": "hf-internal-testing/tiny-random-MPNetModel",
    "roberta": "hf-internal-testing/tiny-random-RobertaModel",
    "roformer": "hf-internal-testing/tiny-random-RoFormerModel",
    "stable-diffusion": "hf-internal-testing/tiny-stable-diffusion-torch",
    "stable-diffusion-xl": "echarlaix/tiny-random-stable-diffusion-xl",
    "xlm": "hf-internal-testing/tiny-random-XLMModel",
    "xlm-roberta": "hf-internal-testing/tiny-xlm-roberta",
}


@pytest.fixture(scope="module", params=[INFERENTIA_MODEL_NAMES[model_arch] for model_arch in ENCODER_ARCHITECTURES])
def inf_encoder_model(request):
    return request.param


@pytest.fixture(scope="module", params=[INFERENTIA_MODEL_NAMES[model_arch] for model_arch in DECODER_ARCHITECTURES])
def inf_decoder_model(request):
    return request.param


@pytest.fixture(scope="module", params=[INFERENTIA_MODEL_NAMES[model_arch] for model_arch in DIFFUSER_ARCHITECTURES])
def inf_diffuser_model(request):
    return request.param


def _hub_test(create_local_cache: bool = False):
    orig_token = get_token()
    orig_custom_cache_repo = load_custom_cache_repo_name_from_hf_home()

    token = os.environ.get("HF_TOKEN", None)
    if token is None:
        raise ValueError(
            "The token of the `optimum-internal-testing` member on the Hugging Face Hub must be specified via the "
            "HF_TOKEN environment variable."
        )

    seed = get_random_string(5)
    custom_cache_repo_with_seed = f"{OPTIMUM_INTERNAL_TESTING_CACHE_REPO}-{seed}"
    create_repo(custom_cache_repo_with_seed, repo_type="model", exist_ok=True)

    local_cache_path_with_seed = Path(f"/var/tmp/neuron-compile-cache-{seed}")
    if create_local_cache:
        set_neuron_cache_path(local_cache_path_with_seed)

    login(token=token)
    set_custom_cache_repo_name_in_hf_home(custom_cache_repo_with_seed)

    if create_local_cache:
        yield (custom_cache_repo_with_seed, local_cache_path_with_seed)
    else:
        yield custom_cache_repo_with_seed

    delete_repo(custom_cache_repo_with_seed, repo_type="model")

    model_repos = HfApi().list_models(author=" optimum-internal-testing-user")
    for repo in model_repos:
        delete_repo(repo.id)

    if local_cache_path_with_seed.is_dir():
        shutil.rmtree(local_cache_path_with_seed)
    if orig_token is not None:
        login(token=orig_token)
    else:
        logout()
    if orig_custom_cache_repo is not None:
        set_custom_cache_repo_name_in_hf_home(orig_custom_cache_repo, check_repo=False)
    else:
        delete_custom_cache_repo_name_from_hf_home()


@pytest.fixture(scope="module")
def hub_test():
    yield from _hub_test()


@pytest.fixture(scope="module")
def hub_test_with_local_cache():
    yield from _hub_test(create_local_cache=True)


### The following part is for running distributed tests.


# This hook is run before the default pytest_runtest_call
@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item):
    # We want to use our own launching function for distributed tests
    if getattr(item.cls, "is_dist_test", False):
        dist_test_class = item.cls()
        dist_test_class(item._request)
        item.runtest = lambda: True  # Dummy function so test is not run twice


# We allow DistributedTest to reuse distributed environments. When the last
# test for a class is run, we want to make sure those distributed environments
# are destroyed.
def pytest_runtest_teardown(item, nextitem):
    if getattr(item.cls, "reuse_dist_env", False) and not nextitem:
        dist_test_class = item.cls()
        for num_procs, pool in dist_test_class._pool_cache.items():
            dist_test_class._close_pool(pool, num_procs, force=True)


@pytest.hookimpl(tryfirst=True)
def pytest_fixture_setup(fixturedef, request):
    if getattr(fixturedef.func, "is_dist_fixture", False):
        dist_fixture_class = fixturedef.func()
        dist_fixture_class(request)


@pytest.fixture
def staging():
    """A pytest fixture only available in huggingface_hub staging mode

    If the huggingface_hub is not operating in staging mode, tests using
    that fixture are automatically skipped.

    Returns:
        a Dict containing a valid staging user and token.
    """
    return {
        "user": USER_STAGING,
        "token": TOKEN_STAGING,
    }
