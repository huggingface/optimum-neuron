import os
import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def init_torch_distributed():
    """Initialize a single-process torch.distributed process group for tests.

    This lets tests that check `torch.distributed.is_initialized()` run without
    launching with `torchrun`. If a launcher already initialized the group,
    this fixture is a no-op.
    """
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed is not available")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    initialized_here = False
    if not torch.distributed.is_initialized():
        try:
            torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
            initialized_here = True
        except Exception:
            # If init fails, tests that require a proper distributed environment
            # will still fail later — we don't want to hide unexpected errors here.
            pass

    # Initialize neuronx distributed model-parallel groups (1:1:1)
    try:
        from neuronx_distributed.parallel_layers.parallel_state import initialize_model_parallel

        # Create trivial model-parallel mesh: TP=1, PP=1, context=1, EP=1
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
            skip_collective_init=True,
            lnc_size=1,
            mesh_only=False,
        )
    except Exception:
        # If neuronx isn't available or initialization fails, allow tests to proceed
        # so that pytest will show the appropriate error. Don't mask failures.
        pass

    yield

    # teardown
    try:
        if initialized_here and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception:
        pass
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
pytest_plugins = [
    "fixtures.llm.vllm_docker_service",
    "fixtures.llm.vllm_service",
    "fixtures.llm.export_models",
]
# ruff: noqa: E402
import os
import random
import shutil
import string
from pathlib import Path

import pytest
from huggingface_hub import HfApi, create_repo, delete_repo, get_token

from optimum.neuron.cache.hub_cache import synchronize_hub_cache
from optimum.neuron.utils.cache_utils import (
    delete_custom_cache_repo_name_from_hf_home,
    load_custom_cache_repo_name_from_hf_home,
    set_custom_cache_repo_name_in_hf_home,
    set_neuron_cache_path,
)
from optimum.neuron.utils.misc import is_precompilation


# Not critical, only usable on the sandboxed CI instance.
USER_STAGING = "__DUMMY_OPTIMUM_USER__"
TOKEN_STAGING = "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"

SEED = 42
OPTIMUM_INTERNAL_TESTING_CACHE_REPO = "optimum-internal-testing/optimum-neuron-cache-for-testing"
OPTIMUM_INTERNAL_TESTING_CACHE_REPO_FOR_CI = "optimum-internal-testing/optimum-neuron-cache-ci"

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


def get_random_string(length) -> str:
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for _ in range(length))


def _hub_test(create_local_cache: bool = False):
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

    set_custom_cache_repo_name_in_hf_home(custom_cache_repo_with_seed)

    if create_local_cache:
        yield (custom_cache_repo_with_seed, local_cache_path_with_seed)
    else:
        yield custom_cache_repo_with_seed

    delete_repo(custom_cache_repo_with_seed, repo_type="model")

    model_repos = HfApi().list_models(author=" optimum-internal-testing-user")
    for repo in model_repos:
        if repo.id.startswith("optimum-neuron-cache-for-testing-"):
            delete_repo(repo.id)

    if local_cache_path_with_seed.is_dir():
        shutil.rmtree(local_cache_path_with_seed)
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


@pytest.fixture(scope="module")
def set_cache_for_ci():
    token = os.environ.get("HF_TOKEN", None)
    if token is None:
        orig_token = get_token()
        if orig_token is None:
            raise ValueError(
                "The token of the `optimum-internal-testing` member on the Hugging Face Hub must be specified via the "
                "HF_TOKEN environment variable."
            )
        else:
            print("Warning: No HF_TOKEN provided. Using the original token.")
    yield

    # This will synchronizee the cache with the cache repo after every test.
    # This is useful to make the CI faster by avoiding recompilation eveyr time.
    if not is_precompilation():
        try:
            synchronize_hub_cache(cache_repo_id=OPTIMUM_INTERNAL_TESTING_CACHE_REPO_FOR_CI, non_blocking=True)
        except Exception as e:
            print(
                f"Warning: Failed to synchronize the cache with the repo {OPTIMUM_INTERNAL_TESTING_CACHE_REPO_FOR_CI}."
            )
            print(f"Error: {e}")


### The following part is for running distributed tests.


# This hook is run before the default pytest_runtest_call
@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item):
    # We want to use our own launching function for distributed tests
    if getattr(item.cls, "is_dist_test", False):
        dist_test_class = item.cls()
        dist_test_class(item._request)
        item.runtest = lambda: True  # Dummy function so test is not run twice


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
