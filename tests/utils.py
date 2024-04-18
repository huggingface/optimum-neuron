# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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
"""Various utilities used in multiple tests."""

import contextlib
import functools
import inspect
import os
import random
import string
from typing import Callable, Dict, List, Optional, Tuple, Type

import torch
from datasets import Dataset, DatasetDict
from huggingface_hub import CommitOperationDelete, HfApi, create_repo, delete_repo, get_token, login, logout
from huggingface_hub.utils import RepositoryNotFoundError
from transformers import AutoConfig, PreTrainedModel
from transformers.testing_utils import ENDPOINT_STAGING

from optimum.neuron.distributed import lazy_load_for_parallelism
from optimum.neuron.utils.cache_utils import (
    delete_custom_cache_repo_name_from_hf_home,
    load_custom_cache_repo_name_from_hf_home,
    set_custom_cache_repo_name_in_hf_home,
)
from optimum.neuron.utils.patching import DynamicPatch, Patcher
from optimum.utils import logging
from optimum.utils.testing_utils import TOKEN, USER


logger = logging.get_logger(__name__)

SEED = 42


def get_random_string(length) -> str:
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for _ in range(length))


def create_dummy_dataset(input_specs: Dict[str, Tuple[Tuple[int, ...], torch.dtype]], num_examples: int) -> Dataset:

    def gen():
        for _ in range(num_examples):
            yield {name: torch.rand(shape) for name, shape in input_specs.items()}

    return Dataset.from_generator(gen)


def create_dummy_text_classification_dataset(
    num_train_examples: int, num_eval_examples: int, num_test_examples: Optional[int]
) -> DatasetDict:
    if num_test_examples is None:
        num_test_examples = num_eval_examples

    def create_gen(num_examples, with_labels: bool = True):
        def gen():
            for _ in range(num_examples):
                yield {
                    "sentence": get_random_string(random.randint(64, 256)),
                    "labels": random.randint(0, 1) if with_labels else -1,
                }

        return gen

    ds = DatasetDict()
    ds["train"] = Dataset.from_generator(create_gen(num_train_examples))
    ds["eval"] = Dataset.from_generator(create_gen(num_eval_examples))
    ds["test"] = Dataset.from_generator(create_gen(num_test_examples, with_labels=False))

    return ds


def generate_input_ids(vocab_size: int, batch_size: int, sequence_length: int) -> torch.Tensor:
    return torch.randint(0, vocab_size, (batch_size, sequence_length))


def create_dummy_causal_lm_dataset(
    vocab_size: int,
    num_train_examples: int,
    num_eval_examples: int,
    num_test_examples: Optional[int] = None,
) -> DatasetDict:
    if num_test_examples is None:
        num_test_examples = num_eval_examples

    def create_gen(num_examples):
        def gen():
            for _ in range(num_examples):
                input_ids = generate_input_ids(vocab_size, 1, 32)
                yield {
                    "input_ids": input_ids,
                    "labels": input_ids,
                }

        return gen

    ds = DatasetDict()
    ds["train"] = Dataset.from_generator(create_gen(num_train_examples))
    ds["eval"] = Dataset.from_generator(create_gen(num_eval_examples))
    ds["test"] = Dataset.from_generator(create_gen(num_test_examples))

    return ds


def default_data_collator_for_causal_lm(features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    feature_names = features[0].keys()
    return {k: torch.cat([torch.tensor(feature[k]) for feature in features], dim=0) for k in feature_names}


def static_initializer_seed(initialization_function: Callable, seed: int):
    @functools.wraps(initialization_function)
    def wrapper(*args, **kwargs):
        from transformers import set_seed

        set_seed(seed)
        return initialization_function(*args, **kwargs)

    return wrapper


@contextlib.contextmanager
def create_static_seed_patcher(model_class: Type["PreTrainedModel"], seed: int):
    """
    Context manager that resets the seed to a given value for every initialization function.
    This is useful because lazy initialization works but does not respect the random state of the non-lazy case.
    This allows us to test that lazy initialization works if we ignore the random seed.
    """
    specialized_static_initializer_seed = functools.partial(static_initializer_seed, seed=seed)

    inspect.getmodule(model_class).__name__
    dynamic_patch = DynamicPatch(specialized_static_initializer_seed)
    patcher = Patcher(
        [
            # (fully_qualified_method_name, dynamic_patch),
            ("torch.nn.Embedding.reset_parameters", dynamic_patch),
            ("torch.nn.Linear.reset_parameters", dynamic_patch),
            ("torch.Tensor.normal_", dynamic_patch),
            ("neuronx_distributed.parallel_layers.layers.ColumnParallelLinear.init_weight_cpu", dynamic_patch),
            ("neuronx_distributed.parallel_layers.layers.RowParallelLinear.init_weight_cpu", dynamic_patch),
            (
                "neuronx_distributed.modules.qkv_linear.GQAQKVColumnParallelLinear._init_per_layer_weight",
                dynamic_patch,
            ),
            ("neuronx_distributed.modules.qkv_linear.GQAQKVColumnParallelLinear._init_per_layer_bias", dynamic_patch),
        ]
    )
    with patcher:
        try:
            yield
        finally:
            pass


def get_model(
    model_class: Type["PreTrainedModel"],
    model_name_or_path: str,
    tp_size: int = 1,
    pp_size: int = 1,
    lazy_load: bool = False,
    from_config: bool = False,
    use_static_seed_patcher: bool = False,
    add_random_noise: bool = False,
    config_overwrite: Optional[Dict[str, str]] = None,
) -> "PreTrainedModel":
    if lazy_load:
        ctx = lazy_load_for_parallelism(tensor_parallel_size=tp_size, pipeline_parallel_size=pp_size)
    else:
        ctx = contextlib.nullcontext()
    if use_static_seed_patcher:
        seed_patcher = create_static_seed_patcher(model_class, SEED)
    else:
        seed_patcher = contextlib.nullcontext()
    with ctx:
        with seed_patcher:
            config = AutoConfig.from_pretrained(model_name_or_path)
            if config_overwrite is not None:
                for key, value in config_overwrite.items():
                    attr_type = type(getattr(config, key))
                    setattr(config, key, attr_type(value))
            if from_config:
                model = model_class(config)
            else:
                model = model_class.from_pretrained(model_name_or_path, config=config, ignore_mismatched_sizes=True)

    if getattr(model.config, "problem_type", None) is None:
        model.config.problem_type = "single_label_classification"

    if add_random_noise:
        for param in model.parameters():
            param.data.add_(torch.randn_like(param))

    return model


class TrainiumTestMixin:
    @classmethod
    def setUpClass(cls):
        cls._token = get_token()
        cls._cache_repo = load_custom_cache_repo_name_from_hf_home()
        cls._env = dict(os.environ)

    @classmethod
    def tearDownClass(cls):
        os.environ = cls._env
        if cls._token is not None:
            login(cls._token)
        if cls._cache_repo is not None:
            try:
                set_custom_cache_repo_name_in_hf_home(cls._cache_repo)
            except Exception:
                logger.warning(f"Could not restore the cache repo back to {cls._cache_repo}")
        else:
            delete_custom_cache_repo_name_from_hf_home()


class StagingTestMixin:
    CUSTOM_CACHE_REPO_NAME = "optimum-neuron-cache-testing"
    CUSTOM_CACHE_REPO = f"{USER}/{CUSTOM_CACHE_REPO_NAME}"
    CUSTOM_PRIVATE_CACHE_REPO = f"{CUSTOM_CACHE_REPO}-private"
    _token = ""
    MAX_NUM_LINEARS = 20

    @classmethod
    def set_hf_hub_token(cls, token: Optional[str]) -> Optional[str]:
        orig_token = get_token()
        login(token=token)
        if token is not None:
            login(token=token)
        else:
            logout()
        cls._env = dict(os.environ, HF_ENDPOINT=ENDPOINT_STAGING)
        return orig_token

    @classmethod
    def setUpClass(cls):
        cls._staging_token = TOKEN
        cls._token = cls.set_hf_hub_token(TOKEN)
        cls._custom_cache_repo_name = load_custom_cache_repo_name_from_hf_home()
        delete_custom_cache_repo_name_from_hf_home()

        # Adding a seed to avoid concurrency issues between staging tests.
        cls.seed = get_random_string(5)
        cls.CUSTOM_CACHE_REPO = f"{cls.CUSTOM_CACHE_REPO}-{cls.seed}"
        cls.CUSTOM_PRIVATE_CACHE_REPO = f"{cls.CUSTOM_PRIVATE_CACHE_REPO}-{cls.seed}"

        create_repo(cls.CUSTOM_CACHE_REPO, repo_type="model", exist_ok=True)
        create_repo(cls.CUSTOM_PRIVATE_CACHE_REPO, repo_type="model", exist_ok=True, private=True)

        # We store here which architectures we already used for compiling tiny models.
        cls.visited_num_linears = set()

    @classmethod
    def tearDownClass(cls):
        delete_repo(repo_id=cls.CUSTOM_CACHE_REPO, repo_type="model")
        delete_repo(repo_id=cls.CUSTOM_PRIVATE_CACHE_REPO, repo_type="model")
        if cls._token:
            cls.set_hf_hub_token(cls._token)
        if cls._custom_cache_repo_name:
            try:
                set_custom_cache_repo_name_in_hf_home(cls._custom_cache_repo_name)
            except Exception:
                logger.warning(f"Could not restore the cache repo back to {cls._custom_cache_repo_name}")
            set_custom_cache_repo_name_in_hf_home(cls._custom_cache_repo_name, check_repo=False)

    def remove_all_files_in_repo(self, repo_id: str):
        api = HfApi()
        filenames = api.list_repo_files(repo_id=repo_id)
        operations = [CommitOperationDelete(path_in_repo=filename) for filename in filenames]
        try:
            api.create_commit(
                repo_id=repo_id,
                operations=operations,
                commit_message="Cleanup the repo",
            )
        except RepositoryNotFoundError:
            pass

    def tearDown(self):
        login(TOKEN)
        self.remove_all_files_in_repo(self.CUSTOM_CACHE_REPO)
        self.remove_all_files_in_repo(self.CUSTOM_PRIVATE_CACHE_REPO)
