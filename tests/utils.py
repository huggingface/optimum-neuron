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

import os
import random
import string
from typing import Dict, Optional, Set, Tuple

import torch
from datasets import Dataset, DatasetDict
from huggingface_hub import CommitOperationDelete, HfApi, create_repo, delete_repo, get_token, login, logout
from huggingface_hub.utils import RepositoryNotFoundError
from transformers import PretrainedConfig, PreTrainedModel
from transformers.testing_utils import ENDPOINT_STAGING

from optimum.neuron.utils.cache_utils import (
    delete_custom_cache_repo_name_from_hf_home,
    load_custom_cache_repo_name_from_hf_home,
    set_custom_cache_repo_name_in_hf_home,
)
from optimum.utils import logging
from optimum.utils.testing_utils import TOKEN, USER


logger = logging.get_logger(__name__)


def get_random_string(length) -> str:
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for _ in range(length))


def create_dummy_dataset(input_specs: Dict[str, Tuple[int, ...]], num_examples: int) -> Dataset:
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


class MyTinyModel(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.linears = torch.nn.ModuleList([torch.nn.Linear(1, 1) for _ in range(config.num_linears)])
        self.relu = torch.nn.ReLU()
        self.criterion = torch.nn.MSELoss()

    def forward(self, x, labels=None):
        for lin in self.linears:
            x = lin(x)
            x = self.relu(x)
        if labels is not None:
            loss = self.criterion(x, labels)
            outputs = (loss, x)
        else:
            outputs = (x,)
        return outputs


def create_tiny_pretrained_model(
    num_linears: int = 1,
    random_num_linears: bool = False,
    max_num_linears: int = 20,
    visited_num_linears: Optional[Set[int]] = None,
) -> PreTrainedModel:
    if visited_num_linears is not None:
        if len(visited_num_linears) == max_num_linears:
            raise RuntimeError(
                f"There are too many tests for the maximum number of linears allowed ({max_num_linears}), please "
                "increase it."
            )
    else:
        visited_num_linears = set()

    if random_num_linears:
        num_linears = random.randint(1, max_num_linears)
        while num_linears in visited_num_linears:
            num_linears = random.randint(1, max_num_linears)
        visited_num_linears.add(num_linears)

    config = PretrainedConfig()
    config.num_linears = num_linears

    return MyTinyModel(config)


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

    def create_tiny_pretrained_model(self, num_linears: int = 1, random_num_linears: bool = False):
        return create_tiny_pretrained_model(
            num_linears=num_linears,
            random_num_linears=random_num_linears,
            visited_num_linears=self.visited_num_linears,
        )

    def create_and_run_tiny_pretrained_model(self, num_linears: int = 1, random_num_linears: bool = False):
        tiny_model = self.create_tiny_pretrained_model(num_linears=num_linears, random_num_linears=random_num_linears)
        tiny_model = tiny_model.to("xla")
        random_input = torch.rand(1, device="xla")
        print(tiny_model(random_input))
        return tiny_model
