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
import shutil
import string
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Optional, Set, Tuple, Union

import torch
from datasets import Dataset, DatasetDict
from huggingface_hub import CommitOperationDelete, HfApi, HfFolder, create_repo, delete_repo, login
from huggingface_hub.utils import RepositoryNotFoundError
from transformers import PretrainedConfig, PreTrainedModel

from optimum.neuron.utils.cache_utils import (
    NEURON_COMPILE_CACHE_NAME,
    NeuronHash,
    path_after_folder,
    push_to_cache_on_hub,
    set_neuron_cache_path,
)


# Use that once optimum==1.7.4 is released.
# from optimum.utils.testing_utils import TOKEN, USER

# Used to test the hub
USER = "__DUMMY_OPTIMUM_USER__"

# Not critical, only usable on the sandboxed CI instance.
TOKEN = "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"


MODELS_TO_TEST_MAPPING = {
    "albert": "albert-base-v2",
    "bart": "facebook/bart-base",
    "bert": "bert-base-uncased",
    "camembert": "camembert-base",
    "distilbert": "distilbert-base-uncased",
    "electra": "google/electra-base-discriminator",
    "gpt2": "gpt2",
    "gpt_neo": "EleutherAI/gpt-neo-125M",
    "marian": "Helsinki-NLP/opus-mt-en-ro",
    "roberta": "roberta-base",
    "t5": "t5-small",
    "vit": "google/vit-base-patch16-224-in21k",
    "xlm-roberta": "xlm-roberta-base",
    # TODO: issue with this model for now.
    "m2m_100": "facebook/m2m100_418M",
    # "wav2vec2": "facebook/wav2vec2-base",
    # Remaning: XLNet, Deberta-v2, MPNet, CLIP
}


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


class StagingTestMixin:
    CUSTOM_CACHE_REPO_NAME = "optimum-neuron-cache-testing"
    CUSTOM_CACHE_REPO = f"{USER}/{CUSTOM_CACHE_REPO_NAME}"
    CUSTOM_PRIVATE_CACHE_REPO = f"{CUSTOM_CACHE_REPO}-private"
    _token = ""
    MAX_NUM_LINEARS = 20

    @classmethod
    def set_hf_hub_token(cls, token: str) -> str:
        orig_token = HfFolder.get_token()
        HfFolder.save_token(token)
        return orig_token

    @classmethod
    def setUpClass(cls) -> None:
        cls._token = cls.set_hf_hub_token(TOKEN)
        create_repo(cls.CUSTOM_CACHE_REPO, repo_type="model", exist_ok=True)
        create_repo(cls.CUSTOM_PRIVATE_CACHE_REPO, repo_type="model", exist_ok=True, private=True)

        # We store here which architectures we already used for compiling tiny models.
        cls.visited_num_linears = set()

    @classmethod
    def tearDownClass(cls) -> None:
        delete_repo(repo_id=cls.CUSTOM_CACHE_REPO, repo_type="model")
        delete_repo(repo_id=cls.CUSTOM_PRIVATE_CACHE_REPO, repo_type="model")
        if cls._token:
            cls.set_hf_hub_token(cls._token)

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

    def tearDown(self) -> None:
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

    def push_tiny_pretrained_model_to_hub(
        self, repo_id: str, cache_dir: Optional[Union[str, Path]] = None
    ) -> NeuronHash:
        neuron_hash = None
        orig_repo_id = os.environ.get("CUSTOM_CACHE_REPO", "")
        os.environ["CUSTOM_CACHE_REPO"] = repo_id
        with TemporaryDirectory() as tmpdirname:
            set_neuron_cache_path(tmpdirname)

            input_shapes = (1,)
            data_type = torch.float32
            tiny_model = self.create_and_run_tiny_pretrained_model(random_num_linears=True)
            neuron_hash = NeuronHash(tiny_model, input_shapes, data_type)

            tmp_cache_dir = Path(tmpdirname) / NEURON_COMPILE_CACHE_NAME
            push_to_cache_on_hub(
                neuron_hash,
                tmp_cache_dir,
            )

            if cache_dir is not None:
                for file_or_dir in tmp_cache_dir.iterdir():
                    if file_or_dir.is_file():
                        shutil.copy(file_or_dir, cache_dir / path_after_folder(file_or_dir, NEURON_COMPILE_CACHE_NAME))
                    else:
                        shutil.copytree(
                            file_or_dir, cache_dir / path_after_folder(file_or_dir, NEURON_COMPILE_CACHE_NAME)
                        )
        os.environ["CUSTOM_CACHE_REPO"] = orig_repo_id
        return neuron_hash
