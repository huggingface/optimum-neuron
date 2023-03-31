# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Tests for the cache utilities."""

import os
import random
import string
from dataclasses import FrozenInstanceError
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List
from unittest import TestCase

import torch
from huggingface_hub import CommitOperationDelete, HfApi, HfFolder, create_repo, delete_repo, login
from transformers import BertConfig, BertModel, PretrainedConfig, PreTrainedModel, set_seed
from transformers.testing_utils import is_staging_test

from optimum.neuron.utils.cache_utils import (
    NEURON_COMPILE_CACHE_NAME,
    NeuronHash,
    get_neuron_cache_path,
    get_num_neuron_cores_used,
    list_files_in_neuron_cache,
    push_to_cache_on_hub,
    set_neuron_cache_path,
)
from optimum.neuron.utils.version_utils import get_neuronxcc_version
from optimum.utils.testing_utils import TOKEN, USER


def get_random_string(length) -> str:
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))


class NeuronUtilsTestCase(TestCase):
    def test_get_neuron_cache_path(self):
        os.environ["NEURON_CC_FLAGS"] = "--some --parameters --here --no-cache --other --paremeters --here"
        assert get_neuron_cache_path() is None

        custom_cache_dir_name = Path("_this/is_/my1/2custom/cache/dir")
        os.environ[
            "NEURON_CC_FLAGS"
        ] = f"--some --parameters --here --cache_dir={custom_cache_dir_name} --other --paremeters --here"
        self.assertEqual(get_neuron_cache_path(), custom_cache_dir_name / NEURON_COMPILE_CACHE_NAME)

        os.environ["NEURON_CC_FLAGS"] = "--some --parameters --here --other --paremeters --here"
        self.assertEqual(get_neuron_cache_path(), Path("/var/tmp") / NEURON_COMPILE_CACHE_NAME)

    def _test_set_neuron_cache_path(self, new_cache_path):
        os.environ["NEURON_CC_FLAGS"] = "--some --parameters --here --no-cache --other --paremeters --here"
        with self.assertRaisesRegex(ValueError, expected_regex=r"Cannot set the neuron compile cache"):
            set_neuron_cache_path(new_cache_path)
        set_neuron_cache_path(new_cache_path, ignore_no_cache=True)
        self.assertEqual(get_neuron_cache_path(), Path(new_cache_path) / NEURON_COMPILE_CACHE_NAME)

        os.environ[
            "NEURON_CC_FLAGS"
        ] = "--some --parameters --here --cache_dir=original_cache_dir --other --paremeters"
        set_neuron_cache_path(new_cache_path)
        self.assertEqual(get_neuron_cache_path(), Path(new_cache_path) / NEURON_COMPILE_CACHE_NAME)

    def test_set_neuron_cache_path(self):
        new_cache_path_str = "path/to/my/custom/cache"
        new_cache_path_path = Path(new_cache_path_str)
        self._test_set_neuron_cache_path(new_cache_path_str)
        self._test_set_neuron_cache_path(new_cache_path_path)

    def test_get_num_neuron_cores_used(self):
        self.assertEqual(get_num_neuron_cores_used(), 1)

        randon_num_cores = random.randint(1, 32)
        os.environ["LOCAL_WORLD_SIZE"] = str(randon_num_cores)
        self.assertEqual(get_num_neuron_cores_used(), randon_num_cores)

    def _create_random_neuron_cache(
        self, directory: Path, number_of_right_cache_files: int = 32, return_only_relevant_files: bool = False
    ) -> List[Path]:
        wrong_extensions = [get_random_string(3) for _ in range(4)]
        right_extensions = ["neff", "pb", "txt"]

        filenames = []

        def create_random_nested_directories(number_of_dirs: int) -> Path:
            p = directory
            for _ in range(number_of_dirs):
                p = p / get_random_string(5)
            p.mkdir(parents=True, exist_ok=True)
            return p

        for _ in range(number_of_right_cache_files):
            number_of_dirs = random.randint(0, 5)
            path = create_random_nested_directories(number_of_dirs)

            wrong_extension = random.choice(wrong_extensions)
            right_extension = random.choice(right_extensions)

            wrong_extension_file = path / f"{get_random_string(6)}.{wrong_extension}"
            wrong_extension_file.touch(exist_ok=True)
            if not return_only_relevant_files:
                filenames.append(wrong_extension_file)
            right_extension_file = path / f"{get_random_string(6)}.{right_extension}"
            right_extension_file.touch(exist_ok=True)
            filenames.append(right_extension_file)

        return filenames

    def test_list_files_in_neuron_cache(self):
        with TemporaryDirectory() as tmpdirname:
            filenames = self._create_random_neuron_cache(Path(tmpdirname), return_only_relevant_files=False)
            self.assertSetEqual(set(filenames), set(list_files_in_neuron_cache(Path(tmpdirname))))

        with TemporaryDirectory() as tmpdirname:
            filenames = self._create_random_neuron_cache(Path(tmpdirname), return_only_relevant_files=True)
            self.assertSetEqual(
                set(filenames), set(list_files_in_neuron_cache(Path(tmpdirname), only_relevant_files=True))
            )


class NeuronHashTestCase(TestCase):
    def test_neuron_hash_is_not_mutable(self):
        bert_model = BertModel(BertConfig())
        neuron_hash = NeuronHash(bert_model, ((4, 12), (4, 12)), torch.float32, 2)

        with self.assertRaises(FrozenInstanceError):
            neuron_hash.model = bert_model

        with self.assertRaises(FrozenInstanceError):
            neuron_hash.input_shapes = ((2, 32), (2, 32))

        with self.assertRaises(FrozenInstanceError):
            neuron_hash.num_neuron_cores = 32

    def _test_neuron_hash(
        self,
        model_a,
        input_shapes_a,
        dtype_a,
        num_neuron_cores_a,
        model_b,
        input_shapes_b,
        dtype_b,
        num_neuron_cores_b,
        should_be_equal,
    ):
        neuron_hash_a = NeuronHash(model_a, input_shapes_a, dtype_a, num_neuron_cores=num_neuron_cores_a)
        neuron_hash_b = NeuronHash(model_b, input_shapes_b, dtype_b, num_neuron_cores=num_neuron_cores_b)
        if should_be_equal:
            self.assertEqual(neuron_hash_a.compute_hash(), neuron_hash_b.compute_hash())
        else:
            self.assertNotEqual(neuron_hash_a.compute_hash(), neuron_hash_b.compute_hash())

    def test_computed_hash_is_same_for_same_models(self):
        set_seed(42)
        bert_model = BertModel(BertConfig())
        set_seed(42)
        same_bert_model = BertModel(BertConfig())

        return self._test_neuron_hash(
            bert_model,
            ((1, 2), (2, 3)),
            torch.bfloat16,
            19,
            same_bert_model,
            ((1, 2), (2, 3)),
            torch.bfloat16,
            19,
            True,
        )

    def test_computed_hash_is_different_for_different_models(self):
        set_seed(42)
        bert_model = BertModel(BertConfig())
        set_seed(38)
        different_bert_model = BertModel(BertConfig())

        return self._test_neuron_hash(
            bert_model,
            ((1, 2), (2, 3)),
            torch.bfloat16,
            19,
            different_bert_model,
            ((1, 2), (2, 3)),
            torch.bfloat16,
            19,
            False,
        )

    def test_computed_hash_is_different_for_different_parameters_but_same_model(self):
        bert_model = BertModel(BertConfig())
        parameters = [[((1, 2), (2, 3)), ((2, 3), (3, 4))], [torch.float32, torch.float16], [32, 2]]
        params_a = [p[0] for p in parameters]
        for i in range(len(parameters)):
            params_b = [p[int(i == j)] for j, p in enumerate(parameters)]
            self._test_neuron_hash(bert_model, *params_a, bert_model, *params_b, False)

    def test_neuron_hash_folders(self):
        bert_model = BertModel(BertConfig())
        input_shapes = ((1, 2), (2, 3))
        data_type = torch.float32
        num_neuron_cores = 32

        neuron_hash = NeuronHash(
            bert_model,
            input_shapes,
            data_type,
            num_neuron_cores=num_neuron_cores,
            neuron_compiler_version="dummy_version",
        )
        hashes = neuron_hash.compute_hash()
        expected_folders = ["dummy_version", "bert"] + list(hashes)
        self.assertListEqual(neuron_hash.folders, expected_folders)

        neuron_hash = NeuronHash(bert_model, input_shapes, data_type, num_neuron_cores=num_neuron_cores)
        hashes = neuron_hash.compute_hash()
        expected_folders = [get_neuronxcc_version(), "bert"] + list(hashes)
        self.assertListEqual(neuron_hash.folders, expected_folders)

    def test_neuron_hash_is_private(self):
        input_shapes = ((1, 2), (2, 3))
        data_type = torch.float32

        bert_model = BertModel(BertConfig())
        neuron_hash = NeuronHash(bert_model, input_shapes, data_type)
        self.assertTrue(neuron_hash.is_private)

        bert_model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")
        neuron_hash = NeuronHash(bert_model, input_shapes, data_type)

        self.assertFalse(neuron_hash.is_private)

        with TemporaryDirectory() as tmpdirname:
            bert_model.save_pretrained(tmpdirname)
            local_bert_model = BertModel.from_pretrained(tmpdirname)
            neuron_hash = NeuronHash(local_bert_model, input_shapes, data_type)
            self.assertTrue(neuron_hash.is_private)


@is_staging_test
class CachedModelOnTheHubTestCase(TestCase):
    CUSTOM_CACHE_REPO_NAME = "optimum-neuron-cache-testing"
    CUSTOM_CACHE_REPO = f"{USER}/{CUSTOM_CACHE_REPO_NAME}"
    CUSTOM_PRIVATE_CACHE_REPO = f"{CUSTOM_CACHE_REPO}-private"
    _token = ""
    MAX_NUM_LINEARS = 20

    @classmethod
    def setUpClass(cls) -> None:
        cls._token = HfFolder.get_token()
        login(TOKEN)
        HfFolder.save_token(TOKEN)
        create_repo(cls.CUSTOM_CACHE_REPO, repo_type="model", exist_ok=True)
        create_repo(cls.CUSTOM_PRIVATE_CACHE_REPO, repo_type="model", exist_ok=True, private=True)

    @classmethod
    def tearDownClass(cls) -> None:
        login(cls._token)
        HfFolder.save_token(cls._token)
        delete_repo(repo_id=cls.CUSTOM_CACHE_REPO, repo_type="model")
        delete_repo(repo_id=cls.CUSTOM_PRIVATE_CACHE_REPO, repo_type="model")

    def remove_all_files_in_repo(self, repo_id: str):
        api = HfApi()
        filenames = api.list_repo_files(repo_id=repo_id)
        operations = [CommitOperationDelete(path_in_repo=filename) for filename in filenames]
        api.create_commit(
            repo_id=repo_id,
            operations=operations,
            commit_message="Cleanup the repo",
        )

    def tearDown(self) -> None:
        self.remove_all_files_in_repo(self.CUSTOM_CACHE_REPO)
        self.remove_all_files_in_repo(self.CUSTOM_PRIVATE_CACHE_REPO)

    def _create_tiny_pretrained_model(self, num_linears: int = 1, random_num_linears: bool = False):
        # input_sizes = getattr(self, "_input_sizes", {input_size})
        # if len(input_sizes) == self.MAX_INPUT_SIZE_ALLOWED:
        #     raise RuntimeError(f"There are too many tests for the maximum input size allowed, please increase it.")

        # if random_input_size:
        #     while input_size in input_sizes:
        #         input_size = random.randint(1, self.MAX_INPUT_SIZE_ALLOWED)

        visited_num_linears = getattr(self, "_num_linears", {num_linears})
        if len(visited_num_linears) == self.MAX_NUM_LINEARS:
            raise RuntimeError(
                f"There are too many tests for the maximum number of linears allowed ({self.MAX_NUM_LINEARS}), please "
                "increase it."
            )
        if random_num_linears:
            num_linears = random.randint(1, self.MAX_NUM_LINEARS)

        class MyTinyModel(PreTrainedModel):
            def __init__(self):
                config = PretrainedConfig()
                super().__init__(config)
                self.linears = torch.nn.ModuleList([torch.nn.Linear(1, 1) for _ in range(num_linears)])
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                for lin in self.linears:
                    x = lin(x)
                    x = self.relu(x)
                return x

        return MyTinyModel()

    def _create_and_run_tiny_pretrained_model(self, num_linears: int = 1, random_num_linears: bool = False):
        tiny_model = self._create_tiny_pretrained_model(num_linears=num_linears, random_num_linears=random_num_linears)
        tiny_model = tiny_model.to("xla")
        random_input = torch.rand(1, device="xla")
        tiny_model(random_input)
        return tiny_model

    def test_push_to_hub_fails_with_private_model_and_public_repo(self):
        with TemporaryDirectory() as tmpdirname:
            set_neuron_cache_path(tmpdirname)

            input_shapes = (1,)
            data_type = torch.float32
            tiny_model = self._create_and_run_tiny_pretrained_model(random_num_linears=True)
            neuron_hash = NeuronHash(tiny_model, input_shapes, data_type)

            cached_files = list_files_in_neuron_cache(Path(tmpdirname) / NEURON_COMPILE_CACHE_NAME)

            # The model being loaded locally is assumed to be private, push to hub should prevent from pushing to a
            # public repo.
            with self.assertRaisesRegex(ValueError, "Cannot push the cached model"):
                push_to_cache_on_hub(neuron_hash, cached_files[0], self.CUSTOM_CACHE_REPO)

            # It should work when using a private repo.
            cached_model_on_the_hub = push_to_cache_on_hub(
                neuron_hash, cached_files[0], self.CUSTOM_PRIVATE_CACHE_REPO
            )
            self.assertIsNotNone(cached_model_on_the_hub)

    def test_push_to_hub_without_specifying_a_cache_repo_id(self):
        with TemporaryDirectory() as tmpdirname:
            set_neuron_cache_path(tmpdirname)

            input_shapes = (1,)
            data_type = torch.float32
            tiny_model = self._create_and_run_tiny_pretrained_model(random_num_linears=True)
            neuron_hash = NeuronHash(tiny_model, input_shapes, data_type)

            cached_files = list_files_in_neuron_cache(Path(tmpdirname) / NEURON_COMPILE_CACHE_NAME)

            os.environ["CUSTOM_CACHE_REPO"] = self.CUSTOM_PRIVATE_CACHE_REPO
            push_to_cache_on_hub(neuron_hash, cached_files[0])

    def test_push_to_hub_overwrite_existing(self):
        with TemporaryDirectory() as tmpdirname:
            set_neuron_cache_path(tmpdirname)

            input_shapes = (1,)
            data_type = torch.float32
            tiny_model = self._create_and_run_tiny_pretrained_model(random_num_linears=True)
            neuron_hash = NeuronHash(tiny_model, input_shapes, data_type)

            cache_dir = Path(tmpdirname) / NEURON_COMPILE_CACHE_NAME
            cached_files = list_files_in_neuron_cache(cache_dir)

            push_to_cache_on_hub(neuron_hash, cached_files[0], self.CUSTOM_PRIVATE_CACHE_REPO)

            # With a file
            with self.assertLogs("optimum", level="INFO") as cm:
                push_to_cache_on_hub(neuron_hash, cached_files[0], self.CUSTOM_PRIVATE_CACHE_REPO)
                self.assertIn(str(cm.output), "Did not push the cached model located at")

            with self.assertLogs("optimum", level="WARNING") as cm:
                push_to_cache_on_hub(
                    neuron_hash, cached_files[0], self.CUSTOM_PRIVATE_CACHE_REPO, overwrite_existing=True
                )
                self.assertIn(
                    str(cm.output), "Overwriting the already existing cached model on the Hub by the one located at"
                )

            # Because the directory contains cached_files[0], we cleanup before testing.
            self.remove_all_files_in_repo(self.CUSTOM_PRIVATE_CACHE_REPO)

            # With a directory
            with self.assertLogs("optimum", level="INFO") as cm:
                push_to_cache_on_hub(neuron_hash, cache_dir, self.CUSTOM_PRIVATE_CACHE_REPO)
                self.assertIn(cm.output, "Did not push the cached model located at")

            with self.assertLogs("optimum", level="WARNING") as cm:
                push_to_cache_on_hub(neuron_hash, cache_dir, self.CUSTOM_PRIVATE_CACHE_REPO, overwrite_existing=True)
                self.assertIn(
                    cm.output, "Overwriting the already existing cached model on the Hub by the one located at"
                )

    def test_push_to_hub_local_path_in_repo(self):
        with TemporaryDirectory() as tmpdirname:
            set_neuron_cache_path(tmpdirname)

            input_shapes = (1,)
            data_type = torch.float32
            tiny_model = self._create_and_run_tiny_pretrained_model(random_num_linears=True)
            neuron_hash = NeuronHash(tiny_model, input_shapes, data_type)

            cache_dir = Path(tmpdirname) / NEURON_COMPILE_CACHE_NAME
            cached_files = list_files_in_neuron_cache(cache_dir)

            def local_path_to_path_in_repo(path):
                return Path("my/awesome/new/path") / path.name

            # With a file
            push_to_cache_on_hub(
                neuron_hash,
                cached_files[0],
                self.CUSTOM_PRIVATE_CACHE_REPO,
                local_path_to_path_in_repo=local_path_to_path_in_repo,
            )
            files_in_repo = HfApi().list_repo_files(repo_id=self.CUSTOM_PRIVATE_CACHE_REPO)
            path_in_repo = f"{neuron_hash.cache_path}/my/awesome/new/path/{cached_files[0].name}"
            self.assertIn(path_in_repo, files_in_repo)

            # Because the directory contains cached_files[0], we cleanup before testing.
            # self.remove_all_files_in_repo(self.CUSTOM_PRIVATE_CACHE_REPO)

            def another_local_path_to_path_in_repo(path):
                return Path("my/another/awesome/new/path") / path.name

            # With a directory
            push_to_cache_on_hub(
                neuron_hash,
                cache_dir,
                self.CUSTOM_PRIVATE_CACHE_REPO,
                local_path_to_path_in_repo=another_local_path_to_path_in_repo,
            )
            files_in_repo = HfApi().list_repo_files(repo_id=self.CUSTOM_PRIVATE_CACHE_REPO)
            for filename in cache_dir.glob("**/*"):
                if filename.is_file():
                    path_in_repo = f"{neuron_hash.cache_path}/my/another/awesome/new/path/{filename.name}"
                    self.assertIn(path_in_repo, files_in_repo)
