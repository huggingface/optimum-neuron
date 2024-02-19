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
# limitations under the License.
"""Tests for the cache utilities."""

import json
import logging
import os
import random
from dataclasses import FrozenInstanceError
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List
from unittest import TestCase

import huggingface_hub
import torch
from huggingface_hub import HfApi, create_repo, delete_repo, get_token, hf_hub_download, login
from transformers import BertConfig, BertModel, set_seed
from transformers.testing_utils import TOKEN as TRANSFORMERS_TOKEN
from transformers.testing_utils import USER as TRANSFORMERS_USER
from transformers.testing_utils import is_staging_test

from optimum.neuron.utils.cache_utils import (
    CACHE_REPO_FILENAME,
    REGISTRY_FILENAME,
    NeuronHash,
    _list_in_registry_dict,
    add_in_registry,
    create_registry_file_if_does_not_exist,
    download_cached_model_from_hub,
    get_cached_model_on_the_hub,
    get_neuron_cache_path,
    get_num_neuron_cores_used,
    has_write_access_to_repo,
    list_files_in_neuron_cache,
    list_in_registry,
    load_custom_cache_repo_name_from_hf_home,
    path_after_folder,
    push_to_cache_on_hub,
    remove_ip_adress_from_path,
    set_custom_cache_repo_name_in_hf_home,
    set_neuron_cache_path,
)
from optimum.neuron.utils.testing_utils import is_trainium_test
from optimum.utils.testing_utils import TOKEN, USER

from .utils import MyTinyModel, StagingTestMixin, TrainiumTestMixin, get_random_string


DUMMY_COMPILER_VERSION = "1.2.3"


@is_trainium_test
class NeuronUtilsTestCase(TrainiumTestMixin, TestCase):
    def tearDown(self):
        # Cleaning the Neuron compiler flags to avoid breaking other tests.
        os.environ["NEURON_CC_FLAGS"] = ""

    def test_load_custom_cache_repo_name_from_hf_home(self):
        with TemporaryDirectory() as tmpdirname:
            hf_home_cache_repo_file = f"{tmpdirname}/{CACHE_REPO_FILENAME}"
            with open(hf_home_cache_repo_file, "w") as fp:
                fp.write("blablabla")
            self.assertEqual(load_custom_cache_repo_name_from_hf_home(hf_home_cache_repo_file), "blablabla")

            bad_hf_home_cache_repo_file = f"{tmpdirname}/does_not_exist"
            self.assertIsNone(load_custom_cache_repo_name_from_hf_home(bad_hf_home_cache_repo_file))

    def test_get_neuron_cache_path(self):
        os.environ["NEURON_CC_FLAGS"] = "--some --parameters --here --no-cache --other --paremeters --here"
        assert get_neuron_cache_path() is None

        custom_cache_dir_name = Path("_this/is_/my1/2custom/cache/dir")
        os.environ["NEURON_CC_FLAGS"] = (
            f"--some --parameters --here --cache_dir={custom_cache_dir_name} --other --paremeters --here"
        )

        self.assertEqual(get_neuron_cache_path(), custom_cache_dir_name)

        os.environ["NEURON_CC_FLAGS"] = "--some --parameters --here --other --paremeters --here"
        self.assertEqual(get_neuron_cache_path(), Path("/var/tmp/neuron-compile-cache"))

    def _test_set_neuron_cache_path(self, new_cache_path):
        os.environ["NEURON_CC_FLAGS"] = "--some --parameters --here --no-cache --other --paremeters --here"
        with self.assertRaisesRegex(ValueError, expected_regex=r"Cannot set the neuron compile cache"):
            set_neuron_cache_path(new_cache_path)
        set_neuron_cache_path(new_cache_path, ignore_no_cache=True)
        self.assertEqual(get_neuron_cache_path(), Path(new_cache_path))

        os.environ["NEURON_CC_FLAGS"] = (
            "--some --parameters --here --cache_dir=original_cache_dir --other --paremeters"
        )
        set_neuron_cache_path(new_cache_path)
        self.assertEqual(get_neuron_cache_path(), Path(new_cache_path))

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
            self.assertSetEqual(set(filenames), set(list_files_in_neuron_cache(tmpdirname)))

        with TemporaryDirectory() as tmpdirname:
            filenames = self._create_random_neuron_cache(Path(tmpdirname), return_only_relevant_files=True)
            self.assertSetEqual(set(filenames), set(list_files_in_neuron_cache(tmpdirname, only_relevant_files=True)))

    def test_list_in_registry_dict(self):
        registry = {
            "2.1.0": {
                "model_1": {
                    "model_name_or_path": "model_1",
                    "model_hash": "my model hash",
                    "features": [
                        {
                            "input_shapes": [["x", [1, 2]], ["y", [2, 3]]],
                            "precision": "torch.float32",
                            "num_neuron_cores": 16,
                            "neuron_hash": "neuron hash 1",
                        },
                        {
                            "input_shapes": [["x", [3, 2]], ["y", [7, 3]]],
                            "precision": "torch.float32",
                            "num_neuron_cores": 8,
                            "neuron_hash": "neuron hash 2",
                        },
                    ],
                },
                "model_2": {
                    "model_name_or_path": "null",
                    "model_hash": "my model hash 2",
                    "features": [
                        {
                            "input_shapes": [["x", [1, 2]], ["y", [2, 3]]],
                            "precision": "torch.float16",
                            "num_neuron_cores": 16,
                            "neuron_hash": "neuron hash 3",
                        },
                        {
                            "input_shapes": [["x", [3, 2]], ["y", [7, 3]]],
                            "precision": "torch.float32",
                            "num_neuron_cores": 8,
                            "neuron_hash": "neuron hash 4",
                        },
                    ],
                },
            },
            "2.5.0": {
                "model_1": {
                    "model_name_or_path": "model_1",
                    "model_hash": "my model hash",
                    "features": [
                        {
                            "input_shapes": [["x", [1, 2]], ["y", [2, 3]]],
                            "precision": "torch.float32",
                            "num_neuron_cores": 16,
                            "neuron_hash": "neuron hash 5",
                        },
                        {
                            "input_shapes": [["x", [3, 2]], ["y", [7, 3]]],
                            "precision": "torch.float32",
                            "num_neuron_cores": 8,
                            "neuron_hash": "neuron hash 6",
                        },
                    ],
                },
            },
        }

        result = _list_in_registry_dict(registry)
        self.assertEqual(len(result), 6)
        self.assertTrue(result[-1].startswith("Model name:\tmodel_1"))

        result = _list_in_registry_dict(registry, model_name_or_path_or_hash="model_1")
        self.assertEqual(len(result), 4)
        self.assertTrue(result[0].startswith("Model name:\tmodel_1"))

        result = _list_in_registry_dict(registry, model_name_or_path_or_hash="my model hash 2")
        self.assertEqual(len(result), 2)
        self.assertTrue(result[0].startswith("Model name:\tnull"))

        result = _list_in_registry_dict(registry, neuron_compiler_version="2.5.0")
        self.assertEqual(len(result), 2)
        self.assertTrue(result[0].startswith("Model name:\tmodel_1"))

        result = _list_in_registry_dict(registry, model_name_or_path_or_hash="random bad string")
        self.assertEqual(len(result), 0)

        result = _list_in_registry_dict(registry, neuron_compiler_version="-1.2")
        self.assertEqual(len(result), 0)


@is_staging_test
class StagingNeuronUtilsTestCase(StagingTestMixin, TestCase):
    def test_set_custom_cache_repo_name_in_hf_home(self):
        orig_token = get_token()
        login(TOKEN)

        repo_name = f"blablabla-{self.seed}"
        repo_id = f"{USER}/{repo_name}"
        create_repo(repo_name, repo_type="model")

        def remove_repo():
            delete_repo(repo_name, repo_type="model")

        with TemporaryDirectory() as tmpdirname:
            try:
                set_custom_cache_repo_name_in_hf_home(repo_id, hf_home=tmpdirname)
            except ValueError as e:
                remove_repo()
                if orig_token:
                    login(orig_token)
                self.fail(str(e))

            with open(f"{tmpdirname}/{CACHE_REPO_FILENAME}", "r") as fp:
                content = fp.read()

            self.assertEqual(content, repo_id, "The stored repo id must match the one specified.")

            with self.assertLogs("optimum", level=logging.WARNING) as cm:
                set_custom_cache_repo_name_in_hf_home(repo_id, hf_home=tmpdirname)
                self.assertIn("A custom cache repo was already", cm.output[0])

            remove_repo()
            if orig_token:
                login(orig_token)

    def test_has_write_access_to_repo(self):
        orig_token = get_token()

        wrong_token = "random_string"
        path = Path(huggingface_hub.constants.HF_TOKEN_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(wrong_token)

        self.assertFalse(has_write_access_to_repo(self.CUSTOM_CACHE_REPO))
        self.assertFalse(has_write_access_to_repo(self.CUSTOM_PRIVATE_CACHE_REPO))

        login(orig_token)

        self.assertTrue(has_write_access_to_repo(self.CUSTOM_CACHE_REPO))
        self.assertTrue(has_write_access_to_repo(self.CUSTOM_PRIVATE_CACHE_REPO))

    @is_trainium_test
    def test_list_in_registry(self):
        def _test_list_in_registry(use_private_cache_repo: bool):
            if use_private_cache_repo:
                cache_repo = self.CUSTOM_PRIVATE_CACHE_REPO
            else:
                cache_repo = self.CUSTOM_CACHE_REPO
            create_registry_file_if_does_not_exist(cache_repo)
            entries = list_in_registry(cache_repo)
            self.assertEqual(len(entries), 0)

            bert_model = BertModel(BertConfig())
            neuron_hash = NeuronHash(
                bert_model,
                (("x", (4, 12)), ("y", (4, 12))),
                torch.float32,
                2,
                neuron_compiler_version=DUMMY_COMPILER_VERSION,
            )
            add_in_registry(cache_repo, neuron_hash)
            entries = list_in_registry(cache_repo)
            self.assertEqual(len(entries), 1)

            bert_model = BertModel(BertConfig())
            neuron_hash = NeuronHash(
                bert_model,
                (("x", (4, 8)), ("y", (4, 12))),
                torch.float32,
                2,
                neuron_compiler_version=DUMMY_COMPILER_VERSION,
            )
            add_in_registry(cache_repo, neuron_hash)
            entries = list_in_registry(cache_repo)
            self.assertEqual(len(entries), 2)

            model_hash = neuron_hash.compute_hash()[0]
            entries = list_in_registry(cache_repo, model_name_or_path_or_hash=model_hash)
            self.assertEqual(len(entries), 1)

            entries = list_in_registry(cache_repo, model_name_or_path_or_hash="dummy hash")
            self.assertEqual(len(entries), 0)

            entries = list_in_registry(cache_repo, neuron_compiler_version=DUMMY_COMPILER_VERSION)
            self.assertEqual(len(entries), 2)

            entries = list_in_registry(cache_repo, neuron_compiler_version="Bad version")
            self.assertEqual(len(entries), 0)

        _test_list_in_registry(False)
        _test_list_in_registry(True)


@is_trainium_test
class NeuronHashTestCase(TestCase):
    def test_neuron_hash_is_not_mutable(self):
        bert_model = BertModel(BertConfig())
        neuron_hash = NeuronHash(
            bert_model,
            (("x", (4, 12)), ("y", (4, 12))),
            torch.float32,
            2,
            neuron_compiler_version=DUMMY_COMPILER_VERSION,
        )

        with self.assertRaises(FrozenInstanceError):
            neuron_hash.model = bert_model

        with self.assertRaises(FrozenInstanceError):
            neuron_hash.input_shapes = (("x", (2, 32)), ("y", (2, 32)))

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
        neuron_hash_a = NeuronHash(
            model_a,
            input_shapes_a,
            dtype_a,
            num_neuron_cores=num_neuron_cores_a,
            neuron_compiler_version=DUMMY_COMPILER_VERSION,
        )
        neuron_hash_b = NeuronHash(
            model_b,
            input_shapes_b,
            dtype_b,
            num_neuron_cores=num_neuron_cores_b,
            neuron_compiler_version=DUMMY_COMPILER_VERSION,
        )
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
        input_shapes = (("x", (1, 2)), ("y", (2, 3)))
        data_type = torch.float32
        num_neuron_cores = 32

        neuron_hash = NeuronHash(
            bert_model,
            input_shapes,
            data_type,
            num_neuron_cores=num_neuron_cores,
            neuron_compiler_version=DUMMY_COMPILER_VERSION,
        )
        hashes = neuron_hash.compute_hash()
        expected_folders = [DUMMY_COMPILER_VERSION, "bert"] + list(hashes)
        self.assertListEqual(neuron_hash.folders, expected_folders)

    def test_neuron_hash_is_private(self):
        input_shapes = (("x", (1, 2)), ("y", (2, 3)))
        data_type = torch.float32

        bert_model = BertModel(BertConfig())
        neuron_hash = NeuronHash(bert_model, input_shapes, data_type, neuron_compiler_version=DUMMY_COMPILER_VERSION)
        self.assertTrue(neuron_hash.is_private)

        bert_model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")
        neuron_hash = NeuronHash(bert_model, input_shapes, data_type, neuron_compiler_version=DUMMY_COMPILER_VERSION)
        self.assertFalse(neuron_hash.is_private)

        with TemporaryDirectory() as tmpdirname:
            bert_model.save_pretrained(tmpdirname)
            local_bert_model = BertModel.from_pretrained(tmpdirname)
            neuron_hash = NeuronHash(
                local_bert_model, input_shapes, data_type, neuron_compiler_version=DUMMY_COMPILER_VERSION
            )
            self.assertTrue(neuron_hash.is_private)


@is_trainium_test
@is_staging_test
class CachedModelOnTheHubTestCase(StagingTestMixin, TestCase):
    def test_push_to_hub_fails_with_private_model_and_public_repo(self):
        with TemporaryDirectory() as tmpdirname:
            set_neuron_cache_path(tmpdirname)

            input_shapes = (("x", (1,)),)
            data_type = torch.float32
            tiny_model = self.create_and_run_tiny_pretrained_model(random_num_linears=True)
            neuron_hash = NeuronHash(tiny_model, input_shapes, data_type)

            cached_files = list_files_in_neuron_cache(tmpdirname)

            # The model being loaded locally is assumed to be private, push to hub should prevent from pushing to a
            # public repo.
            with self.assertRaisesRegex(ValueError, "Could not push the cached model"):
                push_to_cache_on_hub(
                    neuron_hash, cached_files[0], self.CUSTOM_CACHE_REPO, fail_when_could_not_push=True
                )

            # It should work when using a private repo.
            cached_model_on_the_hub = push_to_cache_on_hub(
                neuron_hash, cached_files[0], self.CUSTOM_PRIVATE_CACHE_REPO
            )
            self.assertIsNotNone(cached_model_on_the_hub)

    def test_push_to_hub_without_specifying_a_cache_repo_id(self):
        with TemporaryDirectory() as tmpdirname:
            set_neuron_cache_path(tmpdirname)

            input_shapes = (("x", (1,)),)
            data_type = torch.float32
            tiny_model = self.create_and_run_tiny_pretrained_model(random_num_linears=True)
            neuron_hash = NeuronHash(tiny_model, input_shapes, data_type)

            cached_files = list_files_in_neuron_cache(tmpdirname)

            set_custom_cache_repo_name_in_hf_home(self.CUSTOM_PRIVATE_CACHE_REPO)
            push_to_cache_on_hub(neuron_hash, cached_files[0])

    def test_push_to_hub_overwrite_existing(self):
        with TemporaryDirectory() as tmpdirname:
            set_neuron_cache_path(tmpdirname)

            input_shapes = (("x", (1,)),)
            data_type = torch.float32
            tiny_model = self.create_and_run_tiny_pretrained_model(random_num_linears=True)
            neuron_hash = NeuronHash(tiny_model, input_shapes, data_type)

            cache_dir = Path(tmpdirname)
            cached_files = list_files_in_neuron_cache(cache_dir)

            push_to_cache_on_hub(neuron_hash, cached_files[0], self.CUSTOM_PRIVATE_CACHE_REPO)

            # With a file
            with self.assertLogs("optimum", level="INFO") as cm:
                push_to_cache_on_hub(neuron_hash, cached_files[0], self.CUSTOM_PRIVATE_CACHE_REPO)
                self.assertIn("Did not push the cached model located at", cm.output[0])

            with self.assertLogs("optimum", level="WARNING") as cm:
                push_to_cache_on_hub(
                    neuron_hash, cached_files[0], self.CUSTOM_PRIVATE_CACHE_REPO, overwrite_existing=True
                )
                self.assertIn(
                    "Overwriting the already existing cached model on the Hub by the one located at", cm.output[0]
                )

            # With a directory
            with self.assertLogs("optimum", level="INFO") as cm:
                push_to_cache_on_hub(neuron_hash, cache_dir, self.CUSTOM_PRIVATE_CACHE_REPO)
                self.assertIn("Did not push the cached model located at", cm.output[0])

            with self.assertLogs("optimum", level="WARNING") as cm:
                push_to_cache_on_hub(neuron_hash, cache_dir, self.CUSTOM_PRIVATE_CACHE_REPO, overwrite_existing=True)
                self.assertIn(
                    "Overwriting the already existing cached model on the Hub by the one located at", cm.output[0]
                )

    def test_push_to_hub_local_path_in_repo(self):
        with TemporaryDirectory() as tmpdirname:
            set_neuron_cache_path(tmpdirname)

            input_shapes = (("x", (1,)),)
            data_type = torch.float32
            tiny_model = self.create_and_run_tiny_pretrained_model(random_num_linears=True)
            neuron_hash = NeuronHash(tiny_model, input_shapes, data_type)

            cache_dir = Path(tmpdirname)
            cached_files = list_files_in_neuron_cache(cache_dir)

            def local_path_to_path_in_repo(path):
                return Path("my/awesome/new/path") / path.name

            cached_file = cached_files[0]

            # With a file
            push_to_cache_on_hub(
                neuron_hash,
                cached_file,
                self.CUSTOM_PRIVATE_CACHE_REPO,
                local_path_to_path_in_repo=local_path_to_path_in_repo,
            )
            files_in_repo = HfApi().list_repo_files(repo_id=self.CUSTOM_PRIVATE_CACHE_REPO)
            anonymous_cached_file = remove_ip_adress_from_path(cached_file)
            path_in_repo = f"{neuron_hash.cache_path}/my/awesome/new/path/{anonymous_cached_file.name}"
            self.assertIn(path_in_repo, files_in_repo)

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
                    path_in_cache_dir = path_after_folder(filename, cache_dir, include_folder=True)
                    anonymous_path_in_cache_dir = remove_ip_adress_from_path(path_in_cache_dir)
                    path_in_repo = (
                        f"{neuron_hash.cache_path}/my/another/awesome/new/path/{anonymous_path_in_cache_dir}"
                    )
                    self.assertIn(path_in_repo, files_in_repo)

    def test_push_to_hub_without_writing_rights(self):
        with TemporaryDirectory() as tmpdirname:
            import torch_xla.core.xla_model as xm

            set_neuron_cache_path(tmpdirname)

            input_shapes = (("x", (1,)),)
            data_type = torch.float32
            tiny_model = self.create_and_run_tiny_pretrained_model(random_num_linears=True)
            tiny_model.push_to_hub(f"tiny-public-model-{self.seed}")
            public_tiny_model = MyTinyModel.from_pretrained(f"{USER}/tiny-public-model-{self.seed}")
            neuron_hash = NeuronHash(public_tiny_model, input_shapes, data_type)

            public_tiny_model = public_tiny_model.to("xla")
            input_ = torch.rand((32, 1)).to("xla")
            public_tiny_model(input_)
            xm.mark_step()

            # This should work because we do have writing access to this repo.
            set_custom_cache_repo_name_in_hf_home(self.CUSTOM_CACHE_REPO)
            push_to_cache_on_hub(neuron_hash, get_neuron_cache_path())

            # Creating a repo under the Transformers user.
            orig_token = self.set_hf_hub_token(TRANSFORMERS_TOKEN)
            repo_name = f"optimum-neuron-cache-{self.seed}"
            create_repo(repo_name, repo_type="model", exist_ok=True)
            self.set_hf_hub_token(orig_token)

            set_custom_cache_repo_name_in_hf_home(f"{TRANSFORMERS_USER}/{repo_name}")
            with self.assertLogs("optimum", "WARNING") as cm:
                push_to_cache_on_hub(neuron_hash, get_neuron_cache_path())
                self.assertTrue(any("Could not push the cached model to" in output for output in cm.output))

            self.set_hf_hub_token(TRANSFORMERS_TOKEN)
            delete_repo(repo_name, repo_type="model")
            self.set_hf_hub_token(orig_token)

    def _test_push_to_hub_create_and_add_registry(self, with_model_name_or_path: bool):
        with TemporaryDirectory() as tmpdirname:
            set_neuron_cache_path(tmpdirname)

            input_shapes = (("x", (1,)),)
            data_type = torch.float32
            data_type = torch.float32
            tiny_model = self.create_and_run_tiny_pretrained_model(random_num_linears=True)
            model_name = f"dummy_model-{self.seed}"
            if with_model_name_or_path:
                tiny_model.push_to_hub(model_name)
                model_name = f"{USER}/{model_name}"
                tiny_model.config._model_name_or_path = model_name
            neuron_hash = NeuronHash(tiny_model, input_shapes, data_type)

            set_custom_cache_repo_name_in_hf_home(self.CUSTOM_PRIVATE_CACHE_REPO)
            files_in_repo = HfApi().list_repo_files(repo_id=self.CUSTOM_PRIVATE_CACHE_REPO)
            files_in_repo = [filename for filename in files_in_repo if not filename.startswith(".")]
            self.assertListEqual(files_in_repo, [], "Repo should be empty")

            cached_files = list_files_in_neuron_cache(tmpdirname)
            push_to_cache_on_hub(neuron_hash, cached_files[0])
            files_in_repo = HfApi().list_repo_files(repo_id=self.CUSTOM_PRIVATE_CACHE_REPO)

            self.assertIn(REGISTRY_FILENAME, files_in_repo)
            hf_hub_download(
                self.CUSTOM_PRIVATE_CACHE_REPO,
                REGISTRY_FILENAME,
                force_download=True,
                local_dir=tmpdirname,
                local_dir_use_symlinks=False,
            )
            with open(Path(tmpdirname) / REGISTRY_FILENAME, "r") as fp:
                registry = json.load(fp)

            neuron_compiler_version = list(registry.keys())[0]
            model_key = list(registry[neuron_compiler_version].keys())[0]
            expected_value = model_name if with_model_name_or_path else neuron_hash.compute_hash()[0]
            self.assertEqual(model_key, expected_value)

    def test_push_to_hub_create_and_add_registry_without_model_name_or_path(self):
        return self._test_push_to_hub_create_and_add_registry(False)

    def test_push_to_hub_create_and_add_registry_with_model_name_or_path(self):
        return self._test_push_to_hub_create_and_add_registry(True)

    def test_download_cached_model_from_hub(self):
        set_custom_cache_repo_name_in_hf_home(self.CUSTOM_PRIVATE_CACHE_REPO)
        neuron_hash = self.push_tiny_pretrained_model_cache_to_hub(self.CUSTOM_PRIVATE_CACHE_REPO)

        neuron_cc_flags = os.environ["NEURON_CC_FLAGS"]

        with self.assertRaisesRegex(
            ValueError, "A target directory must be specified when no caching directory is used"
        ):
            os.environ["NEURON_CC_FLAGS"] = "--no-cache"
            self.assertTrue(download_cached_model_from_hub(neuron_hash))

        os.environ["NEURON_CC_FLAGS"] = neuron_cc_flags
        self.assertTrue(download_cached_model_from_hub(neuron_hash))

    def test_download_cached_model_from_hub_with_target_directory(self):
        set_custom_cache_repo_name_in_hf_home(self.CUSTOM_PRIVATE_CACHE_REPO)
        neuron_hash = self.push_tiny_pretrained_model_cache_to_hub(self.CUSTOM_PRIVATE_CACHE_REPO)

        cached_model_on_the_hub = get_cached_model_on_the_hub(neuron_hash)
        if cached_model_on_the_hub is None:
            self.fail("Could not find the model on the Hub, but it should be there.")

        repo_files = set(cached_model_on_the_hub.files_on_the_hub)

        if len(repo_files) == 0:
            self.fail("Could not find any file in the Hub.")

        # With a target directory specified as a string.
        with TemporaryDirectory() as tmpdirname:
            success = download_cached_model_from_hub(neuron_hash, target_directory=tmpdirname)
            self.assertTrue(success)

            tmpdir = Path(tmpdirname)
            target_directory_files = {str(path_after_folder(f, tmpdir)) for f in tmpdir.glob("**/*") if f.is_file()}
            self.assertSetEqual(target_directory_files, repo_files)

        # With a target directory specified as a Path.
        with TemporaryDirectory() as tmpdirname:
            tmpdir = Path(tmpdirname)
            success = download_cached_model_from_hub(neuron_hash, target_directory=tmpdir)
            self.assertTrue(success)

            target_directory_files = {str(path_after_folder(f, tmpdir)) for f in tmpdir.glob("**/*") if f.is_file()}
            self.assertSetEqual(target_directory_files, repo_files)

    def test_download_cached_model_from_hub_with_path_in_repo_to_path_in_target_directory(self):
        set_custom_cache_repo_name_in_hf_home(self.CUSTOM_PRIVATE_CACHE_REPO)
        neuron_hash = self.push_tiny_pretrained_model_cache_to_hub(self.CUSTOM_PRIVATE_CACHE_REPO)

        cached_model_on_the_hub = get_cached_model_on_the_hub(neuron_hash)
        if cached_model_on_the_hub is None:
            self.fail("Could not find the model on the Hub, but it should be there.")

        def path_in_repo_to_path_in_target_directory(path):
            return Path("custom_folder") / path.name

        repo_files = {
            path_in_repo_to_path_in_target_directory(Path(f)) for f in cached_model_on_the_hub.files_on_the_hub
        }

        if len(repo_files) == 0:
            self.fail("Could not find any file in the Hub.")

        # With a target directory specified as a string.
        with TemporaryDirectory() as tmpdirname:
            success = download_cached_model_from_hub(
                neuron_hash,
                target_directory=tmpdirname,
                path_in_repo_to_path_in_target_directory=path_in_repo_to_path_in_target_directory,
            )
            self.assertTrue(success)

            tmpdir = Path(tmpdirname)
            target_directory_files = {Path("custom_folder") / f.name for f in tmpdir.glob("**/*") if f.is_file()}
            self.assertSetEqual(target_directory_files, repo_files)

            # Check the the original download directories do not exist since we specified a
            # path_in_repo_to_path_in_target_directory function.
            # self.assertListEqual([f.name for f in tmpdir.iterdir()], ["custom_folder"])

    # TODO: not passing yet, to fix ASAP.
    # def test_download_cached_model_from_hub_needs_to_download(self):
    #     os.environ["CUSTOM_CACHE_REPO"] = self.CUSTOM_PRIVATE_CACHE_REPO

    #     with TemporaryDirectory() as tmpdirname:
    #         neuron_hash = self._push_tiny_pretrained_model_cache_to_hub(self.CUSTOM_PRIVATE_CACHE_REPO, cache_dir=tmpdirname)

    #         with patch("huggingface_hub.snapshot_download") as mock_snapshot_download:
    #             # All the files are already there, should not download anything.
    #             download_cached_model_from_hub(neuron_hash, target_directory=tmpdirname)
    #             self.assertFalse(mock_snapshot_download.called, "No downloading should be peformed since all the files are already in the cache.")
    #             mock_snapshot_download.reset_mock()
    #
    #             # All the files but one are there, should trigger downloading.
    #             for path in Path(tmpdirname).glob("**/*"):
    #                 if path.is_file():
    #                     if path.suffix in [".json", ".txt"]:
    #                         continue
    #                     path.unlink()
    #                     break

    #             download_cached_model_from_hub(neuron_hash, target_directory=tmpdirname)
    #             self.assertTrue(mock_snapshot_download.called, "Downloading should be peformed since one file is missing in the cache.")
    #             mock_snapshot_download.reset_mock()

    #             # No file at all, should download.
    #             with TemporaryDirectory() as another_tmpdirname:
    #                 download_cached_model_from_hub(neuron_hash, target_directory=another_tmpdirname)
    #                 self.assertTrue(mock_snapshot_download.called, "Downloading should be peformed since no file is in the cache.")
