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

import logging
import os
import random
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List
from unittest import TestCase

import huggingface_hub
from huggingface_hub import create_repo, delete_repo, get_token, login
from transformers.testing_utils import is_staging_test

from optimum.neuron.utils.cache_utils import (
    CACHE_REPO_FILENAME,
    get_neuron_cache_path,
    get_num_neuron_cores_used,
    has_write_access_to_repo,
    list_files_in_neuron_cache,
    load_custom_cache_repo_name_from_hf_home,
    set_custom_cache_repo_name_in_hf_home,
    set_neuron_cache_path,
)
from optimum.neuron.utils.testing_utils import is_trainium_test

from .utils import TOKEN_STAGING, USER_STAGING, StagingTestMixin, TrainiumTestMixin, get_random_string


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
        os.environ["WORLD_SIZE"] = str(randon_num_cores)
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


@is_staging_test
class StagingNeuronUtilsTestCase(StagingTestMixin, TestCase):
    def test_set_custom_cache_repo_name_in_hf_home(self):
        orig_token = get_token()
        login(TOKEN_STAGING)

        repo_name = f"blablabla-{self.seed}"
        repo_id = f"{USER_STAGING}/{repo_name}"
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
