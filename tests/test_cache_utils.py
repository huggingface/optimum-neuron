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
from pathlib import Path
from unittest import TestCase

from optimum.neuron.utils.cache_utils import NEURON_COMPILE_CACHE_NAME, get_neuron_cache_path, get_num_neuron_cores_used, set_neuron_cache_path

def get_random_string(length) -> str:
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))


class NeuronUtilsTestCase(TestCase):

    def test_get_neuron_cache_path(self):
        os.environ["NEURON_CC_FLAGS"] = "--some --parameters --here --no-cache --other --paremeters --here"
        assert get_neuron_cache_path() is None
    
        custom_cache_dir_name = Path("_this/is_/my1/2custom/cache/dir")
        os.environ["NEURON_CC_FLAGS"] = f"--some --parameters --here --cache_dir={custom_cache_dir_name} --other --paremeters --here"
        self.assertEqual(get_neuron_cache_path(), custom_cache_dir_name / NEURON_COMPILE_CACHE_NAME)
    
        os.environ["NEURON_CC_FLAGS"] = "--some --parameters --here --other --paremeters --here"
        self.assertEqual(get_neuron_cache_path(), Path("/var/tmp") / NEURON_COMPILE_CACHE_NAME)
        

    def _test_set_neuron_cache_path(self, new_cache_path):
        os.environ["NEURON_CC_FLAGS"] = "--some --parameters --here --no-cache --other --paremeters --here"
        with self.assertRaisesRegex(ValueError, expected_regex=r"Cannot set the neuron compile cache"):
            set_neuron_cache_path(new_cache_path)
        set_neuron_cache_path(new_cache_path, ignore_no_cache=True)
        self.assertEqual(get_neuron_cache_path(), Path(new_cache_path) / NEURON_COMPILE_CACHE_NAME)

        os.environ["NEURON_CC_FLAGS"] = "--some --parameters --here --cache_dir=original_cache_dir --other --paremeters"
        set_neuron_cache_path(new_cache_path)
        self.assertEqual(get_neuron_cache_path(), Path(new_cache_path) / NEURON_COMPILE_CACHE_NAME)

    def test_set_neuron_cache_path(self):
        new_cache_path_str = "path/to/my/custom/cache"
        new_cache_path_path = Path(new_cache_path_str)
        self._test_set_neuron_cache_path(new_cache_path_str)
        self._test_set_neuron_cache_path(new_cache_path_path)


    def test_get_num_neuron_cores_used(self):
        self.assertEqual(get_num_neuron_cores_used(), 1)

        randon_num_cores = random.randnint(1, 32)
        os.environ["LOCAL_WORLD_SIZE"] = randon_num_cores
        self.assertEqual(get_num_neuron_cores_used(), randon_num_cores)

    def _create_random_neuron_cache(self, number_of_right_cache_files: int = 32):
        wrong_extensions = [get_random_string(3) for _ in range(4)]
        right_extensions = ["neff", "pb", "txt"]
        extensions = right_extensions + wrong_extensions
        for _ in range(number_of_right_cache_files):
            pass
        
    def test_list_files_in_neuron_cache():
        pass

