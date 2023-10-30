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

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch
from huggingface_hub import HfApi
from transformers.testing_utils import is_staging_test

from optimum.neuron.trainers import NeuronCacheCallback
from optimum.neuron.training_args import NeuronTrainingArguments
from optimum.neuron.utils.cache_utils import (
    NeuronHash,
    list_files_in_neuron_cache,
    push_to_cache_on_hub,
    set_neuron_cache_path,
)
from optimum.neuron.utils.testing_utils import is_trainium_test

from .utils import StagingTestMixin


@is_trainium_test
@is_staging_test
class NeuronCacheCallbackTestCase(StagingTestMixin, TestCase):
    def test_neuron_hash_for_model(self):
        with TemporaryDirectory() as tmpdirname:
            args = NeuronTrainingArguments(tmpdirname)
        model = self.create_tiny_pretrained_model(random_num_linears=True)
        inputs = {
            "x": torch.rand((1,)),
        }

        callback = NeuronCacheCallback()

        # We first check that no hashes is in the hash cache already.
        self.assertFalse(callback.neuron_hashes)

        callback.neuron_hash_for_model(args, model, inputs)
        neuron_hash = callback.neuron_hashes[(model, (("x", tuple(inputs["x"].shape)),), torch.float32, 1)]

        same_neuron_hash = callback.neuron_hash_for_model(args, model, inputs)

        self.assertEqual(neuron_hash, same_neuron_hash, "Neuron hashes should be equal")
        self.assertEqual(len(callback.neuron_hashes.keys()), 1, "There should be only one entry in neuron_hashes.")

    def test_try_to_fetch_cached_model(self):
        import torch_xla.core.xla_model as xm

        os.environ["CUSTOM_CACHE_REPO"] = self.CUSTOM_PRIVATE_CACHE_REPO
        model = self.create_tiny_pretrained_model(random_num_linears=True).to("xla")

        with TemporaryDirectory() as tmpdirname:
            set_neuron_cache_path(tmpdirname)
            args = NeuronTrainingArguments(tmpdirname)
            inputs = {"x": torch.rand((8, 1)).to("xla")}
            output = model(**inputs)
            xm.mark_step()
            print(output)
            neuron_hash = NeuronHash(model, (("x", (8, 1)),), torch.float32)
            push_to_cache_on_hub(neuron_hash, Path(tmpdirname) / neuron_hash.neuron_compiler_version_dir_name)

        with TemporaryDirectory() as tmpdirname:
            set_neuron_cache_path(tmpdirname)
            callback = NeuronCacheCallback()
            args = NeuronTrainingArguments(tmpdirname)
            inputs = {"x": torch.rand((24, 1))}
            neuron_hash = callback.neuron_hash_for_model(args, model, inputs)

            found_in_cache = callback.try_to_fetch_cached_model(neuron_hash)
            self.assertFalse(found_in_cache, "No model should have been fetched.")

            inputs = {"x": torch.rand((8, 1))}
            neuron_hash = callback.neuron_hash_for_model(args, model, inputs)

            files_before_fetching = list_files_in_neuron_cache(
                callback.tmp_neuron_cache_path, only_relevant_files=True
            )
            tmp_neuron_cache_state = list(callback.tmp_neuron_cache_state)
            neuron_cache_state = list_files_in_neuron_cache(Path(tmpdirname), only_relevant_files=True)

            found_in_cache = callback.try_to_fetch_cached_model(neuron_hash)
            self.assertTrue(found_in_cache, "A model should have been fetched.")

            files_after_fetching = list_files_in_neuron_cache(callback.tmp_neuron_cache_path, only_relevant_files=True)
            new_tmp_neuron_cache_state = list(callback.tmp_neuron_cache_state)
            new_neuron_cache_state = list_files_in_neuron_cache(Path(tmpdirname), only_relevant_files=True)

            files_diff = [f for f in files_after_fetching if f not in files_before_fetching]
            state_diff = [f for f in new_tmp_neuron_cache_state if f not in tmp_neuron_cache_state]
            neuron_cache_files_diff = [f for f in new_neuron_cache_state if f not in neuron_cache_state]

            self.assertNotEqual(files_diff, [])
            self.assertListEqual(files_diff, state_diff)
            self.assertEqual(len(files_diff), len(neuron_cache_files_diff))

    def test_synchronize_temporary_neuron_cache_state(self):
        import torch_xla.core.xla_model as xm

        with TemporaryDirectory() as tmpdirname:
            set_neuron_cache_path(tmpdirname)
            callback = NeuronCacheCallback()

            diff = callback.synchronize_temporary_neuron_cache_state()
            self.assertListEqual(diff, [], "The diff should be empty.")

            model = self.create_tiny_pretrained_model(random_num_linears=True).to("xla")
            inputs = {"x": torch.rand((8, 1)).to("xla")}
            output = model(**inputs)
            xm.mark_step()
            print(output)
            diff = callback.synchronize_temporary_neuron_cache_state()
            self.assertNotEqual(diff, [], "The diff should not be empty.")

            diff = callback.synchronize_temporary_neuron_cache_state()
            self.assertListEqual(
                diff, [], "The diff should be empty because nothing happened since last synchronization"
            )

    def test_synchronize_temporary_neuron_cache(self):
        import torch_xla.core.xla_model as xm

        os.environ["CUSTOM_CACHE_REPO"] = self.CUSTOM_PRIVATE_CACHE_REPO
        model = self.create_tiny_pretrained_model(random_num_linears=True).to("xla")

        with TemporaryDirectory() as tmpdirname:
            set_neuron_cache_path(tmpdirname)
            args = NeuronTrainingArguments(tmpdirname)
            callback = NeuronCacheCallback()

            callback.synchronize_temporary_neuron_cache()
            files_in_repo = HfApi().list_repo_files(repo_id=self.CUSTOM_PRIVATE_CACHE_REPO)
            files_in_repo = [f for f in files_in_repo if not f.startswith(".")]
            files_in_cache = list_files_in_neuron_cache(callback.neuron_cache_path, only_relevant_files=True)
            self.assertListEqual(files_in_repo, [], "Repo should be empty.")
            self.assertListEqual(files_in_cache, [], "Cache should be empty.")

            # Running some compilation.
            for _ in range(3):
                inputs = {"x": torch.rand((8, 1)).to("xla")}
                output = model(**inputs)
                xm.mark_step()

            xm.mark_step()
            print(output)

            neuron_hash = callback.neuron_hash_for_model(args, model, inputs)
            diff = callback.synchronize_temporary_neuron_cache_state()
            callback.neuron_hash_to_files[neuron_hash].extend(diff)

            callback.synchronize_temporary_neuron_cache()

            files_in_repo = HfApi().list_repo_files(repo_id=self.CUSTOM_PRIVATE_CACHE_REPO)
            files_in_repo = [f for f in files_in_repo if not f.startswith(".")]
            files_in_cache = list_files_in_neuron_cache(callback.neuron_cache_path, only_relevant_files=True)
            self.assertNotEqual(files_in_repo, [], "Repo should not be empty.")
            self.assertNotEqual(files_in_cache, [], "Cache should not be empty.")

            # Using the same inputs, nothing should be uploaded.
            inputs = {"x": torch.rand((8, 1)).to("xla")}
            output = model(**inputs)
            xm.mark_step()
            print(output)

            neuron_hash = callback.neuron_hash_for_model(args, model, inputs)
            diff = callback.synchronize_temporary_neuron_cache_state()
            callback.neuron_hash_to_files[neuron_hash].extend(diff)

            callback.synchronize_temporary_neuron_cache()

            new_files_in_repo = HfApi().list_repo_files(repo_id=self.CUSTOM_PRIVATE_CACHE_REPO)
            new_files_in_repo = [f for f in new_files_in_repo if not f.startswith(".")]
            new_files_in_cache = list_files_in_neuron_cache(callback.neuron_cache_path, only_relevant_files=True)
            self.assertListEqual(files_in_repo, new_files_in_repo, "No new file should be in the Hub.")
            self.assertListEqual(files_in_cache, new_files_in_cache, "No new file should be in the cache.")

            # New shape, should upload.
            inputs = {"x": torch.rand((24, 1)).to("xla")}
            output = model(**inputs)
            xm.mark_step()
            print(output)

            neuron_hash = callback.neuron_hash_for_model(args, model, inputs)
            diff = callback.synchronize_temporary_neuron_cache_state()
            callback.neuron_hash_to_files[neuron_hash].extend(diff)

            callback.synchronize_temporary_neuron_cache()

            files_in_repo = HfApi().list_repo_files(repo_id=self.CUSTOM_PRIVATE_CACHE_REPO)
            files_in_repo = [f for f in files_in_repo if not f.startswith(".")]
            files_in_cache = list_files_in_neuron_cache(callback.neuron_cache_path, only_relevant_files=True)
            self.assertNotEqual(files_in_repo, new_files_in_repo, "New files should be in the Hub.")
            self.assertNotEqual(files_in_cache, new_files_in_cache, "New files should be in the cache.")
