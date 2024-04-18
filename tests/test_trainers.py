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
"""Tests related to the Trainer derived classes."""

import copy
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pytest
from huggingface_hub import HfApi
from transformers import LlamaForCausalLM
from transformers.testing_utils import is_staging_test

from optimum.neuron import NeuronTrainer, NeuronTrainingArguments
from optimum.neuron.utils import is_neuronx_distributed_available
from optimum.neuron.utils.cache_utils import (
    get_neuron_cache_path,
    list_files_in_neuron_cache,
    set_neuron_cache_path,
)
from optimum.neuron.utils.testing_utils import is_trainium_test

from . import DistributedTest
from .utils import (
    StagingTestMixin,
    create_dummy_causal_lm_dataset,
    create_dummy_dataset,
    default_data_collator_for_causal_lm,
    get_model,
)


if is_neuronx_distributed_available():
    from neuronx_distributed.parallel_layers.parallel_state import (
        get_data_parallel_rank,
        get_pipeline_model_parallel_rank,
        get_tensor_model_parallel_rank,
    )


LLAMA_V2_MODEL_NAME = "michaelbenayoun/llama-2-tiny-4kv-heads-4layers-random"


@is_trainium_test
@is_staging_test
class StagingNeuronTrainerTestCase(StagingTestMixin, TestCase):
    @pytest.mark.skip("Seems to be working but takes forever")
    def test_train_and_eval(self):
        os.environ["CUSTOM_CACHE_REPO"] = self.CUSTOM_PRIVATE_CACHE_REPO

        # We take a batch size that does not divide the total number of samples.
        num_train_samples = 1000
        per_device_train_batch_size = 32
        dummy_train_dataset = create_dummy_dataset({"x": (1,), "labels": (1,)}, num_train_samples)

        # We take a batch size that does not divide the total number of samples.
        num_eval_samples = 100
        per_device_eval_batch_size = 16
        dummy_eval_dataset = create_dummy_dataset({"x": (1,), "labels": (1,)}, num_eval_samples)

        model = create_tiny_pretrained_model(random_num_linears=True)
        clone = copy.deepcopy(model)

        with TemporaryDirectory() as tmpdirname:
            set_neuron_cache_path(tmpdirname)

            files_in_repo = HfApi().list_repo_files(repo_id=self.CUSTOM_PRIVATE_CACHE_REPO)
            files_in_repo = [f for f in files_in_repo if not f.startswith(".")]
            files_in_cache = list_files_in_neuron_cache(get_neuron_cache_path(), only_relevant_files=True)
            self.assertListEqual(files_in_repo, [], "Repo should be empty.")
            self.assertListEqual(files_in_cache, [], "Cache should be empty.")

            args = NeuronTrainingArguments(
                tmpdirname,
                do_train=True,
                do_eval=True,
                bf16=True,
                per_device_train_batch_size=per_device_train_batch_size,
                per_device_eval_batch_size=per_device_eval_batch_size,
                save_steps=10,
                num_train_epochs=2,
            )
            trainer = NeuronTrainer(
                model,
                args,
                train_dataset=dummy_train_dataset,
                eval_dataset=dummy_eval_dataset,
            )
            start = time.time()
            trainer.train()
            end = time.time()
            first_training_duration = end - start

            files_in_repo = HfApi().list_repo_files(repo_id=self.CUSTOM_PRIVATE_CACHE_REPO)
            files_in_repo = [f for f in files_in_repo if not f.startswith(".")]
            files_in_cache = list_files_in_neuron_cache(get_neuron_cache_path(), only_relevant_files=True)
            self.assertNotEqual(files_in_repo, [], "Repo should not be empty after first training.")
            self.assertNotEqual(files_in_cache, [], "Cache should not be empty after first training.")

        with TemporaryDirectory() as tmpdirname:
            set_neuron_cache_path(tmpdirname)

            new_files_in_repo = HfApi().list_repo_files(repo_id=self.CUSTOM_PRIVATE_CACHE_REPO)
            new_files_in_repo = [f for f in new_files_in_repo if not f.startswith(".")]
            new_files_in_cache = list_files_in_neuron_cache(get_neuron_cache_path(), only_relevant_files=True)
            self.assertNotEqual(new_files_in_repo, [], "Repo should not be empty.")
            self.assertListEqual(new_files_in_cache, [], "Cache should be empty.")

            args = NeuronTrainingArguments(
                tmpdirname,
                do_train=True,
                do_eval=True,
                bf16=True,
                per_device_train_batch_size=per_device_train_batch_size,
                per_device_eval_batch_size=per_device_eval_batch_size,
                save_steps=10,
                num_train_epochs=2,
            )
            trainer = NeuronTrainer(
                clone,
                args,
                train_dataset=dummy_train_dataset,
                eval_dataset=dummy_eval_dataset,
            )
            start = time.time()
            trainer.train()
            end = time.time()
            second_training_duration = end - start

            last_files_in_repo = HfApi().list_repo_files(repo_id=self.CUSTOM_PRIVATE_CACHE_REPO)
            last_files_in_repo = [f for f in last_files_in_repo if not f.startswith(".")]
            last_files_in_cache = list_files_in_neuron_cache(get_neuron_cache_path(), only_relevant_files=True)
            # TODO: investigate that, not urgent.
            self.assertListEqual(
                files_in_repo, last_files_in_repo, "No file should have been added to the Hub after first training."
            )
            self.assertListEqual(
                files_in_cache,
                last_files_in_cache,
                "No file should have been added to the cache after first training.",
            )

            self.assertTrue(
                second_training_duration < first_training_duration,
                "Second training should be faster because cached graphs can be used.",
            )


@is_trainium_test
class TestNeuronTrainer(DistributedTest):
    @pytest.fixture(
        scope="class",
        params=[[2, 1, 1], [2, 2, 1], [2, 1, 2], [16, 2, 2]],
        ids=["dp=2", "tp=2", "pp=2", "dp=4,tp=pp=2"],
    )
    def parallel_sizes(self, request):
        return request.param

    def test_save_checkpoint(self, parallel_sizes, tmpdir):
        world_size, tp_size, pp_size = parallel_sizes
        output_dir = Path(tmpdir)

        dp_rank = get_data_parallel_rank()
        tp_rank = get_tensor_model_parallel_rank()
        pp_rank = get_pipeline_model_parallel_rank()

        model = get_model(LlamaForCausalLM, LLAMA_V2_MODEL_NAME, tp_size=tp_size, pp_size=pp_size)
        datasets = create_dummy_causal_lm_dataset(model.config.vocab_size, 120, 1)

        args = NeuronTrainingArguments(
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            per_device_train_batch_size=2,
            save_steps=5,
            max_steps=20,
            output_dir=output_dir.as_posix(),
        )

        trainer = NeuronTrainer(
            args=args, model=model, train_dataset=datasets["train"], data_collator=default_data_collator_for_causal_lm
        )

        trainer.train()
