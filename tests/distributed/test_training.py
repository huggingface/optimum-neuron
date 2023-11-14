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
"""Tests related to training with `neuronx_distributed`."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from huggingface_hub import HfFolder

from optimum.neuron.utils.cache_utils import (
    delete_custom_cache_repo_name_from_hf_home,
    load_custom_cache_repo_name_from_hf_home,
    set_custom_cache_repo_name_in_hf_home,
)
from optimum.neuron.utils.runner import ExampleRunner
from optimum.neuron.utils.testing_utils import is_trainium_test


_TINY_BERT_MODEL_NAME = "hf-internal-testing/tiny-random-bert"


@is_trainium_test
class DistributedTrainingTestCase(TestCase):
    CACHE_REPO_NAME = "optimum-internal-testing/optimum-neuron-cache-for-testing"

    @classmethod
    def setUpClass(cls):
        orig_token = HfFolder.get_token()
        orig_cache_repo = load_custom_cache_repo_name_from_hf_home()
        ci_token = os.environ.get("HF_TOKEN_OPTIMUM_NEURON_CI", None)
        if ci_token is not None:
            HfFolder.save_token(ci_token)
            set_custom_cache_repo_name_in_hf_home(cls.CACHE_REPO_NAME)
        cls._token = orig_token
        cls._cache_repo = orig_cache_repo
        cls._env = dict(os.environ)

    @classmethod
    def tearDownClass(cls):
        os.environ = cls._env
        if cls._token is not None:
            HfFolder.save_token(cls._token)
        if cls._cache_repo is not None:
            set_custom_cache_repo_name_in_hf_home(cls._cache_repo)
        else:
            delete_custom_cache_repo_name_from_hf_home()

    def test_tp_save_and_resume_from_checkpoint(self):
        num_cores = 8
        precision = "bf16"
        tensor_parallel_size = 2
        train_batch_size = 2
        eval_batch_size = 2
        sequence_length = 16
        max_steps = 10
        save_steps = 2
        do_eval = True
        max_eval_samples = 16

        with TemporaryDirectory() as tmpdirname:
            output_dir = Path(tmpdirname)

            runner = ExampleRunner(_TINY_BERT_MODEL_NAME, "text-classification")

            first_output_dir = output_dir / "first_run"
            returncode, _ = runner.run(
                num_cores,
                precision,
                train_batch_size,
                eval_batch_size=eval_batch_size,
                sequence_length=sequence_length,
                tensor_parallel_size=tensor_parallel_size,
                max_steps=max_steps,
                save_steps=save_steps,
                do_eval=do_eval,
                max_eval_samples=max_eval_samples,
                output_dir=first_output_dir,
                print_outputs=True,
            )
            assert returncode == 0, "First run failed."

            # Case 1: Resuming from checkpoint by specifying a checkpoint directory.
            second_output_dir = output_dir / "second_run"
            returncode, _ = runner.run(
                num_cores,
                precision,
                train_batch_size,
                eval_batch_size=eval_batch_size,
                sequence_length=sequence_length,
                tensor_parallel_size=tensor_parallel_size,
                max_steps=max_steps,
                save_steps=save_steps,
                do_eval=do_eval,
                max_eval_samples=max_eval_samples,
                output_dir=second_output_dir,
                resume_from_checkpoint=first_output_dir / "checkpoint-4",
                print_outputs=True,
            )
            assert returncode == 0, "Second run failed."

            # Case 2: Resuming from checkpoint by specifying a boolean, in this case it should look inside the output
            # directory.
            returncode, _ = runner.run(
                num_cores,
                precision,
                train_batch_size,
                eval_batch_size=eval_batch_size,
                sequence_length=sequence_length,
                tensor_parallel_size=tensor_parallel_size,
                max_steps=max_steps + 10,  # So that it makes more steps since we are restauring from the third run.
                save_steps=save_steps,
                do_eval=do_eval,
                max_eval_samples=max_eval_samples,
                output_dir=second_output_dir,
                print_outputs=True,
            )
            assert returncode == 0, "Third run failed."
