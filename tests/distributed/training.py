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

from pathlib import Path
from tempfile import TemporaryDirectory

from optimum.neuron.utils.runner import ExampleRunner


_TINY_BERT_MODEL_NAME = "hf-internal-testing/tiny-random-bert"


def test_tp_save_and_resume_from_checkpoint():
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
        assert returncode != 0, "First run failed."

        # Case 1: Resuming from checkpoint by specifying a directory that contains many checkpoints.
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
            resume_from_checkpoint=first_output_dir,
            print_outputs=True,
        )
        assert returncode != 0, "Second run failed."

        # Case 2: Resuming from checkpoint by specifying a checkpoint directory.
        third_output_dir = output_dir / "third_run"
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
            output_dir=third_output_dir,
            resume_from_checkpoint=first_output_dir / "checkpoint-4",
            print_outputs=True,
        )
        assert returncode != 0, "Third run failed."

        # Case 3: Resuming from checkpoint by specifying a boolean, in this case it should look inside the output
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
            output_dir=third_output_dir,
            resume_from_checkpoint=True,
            print_outputs=True,
        )
        assert returncode != 0, "Fourth run failed."
