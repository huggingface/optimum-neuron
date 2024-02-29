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

import json
from pathlib import Path

import pytest
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from optimum.neuron.training_args import NeuronTrainingArguments
from optimum.neuron.utils.testing_utils import is_trainium_test

from .distributed import DistributedTest


MODEL_NAME = "michaelbenayoun/llama-2-tiny-16layers-random"


@is_trainium_test
class TestDistributedTraining(DistributedTest):
    CACHE_REPO_NAME = "optimum-internal-testing/optimum-neuron-cache-for-testing"

    @pytest.fixture(
        scope="class",
        # params=[[2, 1, 1], [2, 2, 1], [2, 1, 2]],
        # ids=["dp=2", "tp=2", "pp=2"],
        # TODO: fix pp=2 case since it is flaky and can hang.
        params=[[2, 1, 1], [2, 2, 1]],
        ids=["dp=2", "tp=2"],
    )
    def parallel_sizes(self, request):
        return request.param

    def test_save_and_resume_from_checkpoint(self, parallel_sizes, tmpdir):
        from optimum.neuron.trainers import NeuronTrainer

        tmpdir = Path(tmpdir)
        _, tp_size, pp_size = parallel_sizes
        train_batch_size = 2
        eval_batch_size = 2
        max_steps = 10
        do_eval = True
        max_train_samples = 100
        max_eval_samples = 16

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token

        def create_training_args(output_dir, resume_from_checkpoint=None, max_steps=max_steps):
            if isinstance(output_dir, Path):
                output_dir = output_dir.as_posix()
            if isinstance(resume_from_checkpoint, Path):
                resume_from_checkpoint = resume_from_checkpoint.as_posix()
            args = NeuronTrainingArguments(
                tensor_parallel_size=tp_size,
                pipeline_parallel_size=pp_size,
                bf16=True,
                per_device_train_batch_size=train_batch_size,
                per_device_eval_batch_size=eval_batch_size,
                max_steps=max_steps,
                logging_steps=1,
                save_steps=5,
                do_eval=do_eval,
                output_dir=output_dir,
                resume_from_checkpoint=resume_from_checkpoint,
                skip_cache_push=False,
            )
            return args

        def create_model():
            config = AutoConfig.from_pretrained(MODEL_NAME)
            config.num_hidden_layers = 2 * max(1, pp_size)
            config.num_attention_heads = 2
            config.num_key_value_heads = 2
            config.problem_type = "single_label_classification"
            # config.use_cache = False
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, config=config, ignore_mismatched_sizes=True
            )
            return model

        # First run setting.
        first_output_dir = tmpdir / "first_run"
        args = create_training_args(first_output_dir)
        model = create_model()

        # Dataset preprocessing
        raw_datasets = load_dataset("glue", "sst2")
        sentence1_key = "sentence"
        sentence2_key = None
        label_to_id = None
        max_seq_length = 32
        padding = "max_length"

        def preprocess_function(examples):
            # Tokenize the texts
            args = (
                (examples[sentence1_key],)
                if sentence2_key is None
                else (examples[sentence1_key], examples[sentence2_key])
            )
            result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

            # Map labels to IDs (not necessary for GLUE tasks)
            if label_to_id is not None and "label" in examples:
                result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
            return result

        with args.main_process_first(desc="dataset map pre-processing"):
            raw_datasets = raw_datasets.map(preprocess_function, batched=True)
            train_dataset = raw_datasets["train"]
            train_dataset = train_dataset.select(range(max_train_samples))
            eval_dataset = raw_datasets["validation"]
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        trainer = NeuronTrainer(
            model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, tokenizer=tokenizer
        )

        train_result = trainer.train()
        trainer.evaluate()
        trainer.save_metrics("train", train_result.metrics)

        with open(first_output_dir / "train_results.json") as fp:
            first_training_report = json.load(fp)

        # Case 1: Resuming from checkpoint by specifying a checkpoint directory.
        second_output_dir = tmpdir / "second_run"
        resume_from_checkpoint = first_output_dir / "checkpoint-5"
        args = create_training_args(second_output_dir, resume_from_checkpoint=resume_from_checkpoint)
        model = create_model()
        trainer = NeuronTrainer(
            model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, tokenizer=tokenizer
        )

        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        trainer.evaluate()
        trainer.save_metrics("train", train_result.metrics)

        with open(first_output_dir / "train_results.json") as fp:
            second_training_report = json.load(fp)

        assert first_training_report["train_loss"] == second_training_report["train_loss"]

        # Case 2: Resuming from checkpoint by specifying an output_dir with checkpoints.
        # max_steps + 10 to do a some training steps than the previous run.
        second_output_dir = first_output_dir
        args = create_training_args(second_output_dir, max_steps=max_steps + 10)
        model = create_model()

        trainer = NeuronTrainer(
            model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, tokenizer=tokenizer
        )

        trainer.train(resume_from_checkpoint=True)
        trainer.evaluate()
