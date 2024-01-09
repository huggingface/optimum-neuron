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

import pytest
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from optimum.neuron.training_args import NeuronTrainingArguments
from optimum.neuron.utils.testing_utils import is_trainium_test

from .distributed import DistributedTest


_TINY_BERT_MODEL_NAME = "hf-internal-testing/tiny-random-bert"
MODEL_NAME = "michaelbenayoun/llama-2-tiny-16layers-random"


@is_trainium_test
class TestDistributedTraining(DistributedTest):
    CACHE_REPO_NAME = "optimum-internal-testing/optimum-neuron-cache-for-testing"

    @pytest.fixture(
        scope="class",
        params=[[2, 1, 1], [2, 2, 1], [2, 1, 2], [16, 2, 2]],
        ids=["dp=2", "tp=2", "pp=2", "dp=4,tp=pp=2"],
    )
    def parallel_sizes(self, request):
        return request.param

    def test_tp_save_and_resume_from_checkpoint(self, parallel_sizes, tmpdir):
        from optimum.neuron.trainers import NeuronTrainer

        tmpdir = Path(tmpdir)
        _, tp_size, pp_size = parallel_sizes
        train_batch_size = 2
        eval_batch_size = 2
        max_steps = 10
        do_eval = True
        max_eval_samples = 16

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token

        def create_training_args(output_dir, resume_from_checkpoint=None, max_steps=max_steps):
            args = NeuronTrainingArguments(
                tensor_parallel_size=tp_size,
                pipeline_parallel_size=pp_size,
                bf16=True,
                per_device_train_batch_size=train_batch_size,
                per_device_eval_batch_size=eval_batch_size,
                max_steps=max_steps,
                do_eval=do_eval,
                output_dir=output_dir,
                resume_from_checkpoint=resume_from_checkpoint,
                skip_cache_push=True,
            )
            return args

        def create_model():
            config = AutoConfig.from_pretrained(MODEL_NAME)
            config.num_hidden_layers = 2
            config.num_attention_heads = 2
            config.num_key_value_heads = 2
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
            eval_dataset = raw_datasets["validation"]
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        trainer = NeuronTrainer(
            model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, tokenizer=tokenizer
        )

        trainer.train()
        trainer.evaluate()

        # Case 1: Resuming from checkpoint by specifying a checkpoint directory.
        second_output_dir = tmpdir / "second_run"
        resume_from_checkpoint = first_output_dir / "checkpoint-4"
        args = create_training_args(second_output_dir, resume_from_checkpoint=resume_from_checkpoint)
        model = create_model()

        trainer = NeuronTrainer(
            model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, tokenizer=tokenizer
        )

        trainer.train()
        trainer.evaluate()

        # Case 2: Resuming from checkpoint by specifying an output_dir with checkpoints.
        # max_steps + 10 to do a some training steps than the previous run.
        args = create_training_args(second_output_dir, max_steps=max_steps + 10)
        model = create_model()

        trainer = NeuronTrainer(
            model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, tokenizer=tokenizer
        )

        trainer.train()
        trainer.evaluate()
