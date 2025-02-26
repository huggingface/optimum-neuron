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
import json
import shutil
import time
from pathlib import Path

import pytest
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
)
from transformers.testing_utils import is_staging_test

from optimum.neuron import NeuronSFTConfig, NeuronSFTTrainer, NeuronTrainer, NeuronTrainingArguments
from optimum.neuron.distributed.utils import MODEL_PARALLEL_SHARDS_DIR_NAME
from optimum.neuron.utils import is_neuronx_distributed_available
from optimum.neuron.utils.cache_utils import (
    list_files_in_neuron_cache,
)
from optimum.neuron.utils.testing_utils import is_trainium_test
from optimum.neuron.utils.training_utils import get_model_param_count

from . import DistributedTest
from .utils import (
    MODEL_NAME,
    create_dummy_causal_lm_dataset,
    default_data_collator_for_causal_lm,
    get_tokenizer_and_tiny_llama_model,
)


if is_neuronx_distributed_available():
    from neuronx_distributed.parallel_layers.parallel_state import (
        get_data_parallel_rank,
        get_pipeline_model_parallel_rank,
        get_tensor_model_parallel_rank,
    )


@is_trainium_test
class TestNeuronTrainingUtils(DistributedTest):
    @pytest.fixture(
        scope="class",
        # TODO: enable dp + tp + pp, currently produces communication error between replicas.
        params=[[2, 1, 1], [2, 2, 1], [2, 1, 2]],  # , [32, 2, 2]],
        ids=["dp=2", "tp=2", "pp=2"],  # , "dp=8,tp=pp=2"],
    )
    def parallel_sizes(self, request):
        return request.param

    def test_get_model_param_count(self, parallel_sizes, tmpdir):
        _, tp_size, pp_size = parallel_sizes
        output_dir = Path(tmpdir)

        _, model = get_tokenizer_and_tiny_llama_model()

        target_num_parameters = sum(p.numel() for p in model.parameters())

        args = NeuronTrainingArguments(
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            output_dir=output_dir.as_posix(),
        )
        trainer = NeuronTrainer(args=args, model=model)
        prepared_model = trainer.accelerator.prepare_model(model)
        num_parameters = get_model_param_count(prepared_model)

        assert num_parameters == target_num_parameters


@is_trainium_test
class TestNeuronTrainer(DistributedTest):
    @pytest.fixture(
        scope="class",
        # TODO: enable dp + tp + pp, currently produces communication error between replicas.
        # TODO: Fix pp as well.
        params=[[2, 1, 1], [2, 2, 1]],  # , [2, 1, 2]],  # [8, 2, 2]],
        ids=[
            "dp=2",
            "tp=2",
        ],  # "pp=2"],  # , "dp=4,tp=pp=2"],
    )
    def parallel_sizes(self, request):
        return request.param

    def test_save_checkpoint(self, hub_test, tmpdir, parallel_sizes):
        world_size, tp_size, pp_size = parallel_sizes
        output_dir = Path(tmpdir)

        dp_rank = get_data_parallel_rank()
        tp_rank = get_tensor_model_parallel_rank()
        pp_rank = get_pipeline_model_parallel_rank()

        args = NeuronTrainingArguments(
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            do_train=True,
            do_eval=False,
            per_device_train_batch_size=1,
            save_steps=5,
            max_steps=20,
            output_dir=output_dir.as_posix(),
        )

        tokenizer, model = get_tokenizer_and_tiny_llama_model()
        datasets = create_dummy_causal_lm_dataset(model.config.vocab_size, 120, 1, sequence_length=128)

        trainer = NeuronTrainer(
            args=args,
            model=model,
            tokenizer=tokenizer,
            train_dataset=datasets["train"],
            data_collator=default_data_collator_for_causal_lm,
        )

        trainer.train()

        checkpoint_directories = [f"checkpoint-{k}" for k in range(5, 20, 5)]
        for checkpoint_dir_name in checkpoint_directories:
            # We check that each checkpoint dir exists and contains:
            #   - The model config
            #   - The tokenizer and its config
            #   - The model weights, one file if tp_size = pp_size = 1 or the shards otherwise.
            #   - The optimizer state if tp_size = pp_size = 1 (otherwise the state is in the sharded checkpoints)
            #   - The scheduler state
            #   - The trainer state
            #   - The RNG state
            #   - The training args
            checkpoint_dir = output_dir / checkpoint_dir_name
            assert checkpoint_dir.is_dir()
            assert (checkpoint_dir / "config.json").is_file()
            assert (checkpoint_dir / "tokenizer.json").is_file()
            assert (checkpoint_dir / "tokenizer.model").is_file()
            assert (checkpoint_dir / "tokenizer_config.json").is_file()
            assert (checkpoint_dir / "special_tokens_map.json").is_file()

            if tp_size == pp_size == 1:
                assert (checkpoint_dir / "model.safetensors").is_file()
                assert (checkpoint_dir / "optimizer.pt").is_file()
            else:
                shards_dir = checkpoint_dir / MODEL_PARALLEL_SHARDS_DIR_NAME
                assert shards_dir.is_dir()
                assert (shards_dir / "model").is_dir()
                assert (shards_dir / "optim").is_dir()
                sharded_stem = f"dp_rank_{dp_rank:02d}_tp_rank_{tp_rank:02d}_pp_rank_{pp_rank:02d}"
                assert (shards_dir / "model" / f"{sharded_stem}.pt").is_file()
                assert (shards_dir / "model" / f"{sharded_stem}.pt.tensors").is_dir()
                assert (shards_dir / "optim" / f"{sharded_stem}.pt").is_file()
                assert (shards_dir / "optim" / f"{sharded_stem}.pt.tensors").is_dir()

            assert (checkpoint_dir / "scheduler.pt").is_file()
            assert (checkpoint_dir / "trainer_state.json").is_file()
            for worker_id in range(world_size):
                assert (checkpoint_dir / f"rng_state_{worker_id}.pth").is_file()

            assert (checkpoint_dir / "training_args.bin").is_file()

    @pytest.mark.skip("Maybe merge with test_save_and_resume_from_checkpoint")
    def test_train_and_eval_use_remote_cache(self, hub_test_with_local_cache, tmpdir, parallel_sizes):
        repo_id, local_cache_path = hub_test_with_local_cache
        output_dir = Path(tmpdir)
        _, tp_size, pp_size = parallel_sizes

        # We take a batch size that does not divide the total number of samples.
        num_train_samples = 200
        per_device_train_batch_size = 32

        # We take a batch size that does not divide the total number of samples.
        num_eval_samples = 100
        per_device_eval_batch_size = 16

        tokenizer, model = get_tokenizer_and_tiny_llama_model()
        clone = copy.deepcopy(model)

        datasets = create_dummy_causal_lm_dataset(model.config.vocab_size, num_train_samples, num_eval_samples)

        files_in_repo = HfApi().list_repo_files(repo_id=repo_id)
        files_in_repo = [f for f in files_in_repo if not f.startswith(".")]
        files_in_cache = list_files_in_neuron_cache(local_cache_path, only_relevant_files=True)

        assert files_in_repo == [], "Repo should be empty."
        assert files_in_cache == [], "Cache should be empty."

        args = NeuronTrainingArguments(
            output_dir=(output_dir / "first_run").as_posix(),
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
            tokenizer=tokenizer,
            train_dataset=datasets["train"],
            eval_dataset=datasets["eval"],
            data_collator=default_data_collator_for_causal_lm,
        )
        start = time.time()
        trainer.train()
        end = time.time()
        first_training_duration = end - start

        files_in_repo = HfApi().list_repo_files(repo_id=repo_id)
        files_in_repo = [f for f in files_in_repo if not f.startswith(".")]
        files_in_cache = list_files_in_neuron_cache(local_cache_path, only_relevant_files=True)

        assert files_in_repo != [], "Repo should not be empty after first training."
        assert files_in_cache != [], "Cache should not be empty after first training."

        shutil.rmtree(local_cache_path)

        new_files_in_repo = HfApi().list_repo_files(repo_id=repo_id)
        new_files_in_repo = [f for f in new_files_in_repo if not f.startswith(".")]
        new_files_in_cache = list_files_in_neuron_cache(local_cache_path, only_relevant_files=True)

        assert new_files_in_repo != [], "Repo should not be empty."
        assert new_files_in_cache == [], "Cache should be empty."

        args = NeuronTrainingArguments(
            output_dir=(output_dir / "second_run").as_posix(),
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
            tokenizer=tokenizer,
            train_dataset=datasets["train"],
            eval_dataset=datasets["eval"],
            data_collator=default_data_collator_for_causal_lm,
        )
        start = time.time()
        trainer.train()
        end = time.time()
        second_training_duration = end - start

        last_files_in_repo = HfApi().list_repo_files(repo_id=repo_id)
        last_files_in_repo = [f for f in last_files_in_repo if not f.startswith(".")]
        last_files_in_cache = list_files_in_neuron_cache(local_cache_path, only_relevant_files=True)

        # TODO: investigate that, not urgent.
        assert files_in_repo == last_files_in_repo, "No file should have been added to the Hub after first training."
        assert files_in_cache == last_files_in_cache, (
            "No file should have been added to the cache after first training."
        )
        assert second_training_duration < first_training_duration, (
            "Second training should be faster because cached graphs can be used."
        )

    @pytest.mark.skip("Test in later release")
    def test_save_and_resume_from_checkpoint(self, parallel_sizes, tmpdir):
        tmpdir = Path(tmpdir)
        _, tp_size, pp_size = parallel_sizes
        train_batch_size = 2
        eval_batch_size = 2
        max_steps = 10
        do_eval = True
        max_train_samples = 100
        max_eval_samples = 16

        tokenizer, _ = get_tokenizer_and_tiny_llama_model()
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
                logging_steps=2,
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

        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint.as_posix())
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


@is_trainium_test
class TestNeuronSFTTrainer(DistributedTest):
    @pytest.fixture(
        scope="class",
        params=[[2, 1, 1], [2, 2, 1]],
        ids=["dp=2", "tp=2"],
    )
    def parallel_sizes(self, request):
        return request.param

    def _test_sft_trainer(self, parallel_sizes, tmpdir, packing):
        _, tp_size, pp_size = parallel_sizes

        output_dir = Path(tmpdir)

        dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

        def format_dolly(sample):
            instruction = f"### Instruction\n{sample['instruction']}"
            context = f"### Context\n{sample['context']}" if len(sample["context"]) > 0 else None
            response = f"### Answer\n{sample['response']}"
            # join all the parts together
            prompt = "\n\n".join([i for i in [instruction, context, response] if i is not None])
            if packing:
                return prompt
            return [prompt]

        tokenizer, model = get_tokenizer_and_tiny_llama_model()
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # to prevent warnings

        args = NeuronTrainingArguments(
            output_dir=output_dir,
            do_train=True,
            max_steps=10,
            per_device_train_batch_size=1,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            bf16=True,
            logging_steps=1,
        )
        args = args.to_dict()
        sft_config = NeuronSFTConfig(
            # Using a small sequence-length since we are not validating the outputs.
            max_seq_length=128,
            packing=packing,
            dataset_num_proc=1,
            **args,
        )

        # Create Trainer instance
        trainer = NeuronSFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            formatting_func=format_dolly,
            args=sft_config,
        )

        trainer.train()

    def test_without_packing(self, parallel_sizes, tmpdir):
        return self._test_sft_trainer(parallel_sizes, tmpdir, False)

    def test_with_packing(self, parallel_sizes, tmpdir):
        return self._test_sft_trainer(parallel_sizes, tmpdir, True)


@is_trainium_test
@is_staging_test
def test_dummy_staging_test():
    pass
