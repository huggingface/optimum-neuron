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

import copy
import json
import os
import random
import subprocess
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pytest
from huggingface_hub import HfApi, delete_repo
from huggingface_hub.utils import RepositoryNotFoundError
from transformers import BertConfig, BertModel, BertTokenizer
from transformers.testing_utils import is_staging_test

from optimum.neuron.trainers import NeuronTrainer
from optimum.neuron.training_args import NeuronTrainingArguments
from optimum.neuron.utils.cache_utils import (
    get_neuron_cache_path,
    list_files_in_neuron_cache,
    remove_ip_adress_from_path,
    set_neuron_cache_path,
)
from optimum.neuron.utils.testing_utils import is_trainium_test
from optimum.utils.testing_utils import USER

from .utils import (
    StagingTestMixin,
    TrainiumTestMixin,
    create_dummy_dataset,
    create_dummy_text_classification_dataset,
    create_tiny_pretrained_model,
    get_random_string,
)


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
            last_files_in_cache = [remove_ip_adress_from_path(p) for p in last_files_in_cache]
            # TODO: investigate that, not urgent.
            # self.assertListEqual(
            #     files_in_repo, last_files_in_repo, "No file should have been added to the Hub after first training."
            # )
            # self.assertListEqual(
            #     files_in_cache,
            #     last_files_in_cache,
            #     "No file should have been added to the cache after first training.",
            # )

            self.assertTrue(
                second_training_duration < first_training_duration,
                "Second training should be faster because cached graphs can be used.",
            )

    # Not using a fixture because they do not work with unittest.TestCase.
    def create_model_and_dataset_on_staging_hub(self):
        dataset_name = f"{USER}/random-text-dataset"
        model_name = "tiny-bert"

        self.remove_model_and_dataset_on_staging_hub(dataset_name, f"{USER}/{model_name}")

        dummy_dataset = create_dummy_text_classification_dataset(1000, 100, 100)
        dummy_dataset.push_to_hub(dataset_name)

        with TemporaryDirectory() as tmpdirname:
            vocab_size = 1024
            config = BertConfig(
                vocab_size=vocab_size,
                hidden_size=128,
                num_hidden_layers=2,
                num_attention_heads=2,
                intermediate_size=256,
            )

            special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

            vocab_path = Path(tmpdirname) / "vocab.txt"

            with open(vocab_path, "w") as fp:
                tokens = [get_random_string(random.randint(1, 5)) for _ in range(vocab_size)]
                fp.write("\n".join(special_tokens + tokens))

            tokenizer = BertTokenizer(vocab_path.as_posix())
            model = BertModel(config)

            tokenizer.push_to_hub(model_name)
            model.push_to_hub(model_name)

        model_name = f"{USER}/{model_name}"

        return dataset_name, model_name

    def remove_model_and_dataset_on_staging_hub(self, dataset_name: str, model_name: str):
        try:
            delete_repo(repo_id=dataset_name, repo_type="dataset")
        except RepositoryNotFoundError:
            pass
        try:
            delete_repo(repo_id=model_name, repo_type="model")
        except RepositoryNotFoundError:
            pass

    @unittest.skip("Need to understand how to work with staging and datasets")
    def test_train_and_eval_multiple_workers(self):
        os.environ["CUSTOM_CACHE_REPO"] = self.CUSTOM_PRIVATE_CACHE_REPO

        dataset_name, model_name = self.create_model_and_dataset_on_staging_hub()

        with TemporaryDirectory() as tmpdirname:
            set_neuron_cache_path(tmpdirname)

            files_in_repo = HfApi().list_repo_files(repo_id=self.CUSTOM_PRIVATE_CACHE_REPO)
            files_in_repo = [f for f in files_in_repo if not f.startswith(".")]
            files_in_cache = list_files_in_neuron_cache(get_neuron_cache_path(), only_relevant_files=True)
            self.assertListEqual(files_in_repo, [], "Repo should be empty.")
            self.assertListEqual(files_in_cache, [], "Cache should be empty.")

            cmd = [
                "torchrun",
                "--nproc_per_node=2",
                "examples/text-classification/run_glue.py",
                f"--model_name_or_path={model_name}",
                f"--dataset_name={dataset_name}",
                "--per_device_train_batch_size=16",
                "--per_device_eval_batch_size=16",
                f"--output_dir={tmpdirname}",
                "--save_strategy=steps",
                "--save_steps=10",
                "--max_steps=100",
                "--do_train",
                "--do_eval",
                "--bf16",
            ]

            start = time.time()
            proc = subprocess.Popen(cmd, stderr=subprocess.PIPE)
            end = time.time()
            first_training_duration = end - start

            _, stderr = proc.communicate()
            stderr = stderr.decode("utf-8")
            if stderr:
                print(stderr)

            self.assertEqual(proc.returncode, 0, "The first torchrun training command failed.")

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

            cmd = [
                "torchrun",
                "--nproc_per_node=2",
                "examples/text-classification/run_glue.py",
                f"--model_name_or_path={model_name}",
                f"--dataset_name={dataset_name}",
                "--per_device_train_batch_size=16",
                "--per_device_eval_batch_size=16",
                f"--output_dir={tmpdirname}",
                "--save_strategy=steps",
                "--save_steps=10",
                "--max_steps=100",
                "--do_train",
                "--do_eval",
                "--bf16",
            ]

            start = time.time()
            proc = subprocess.Popen(cmd, stderr=subprocess.PIPE)
            end = time.time()
            second_training_duration = end - start

            _, stderr = proc.communicate()
            stderr = stderr.decode("utf-8")
            if stderr:
                print(stderr)

            self.assertEqual(proc.returncode, 0, "The second torchrun training command failed.")

            last_files_in_repo = HfApi().list_repo_files(repo_id=self.CUSTOM_PRIVATE_CACHE_REPO)
            last_files_in_repo = [f for f in last_files_in_repo if not f.startswith(".")]
            last_files_in_cache = list_files_in_neuron_cache(get_neuron_cache_path(), only_relevant_files=True)
            last_files_in_cache = [remove_ip_adress_from_path(p) for p in last_files_in_cache]
            # TODO: investigate that, not urgent.
            # self.assertListEqual(
            #     files_in_repo, last_files_in_repo, "No file should have been added to the Hub after first training."
            # )
            # self.assertListEqual(
            #     files_in_cache,
            #     last_files_in_cache,
            #     "No file should have been added to the cache after first training.",
            # )

            self.assertTrue(
                second_training_duration < first_training_duration,
                "Second training should be faster because cached graphs can be used.",
            )

        self.remove_model_and_dataset_on_staging_hub(dataset_name, model_name)


@is_trainium_test
class NeuronTrainerTestCase(TrainiumTestMixin, TestCase):
    def _test_training_with_fsdp_mode(self, fsdp_mode: str):
        model_name = "prajjwal1/bert-tiny"
        task_name = "sst2"

        with TemporaryDirectory() as tmpdirname:
            set_neuron_cache_path(tmpdirname)
            output_1 = Path(tmpdirname) / "out_1"
            fsdp_cmd = [
                "torchrun",
                "--nproc_per_node=2",
                "examples/text-classification/run_glue.py",
                f"--model_name_or_path={model_name}",
                f"--task_name={task_name}",
                "--per_device_train_batch_size=16",
                "--per_device_eval_batch_size=16",
                f"--output_dir={output_1}",
                "--save_strategy=steps",
                "--save_steps=10",
                "--max_steps=100",
                "--do_train",
                # "--do_eval",
                "--bf16",
                f"--fsdp={fsdp_mode}",
            ]

            proc = subprocess.Popen(fsdp_cmd)
            proc.wait()

            checkpoint = output_1 / "checkpoint-50"
            output_2 = Path(tmpdirname) / "out_2"

            resume_fsdp_cmd = [
                "torchrun",
                "--nproc_per_node=2",
                "examples/text-classification/run_glue.py",
                f"--model_name_or_path={model_name}",
                f"--task_name={task_name}",
                "--per_device_train_batch_size=16",
                "--per_device_eval_batch_size=16",
                f"--output_dir={output_2}",
                "--save_strategy=steps",
                "--save_steps=10",
                "--max_steps=100",
                "--do_train",
                # "--do_eval",
                "--bf16",
                f"--resume_from_checkpoint={checkpoint}",
                f"--fsdp={fsdp_mode}",
            ]

            proc = subprocess.Popen(resume_fsdp_cmd)
            proc.wait()

            training_fsdp_metrics = {}
            with open(output_1 / "all_results.json") as fp:
                training_fsdp_metrics = json.load(fp)

            resume_training_fsdp_metrics = {}
            with open(output_2 / "all_results.json") as fp:
                resume_training_fsdp_metrics = json.load(fp)

            print(training_fsdp_metrics)
            print(resume_training_fsdp_metrics)
            # self.assertEqual(training_fsdp_metrics["eval_loss"], resume_training_fsdp_metrics["eval_loss"])
            # self.assertEqual(training_fsdp_metrics["eval_accuracy"], resume_training_fsdp_metrics["eval_accuracy"])

            output_3 = Path(tmpdirname) / "out_3"

            cmd = [
                "torchrun",
                "--nproc_per_node=2",
                "examples/text-classification/run_glue.py",
                f"--model_name_or_path={model_name}",
                f"--task_name={task_name}",
                "--per_device_train_batch_size=16",
                "--per_device_eval_batch_size=16",
                f"--output_dir={output_3}",
                "--save_strategy=steps",
                "--save_steps=10",
                "--max_steps=100",
                "--do_train",
                # "--do_eval",
                "--bf16",
                f"--fsdp={fsdp_mode}",
            ]

            proc = subprocess.Popen(cmd)
            proc.wait()

            regular_training_metrics = {}
            with open(output_3 / "all_results.json") as fp:
                regular_training_metrics = json.load(fp)

            self.assertEqual(training_fsdp_metrics["train_loss"], regular_training_metrics["train_loss"])
            # self.assertEqual(training_fsdp_metrics["eval_loss"], regular_training_metrics["eval_loss"])
            # self.assertEqual(training_fsdp_metrics["eval_accuracy"], regular_training_metrics["eval_accuracy"])

    @pytest.mark.skip("FSDP not supported yet")
    def test_training_with_fsdp_full_shard(self):
        return self._test_training_with_fsdp_mode("full_shard")

    @pytest.mark.skip("FSDP not supported yet")
    def test_training_with_fsdp_shard_grad_op(self):
        return self._test_training_with_fsdp_mode("shard_grad_op")

    @pytest.mark.skip("FSDP not supported yet")
    def test_training_with_fsdp_no_shard(self):
        return self._test_training_with_fsdp_mode("no_shard")

    @pytest.mark.skip("FSDP not supported yet")
    def test_training_with_fsdp_offload(self):
        return self._test_training_with_fsdp_mode("offload")

    @pytest.mark.skip("FSDP not supported yet")
    def test_training_with_fsdp_auto_wrap(self):
        return self._test_training_with_fsdp_mode("auto_wrap")
