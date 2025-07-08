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
"""Defines the command line related to dealing with the Neuron cache repo."""

from argparse import ArgumentParser

from ...neuron.cache import get_hub_cached_entries, synchronize_hub_cache
from ...neuron.utils.cache_utils import (
    CACHE_REPO_NAME,
    HF_HOME_CACHE_REPO_FILE,
    create_custom_cache_repo,
    set_custom_cache_repo_name_in_hf_home,
)
from ...neuron.utils.require_utils import requires_torch_neuronx
from ...neuron.utils.runner import ExampleRunner
from ...utils import logging
from ..base import BaseOptimumCLICommand, CommandInfo


logger = logging.get_logger()


class CreateCustomCacheRepoCommand(BaseOptimumCLICommand):
    @staticmethod
    def parse_args(parser: ArgumentParser):
        parser.add_argument(
            "-n",
            "--name",
            type=str,
            default=CACHE_REPO_NAME,
            help="The name of the repo that will be used as a remote cache for the compilation files.",
        )
        parser.add_argument(
            "--public",
            action="store_true",
            help="If set, the created repo will be public. By default the cache repo is private.",
        )

    def run(self):
        repo_url = create_custom_cache_repo(repo_id=self.args.name, private=not self.args.public)
        public_or_private = "public" if self.args.public else "private"
        logger.info(f"Neuron cache created on the Hugging Face Hub: {repo_url.repo_id} [{public_or_private}].")
        logger.info(f"Neuron cache name set locally to {repo_url.repo_id} in {HF_HOME_CACHE_REPO_FILE}.")


class SetCustomCacheRepoCommand(BaseOptimumCLICommand):
    @staticmethod
    def parse_args(parser: "ArgumentParser"):
        parser.add_argument("name", type=str, help="The name of the repo to use as remote cache.")

    def run(self):
        set_custom_cache_repo_name_in_hf_home(self.args.name)
        logger.info(f"Neuron cache name set locally to {self.args.name} in {HF_HOME_CACHE_REPO_FILE}.")


class AddToCacheRepoCommand(BaseOptimumCLICommand):
    @staticmethod
    def parse_args(parser: "ArgumentParser"):
        parser.add_argument("-m", "--model", type=str, required=True, help="The name of model or path of the model.")
        parser.add_argument("--task", type=str, required=True, help="The task for which the model should be compiled.")

        # Shapes
        parser.add_argument(
            "--train_batch_size",
            type=int,
            required=True,
            help="The batch size to use during the model compilation for training.",
        )

        parser.add_argument(
            "--eval_batch_size",
            type=int,
            default=None,
            help="The batch size to use during model compilation for evaluation.",
        )

        sequence_length_group = parser.add_mutually_exclusive_group()

        sequence_length_group.add_argument(
            "--sequence_length", type=int, help="The sequence length of the model during compilation."
        )

        seq2seq_sequence_length_group = sequence_length_group.add_argument_group()
        seq2seq_sequence_length_group.add_argument(
            "--encoder_sequence_length",
            type=int,
            help="The sequence length of the encoder part of the model during compilation.",
        )
        seq2seq_sequence_length_group.add_argument(
            "--decoder_sequence_length",
            type=int,
            help="The sequence length of the decoder part of the model during compilation.",
        )

        parser.add_argument(
            "--gradient_accumulation_steps", type=int, default=1, help="The number of gradient accumulation steps.."
        )

        parser.add_argument(
            "--precision",
            choices=["fp", "bf16"],
            type=str,
            required=True,
            help="The precision to use during the model compilation.",
        )
        parser.add_argument(
            "--num_cores",
            choices=list(range(1, 33)),
            type=int,
            required=True,
            help="The number of neuron cores to use during compilation.",
        )
        parser.add_argument(
            "--example_dir", type=str, default=None, help="Path to where the example scripts are stored."
        )
        parser.add_argument(
            "--max_steps", type=int, default=10, help="The maximum number of steps to run compilation for."
        )

    def run(self):
        runner = ExampleRunner(self.args.model, self.args.task, example_dir=self.args.example_dir)
        if self.args.eval_batch_size is None:
            self.args.eval_batch_size = self.args.train_batch_size

        if self.args.sequence_length is not None:
            sequence_length = self.args.sequence_length
        elif self.args.encoder_sequence_length is None and self.args.decoder_sequence_length is None:
            raise ValueError(
                "You need to specify either sequence_length or encoder_sequence_length and decoder_sequence_length"
            )
        elif self.args.encoder_sequence_length is None or self.args.decoder_sequence_length is None:
            raise ValueError("Both the encoder_sequence_length and the decoder_sequence_length must be provided.")
        else:
            sequence_length = [self.args.encoder_sequence_length, self.args.decoder_sequence_length]
        returncode, stdout = runner.run(
            self.args.num_cores,
            self.args.precision,
            self.args.train_batch_size,
            sequence_length,
            do_eval=True,
            eval_batch_size=self.args.eval_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            num_epochs=3,
            max_steps=self.args.max_steps,
            save_steps=self.args.max_steps // 2,
        )
        if returncode != 0:
            raise ValueError(f"Could not add the model to the cache. Full log:\n{stdout}.")


class SynchronizeRepoCommand(BaseOptimumCLICommand):
    @staticmethod
    def parse_args(parser: "ArgumentParser"):
        parser.add_argument("--repo_id", type=str, default=None, help="The name of the repo to use as remote cache.")
        parser.add_argument(
            "--cache_dir", type=str, default=None, help="The cache directory that contains the compilation files."
        )

    @requires_torch_neuronx
    def run(self):
        synchronize_hub_cache(cache_path=self.args.cache_dir, cache_repo_id=self.args.repo_id)


class LookupRepoCommand(BaseOptimumCLICommand):
    @staticmethod
    def parse_args(parser: "ArgumentParser"):
        parser.add_argument(
            "model_id",
            type=str,
            help="The model_id to lookup cached versions for.",
        )
        parser.add_argument(
            "--task",
            type=str,
            default=None,
            help="The optional task to lookup cached versions for models supporting multiple tasks.",
        )
        parser.add_argument("--repo_id", type=str, default=None, help="The name of the repo to use as remote cache.")

    def _list_entries(self):
        entries = get_hub_cached_entries(self.args.model_id, task=self.args.task, cache_repo_id=self.args.repo_id)
        n_entries = len(entries)
        output = f"\n*** {n_entries} entrie(s) found in cache for {self.args.model_id}.***\n\n"
        for entry in entries:
            for key, value in entry.items():
                output += f"\n{key}: {value}"
            output += "\n"
        print(output)

    def run(self):
        self._list_entries()


class CustomCacheRepoCommand(BaseOptimumCLICommand):
    SUBCOMMANDS = (
        CommandInfo(
            name="create",
            help="Create a model repo on the Hugging Face Hub to store Neuron X compilation files.",
            subcommand_class=CreateCustomCacheRepoCommand,
        ),
        CommandInfo(
            name="set",
            help="Set the name of the Neuron cache repo to use locally.",
            subcommand_class=SetCustomCacheRepoCommand,
        ),
        CommandInfo(
            name="add",
            help="Add a model to the cache of your choice (trainium only).",
            subcommand_class=AddToCacheRepoCommand,
        ),
        CommandInfo(
            name="synchronize",
            help="Synchronize the neuronx compiler cache with a hub cache repo.",
            subcommand_class=SynchronizeRepoCommand,
        ),
        CommandInfo(
            name="lookup",
            help="Lookup the neuronx compiler hub cache for the specified model id.",
            subcommand_class=LookupRepoCommand,
        ),
    )
