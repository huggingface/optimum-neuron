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

from typing import TYPE_CHECKING

from ...neuron.utils.cache_utils import (
    CACHE_REPO_NAME,
    HF_HOME_CACHE_REPO_FILE,
    create_custom_cache_repo,
    list_in_registry,
    load_custom_cache_repo_name_from_hf_home,
    set_custom_cache_repo_name_in_hf_home,
)
from ...neuron.utils.runner import ExampleRunner
from ...utils import logging
from ..base import BaseOptimumCLICommand, CommandInfo


if TYPE_CHECKING:
    from argparse import ArgumentParser


logger = logging.get_logger()


class CreateCustomCacheRepoCommand(BaseOptimumCLICommand):
    @staticmethod
    def parse_args(parser: "ArgumentParser"):
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
            "--max_steps", type=int, default=200, help="The maximum number of steps to run compilation for."
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
        runner.run(
            self.args.num_cores,
            self.args.precision,
            self.args.train_batch_size,
            sequence_length,
            do_eval=True,
            eval_batch_size=self.args.eval_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            num_epochs=3,
            max_steps=self.args.max_steps,
            save_steps=10,
        )


class ListRepoCommand(BaseOptimumCLICommand):
    @staticmethod
    def parse_args(parser: "ArgumentParser"):
        parser.add_argument(
            "name",
            type=str,
            nargs="?",
            default=None,
            help="The name of the repo to list. Will use the locally saved cache repo if left unspecified.",
        )
        parser.add_argument(
            "-m",
            "--model",
            type=str,
            default=None,
            help="The model name or path of the model to consider. If left unspecified, will list all available models.",
        )
        parser.add_argument(
            "-v",
            "--version",
            type=str,
            default=None,
            help=(
                "The version of the Neuron X Compiler to consider. Will list all available versions if left "
                "unspecified."
            ),
        )

    def run(self):
        if self.args.name is None:
            custom_cache_repo_name = load_custom_cache_repo_name_from_hf_home()
            if custom_cache_repo_name is None:
                raise ValueError("No custom cache repo was set locally so you need to specify a cache repo name.")
            self.args.name = custom_cache_repo_name

        entries = list_in_registry(
            self.args.name, model_name_or_path_or_hash=self.args.model, neuron_compiler_version=self.args.version
        )
        if not entries:
            entries = ["Nothing was found."]
        line = "\n" + "=" * 50 + "\n"
        result = line.join(entries)

        print(f"\n*** Repo id: {self.args.name} ***\n\n{result}")


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
            help="Add a model to the cache of your choice.",
            subcommand_class=AddToCacheRepoCommand,
        ),
        CommandInfo(
            name="list",
            help="List models in a cache repo.",
            subcommand_class=ListRepoCommand,
        ),
    )
