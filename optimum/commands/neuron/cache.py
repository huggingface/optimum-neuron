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

from ...neuron.cache.hub_cache import select_hub_cached_entries, synchronize_hub_cache
from ...neuron.utils.cache_utils import (
    CACHE_REPO_NAME,
    HF_HOME_CACHE_REPO_FILE,
    create_custom_cache_repo,
    set_custom_cache_repo_name_in_hf_home,
)
from ...neuron.utils.import_utils import is_package_available
from ...neuron.utils.instance import SUPPORTED_INSTANCE_TYPES
from ...neuron.utils.require_utils import requires_torch_neuronx
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
        parser.add_argument(
            "--instance_type",
            type=str,
            choices=SUPPORTED_INSTANCE_TYPES,
            help=f"Only look for cached models for the specified instance type (e.g. {SUPPORTED_INSTANCE_TYPES}).",
        )
        parser.add_argument(
            "--dtype",
            type=str,
            choices=["bfloat16", "float16"],
            help="Only look for cached models for the specified `torch.dtype`.",
        )
        parser.add_argument(
            "--tensor_parallel_size",
            type=int,
            help="Only look for cached models with the specified tensor parallel size.",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            help="Only look for cached models supporting at least the specified batch size.",
        )
        parser.add_argument(
            "--sequence_length",
            type=int,
            help="Only look for cached models supporting at least the specified sequence length.",
        )
        parser.add_argument("--repo_id", type=str, default=None, help="The name of the repo to use as remote cache.")

    def _list_entries(self):
        entries = select_hub_cached_entries(
            self.args.model_id,
            task=self.args.task,
            cache_repo_id=self.args.repo_id,
            instance_type=self.args.instance_type,
            batch_size=self.args.batch_size,
            sequence_length=self.args.sequence_length,
            tensor_parallel_size=self.args.tensor_parallel_size,
            torch_dtype=self.args.dtype,
        )
        n_entries = len(entries)
        if n_entries == 0:
            print(f"No cached entries found for {self.args.model_id}.")
            return
        # Prepare output table data
        title = f"Cached entries for {self.args.model_id}"
        columns = ["batch size", "sequence length", "tensor parallel", "dtype", "instance type"]
        rows = []
        for entry in entries:
            rows.append(
                (
                    str(entry["batch_size"]),
                    str(entry["sequence_length"]),
                    str(entry.get("tp_degree", entry.get("tensor_parallel_size"))),
                    str(entry["dtype"]),
                    str(entry["target"]),
                )
            )
        # Remove duplicates (might happen if the same arch was compiled several times with different models and sync'ed afterwards)
        rows = list(set(rows))
        # Sort by tensor parallel size, then batch size, sequence length, dtype
        rows = sorted(rows, key=lambda x: (int(x[2]), int(x[0]), int(x[1]), x[3]))
        if is_package_available("rich", "14.1.0"):
            from rich.console import Console
            from rich.table import Table

            table = Table(title=title)
            for column in columns:
                table.add_column(column, justify="center", no_wrap=True)
            for row in rows:
                table.add_row(*row)
            Console().print(table)
        else:
            print(title)
            row_format = "{:^16}" * len(columns)
            print(row_format.format(*columns))
            for row in rows:
                print(row_format.format(*row))

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
            name="synchronize",
            help="Synchronize the neuronx compiler cache with a hub cache repo.",
            subcommand_class=SynchronizeRepoCommand,
        ),
        CommandInfo(
            name="lookup",
            help="Lookup the neuronx compiler hub cache for the specified model id. Tip: install rich for a nicer display",
            subcommand_class=LookupRepoCommand,
        ),
    )
