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
"""Defines the command line related to dealing with the Trainium cache repo."""

from typing import TYPE_CHECKING

from ...neuron.utils.cache_utils import (
    CACHE_REPO_NAME,
    HF_HOME_CACHE_REPO_FILE,
    create_custom_cache_repo,
    set_custom_cache_repo_name_in_hf_home,
)
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
        logger.info(f"Trainium cache created on the Hugging Face Hub: {repo_url.repo_id} [{public_or_private}].")
        logger.info(f"Trainium cache name set locally to {repo_url.repo_id} in {HF_HOME_CACHE_REPO_FILE}.")


class SetCustomCacheRepoCommand(BaseOptimumCLICommand):
    @staticmethod
    def parse_args(parser: "ArgumentParser"):
        parser.add_argument(
            "-n", "--name", type=str, required=True, help="The name of the repo that to use as remote cache."
        )

    def run(self):
        set_custom_cache_repo_name_in_hf_home(self.args.name)
        logger.info(f"Trainium cache name set locally to {self.args.name} in {HF_HOME_CACHE_REPO_FILE}.")


class CustomCacheRepoCommand(BaseOptimumCLICommand):
    SUBCOMMANDS = (
        CommandInfo(
            name="create",
            help="Create a model repo on the Hugging Face Hub to store Neuron X compilation files.",
            subcommand_class=CreateCustomCacheRepoCommand,
        ),
        CommandInfo(
            name="set",
            help="Set the name of the Trainium cache repo to use locally.",
            subcommand_class=SetCustomCacheRepoCommand,
        ),
    )
