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
"""Defines the root command line class for optimum-neuron."""

from ...utils import logging
from ..base import BaseOptimumCLICommand, CommandInfo
from .cache import CustomCacheRepoCommand
from .serve import ServeCommand
from .subcommands import ConsolidateCommand


logger = logging.get_logger()
logger.setLevel(logging.INFO)


class NeuronCommand(BaseOptimumCLICommand):
    COMMAND = CommandInfo(name="neuron", help="Optimum Neuron CLI")
    SUBCOMMANDS = (
        CommandInfo(
            name="cache",
            help="Manage the Neuron cache.",
            subcommand_class=CustomCacheRepoCommand,
        ),
        CommandInfo(
            name="consolidate",
            help="Consolidate checkpoints that were produced during a parallel training setting.",
            subcommand_class=ConsolidateCommand,
        ),
        CommandInfo(
            name="serve",
            help="Serve a neuron model.",
            subcommand_class=ServeCommand,
        ),
    )
