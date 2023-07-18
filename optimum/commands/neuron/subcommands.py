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
"""Defines the subcommands for the `optimum-cli neuron` command."""

from typing import TYPE_CHECKING

from ...neuron.distributed import consolidate_tensor_parallel_checkpoints_to_unified_checkpoint
from ...utils import logging
from ..base import BaseOptimumCLICommand


logger = logging.get_logger()


if TYPE_CHECKING:
    from argparse import ArgumentParser


class ConsolidateCommand(BaseOptimumCLICommand):
    @staticmethod
    def parse_args(parser: "ArgumentParser"):
        parser.add_argument(
            "checkpoint_dir",
            type=str,
            help="The path to the directory containing the checkpoints.",
        )
        parser.add_argument(
            "output_dir",
            type=str,
            help="The path to the output directory containing the consolidated checkpoint.",
        )
        parser.add_argument(
            "-f",
            "--format",
            type=str,
            choices=["pytorch", "safetensors"],
            default="safetensors",
            help="The format used to save the consolidated checkpoint.",
        )

    def run(self):
        checkpoint_format = "safetensors" if self.args.format == "safetensors" else "pytorch"
        logger.info(f"Consolidating checkpoints from {self.args.checkpoint_dir} to the {checkpoint_format} format...")
        consolidate_tensor_parallel_checkpoints_to_unified_checkpoint(
            self.args.checkpoint_dir,
            self.args.output_dir,
            save_format=self.args.format,
        )
        logger.info(f"Consolidated checkpoint saved at {self.args.output_dir}")
