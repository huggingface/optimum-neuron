# Copyright 2023 The HuggingFace Team. All rights reserved.
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
"""Defines the command line for the export with Neuronx compiler."""

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ...exporters import TasksManager
from ...utils import is_diffusers_available
from ..base import BaseOptimumCLICommand, CommandInfo


if is_diffusers_available():
    # Mandatory for applying optimized attention score of Stable Diffusion
    import os

    os.environ["NEURON_FUSE_SOFTMAX"] = "1"

if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace, _SubParsersAction


def parse_args_neuronx(parser: "ArgumentParser"):
    required_group = parser.add_argument_group("Required arguments")
    required_group.add_argument(
        "-m", "--model", type=str, required=True, help="Model ID on huggingface.co or path on disk to load model from."
    )
    required_group.add_argument(
        "output",
        type=Path,
        help="Path indicating the directory where to store generated Neuronx compiled TorchScript model.",
    )

    optional_group = parser.add_argument_group("Optional arguments")
    optional_group.add_argument(
        "--task",
        default="auto",
        help=(
            "The task to export the model for. If not specified, the task will be auto-inferred based on the model. Available tasks depend on the model, but are among:"
            f" {str(list(TasksManager._TRANSFORMERS_TASKS_TO_MODEL_LOADERS.keys()) + list(TasksManager._DIFFUSERS_TASKS_TO_MODEL_LOADERS.keys()))}."
        ),
    )
    optional_group.add_argument(
        "--atol",
        type=float,
        default=None,
        help="If specified, the absolute difference tolerance when validating the model. Otherwise, the default atol for the model will be used.",
    )
    optional_group.add_argument("--cache_dir", type=str, default=None, help="Path indicating where to store cache.")
    optional_group.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow to use custom code for the modeling hosted in the model repository. This option should only be set for repositories you trust and in which you have read the code, as it will execute on your local machine arbitrary code present in the model repository.",
    )
    optional_group.add_argument(
        "--disable-validation",
        action="store_true",
        help="Whether to disable the validation of inference on neuron device compared to the outputs of original PyTorch model on CPU.",
    )
    optional_group.add_argument(
        "--auto_cast",
        type=str,
        default=None,
        choices=["none", "matmul", "all"],
        help='Whether to cast operations from FP32 to lower precision to speed up the inference. Can be `"none"`, `"matmul"` or `"all"`.',
    )
    optional_group.add_argument(
        "--auto_cast_type",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "tf32"],
        help='The data type to cast FP32 operations to when auto-cast mode is enabled. Can be `"bf16"`, `"fp16"` or `"tf32"`.',
    )
    optional_group.add_argument(
        "--dynamic-batch-size",
        action="store_true",
        help="Enable dynamic batch size for neuron compiled model. If this option is enabled, the input batch size can be a multiple of the batch size during the compilation, but it comes with a potential tradeoff in terms of latency.",
    )

    input_group = parser.add_argument_group("Input shapes")
    doc_input = "that the Neuronx-cc compiler exported model will be able to take as input."
    input_group.add_argument(
        "--batch_size",
        type=int,
        help=f"Batch size {doc_input}",
    )
    input_group.add_argument(
        "--sequence_length",
        type=int,
        help=f"Sequence length {doc_input}",
    )
    input_group.add_argument(
        "--num_choices",
        type=int,
        help=f"Only for the multiple-choice task. Num choices {doc_input}",
    )
    input_group.add_argument(
        "--num_channels",
        type=int,
        help=f"Image tasks only. Number of channels {doc_input}",
    )
    input_group.add_argument(
        "--width",
        type=int,
        help=f"Image tasks only. Width {doc_input}",
    )
    input_group.add_argument(
        "--height",
        type=int,
        help=f"Image tasks only. Height {doc_input}",
    )
    input_group.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help=f"Stable diffusion only. Number of images per prompt {doc_input}",
    )


class NeuronxExportCommand(BaseOptimumCLICommand):
    COMMAND = CommandInfo(name="neuron", help="Export PyTorch models to Neuronx compiled TorchScript models.")

    def __init__(
        self,
        subparsers: "_SubParsersAction",
        args: Optional["Namespace"] = None,
        command: Optional["CommandInfo"] = None,
        from_defaults_factory: bool = False,
        parser: Optional["ArgumentParser"] = None,
    ):
        super().__init__(
            subparsers, args=args, command=command, from_defaults_factory=from_defaults_factory, parser=parser
        )
        self.args_string = " ".join(sys.argv[3:])

    @staticmethod
    def parse_args(parser: "ArgumentParser"):
        return parse_args_neuronx(parser)

    def run(self):
        full_command = f"python3 -m optimum.exporters.neuron {self.args_string}"
        subprocess.run(full_command, shell=True, check=True)
