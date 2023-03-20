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

import sys
from argparse import ArgumentParser, RawTextHelpFormatter

from ...neuron.utils import is_neuron_available, is_neuronx_available
from .. import BaseOptimumCLICommand


if is_neuron_available():
    from .neuron import NeuronExportCommand, parse_args_neuron

    NEURON_COMPILER = "Neuron"

if is_neuronx_available():
    from .neuronx import (
        NeuronxExportCommand as NeuronExportCommand,
    )
    from .neuronx import (
        parse_args_neuronx as parse_args_neuron,
    )

    NEURON_COMPILER = "Neuronx"


def neuron_export_factory(_):
    return NeuronExportCommand(" ".join(sys.argv[3:]))


class ExportCommand(BaseOptimumCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        export_parser = parser.add_parser(
            "export", help=f"Export PyTorch models to {NEURON_COMPILER} compiled TorchScript models."
        )
        export_sub_parsers = export_parser.add_subparsers()

        neuron_parser = export_sub_parsers.add_parser("neuron", help=f"Export PyTorch to {NEURON_COMPILER}.")

        parse_args_neuron(neuron_parser)
        neuron_parser.set_defaults(func=neuron_export_factory)

    def run(self):
        raise NotImplementedError()
