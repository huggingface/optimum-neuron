# coding=utf-8
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
import os
import subprocess
import tempfile
import unittest

from optimum.utils import logging

from optimum.exporters.neuron.model_configs import *  # noqa: F403
from optimum.neuron.utils import is_neuronx_available
from optimum.neuron.utils.testing_utils import requires_neuronx


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class TestExportCLI(unittest.TestCase):
    def test_helps_no_raise(self):
        commands = [
            "optimum-cli --help",
            "optimum-cli export --help",
            "optimum-cli export neuron --help",
        ]

        for command in commands:
            subprocess.run(command, shell=True, check=True)

    def test_export_commands(self):
        model_id = "hf-internal-testing/tiny-random-BertModel"
        with tempfile.TemporaryDirectory() as tempdir:
            subprocess.run(
                [
                    "optimum-cli",
                    "export",
                    "neuron",
                    "--model",
                    model_id,
                    "--sequence_length",
                    "16",
                    "--batch_size",
                    "1",
                    "--task",
                    "text-classification",
                    tempdir,
                ],
                shell=False,
                check=True,
            )

    @requires_neuronx
    def test_dynamic_batching(self):
        model_id = "hf-internal-testing/tiny-random-BertModel"
        with tempfile.TemporaryDirectory() as tempdir:
            subprocess.run(
                [
                    "optimum-cli",
                    "export",
                    "neuron",
                    "--dynamic-batch-size",
                    "--model",
                    model_id,
                    "--sequence_length",
                    "16",
                    "--batch_size",
                    "1",
                    "--task",
                    "text-classification",
                    tempdir,
                ],
                shell=False,
                check=True,
            )

    @requires_neuronx
    def test_opt_level(self):
        model_id = "hf-internal-testing/tiny-random-BertModel"
        optlevels = ["-O1", "-O2", "-O3"]
        for optlevel in optlevels:
            with tempfile.TemporaryDirectory() as tempdir:
                subprocess.run(
                    [
                        "optimum-cli",
                        "export",
                        "neuron",
                        "--model",
                        model_id,
                        "--sequence_length",
                        "16",
                        "--batch_size",
                        "1",
                        "--task",
                        "text-classification",
                        optlevel,
                        tempdir,
                    ],
                    shell=False,
                    check=True,
                )

    def test_store_intemediary(self):
        model_id = "hf-internal-testing/tiny-random-BertModel"
        with tempfile.TemporaryDirectory() as tempdir:
            save_path = f"{tempdir}/neff"
            subprocess.run(
                [
                    "optimum-cli",
                    "export",
                    "neuron",
                    "--model",
                    model_id,
                    "--sequence_length",
                    "16",
                    "--batch_size",
                    "1",
                    "--task",
                    "text-classification",
                    "--compiler_workdir",
                    save_path,
                    tempdir,
                ],
                shell=False,
                check=True,
            )
            if is_neuronx_available():
                neff_path = os.path.join(save_path, "graph.neff")
                self.assertTrue(os.path.exists(neff_path))
            else:
                neff_path = os.path.join(save_path, "32", "neff.json")

    @requires_neuronx
    def test_whisper(self):
        model_id = "openai/whisper-tiny"
        with tempfile.TemporaryDirectory() as tempdir:
            subprocess.run(
                [
                    "optimum-cli",
                    "export",
                    "neuron",
                    "--model",
                    model_id,
                    "--task",
                    "automatic-speech-recognition",
                    "--batch_size",
                    "1",
                    "--sequence_length",
                    "32",
                    "--auto_cast",
                    "matmul",
                    "--auto_cast_type",
                    "bf16",
                    tempdir,
                ],
                shell=False,
                check=True,
            )
