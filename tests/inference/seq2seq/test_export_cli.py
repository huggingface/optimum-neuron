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
import subprocess
import tempfile
import unittest

import pytest
from optimum.utils import logging

from optimum.exporters.neuron.model_configs import *  # noqa: F403
from optimum.neuron.utils.testing_utils import requires_neuronx


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class TestExportCLI(unittest.TestCase):
    @requires_neuronx
    def test_encoder_decoder(self):
        model_id = "hf-internal-testing/tiny-random-t5"
        with tempfile.TemporaryDirectory() as tempdir:
            subprocess.run(
                [
                    "optimum-cli",
                    "export",
                    "neuron",
                    "--model",
                    model_id,
                    "--task",
                    "text2text-generation",
                    "--batch_size",
                    "1",
                    "--sequence_length",
                    "18",
                    "--num_beams",
                    "4",
                    "--auto_cast",
                    "matmul",
                    "--auto_cast_type",
                    "bf16",
                    tempdir,
                ],
                shell=False,
                check=True,
            )

    @pytest.mark.skip(reason="Skipping the test since `parallel_model_trace` is deprecated(to fix).")
    @requires_neuronx
    def test_encoder_decoder_optional_outputs(self):
        model_id = "hf-internal-testing/tiny-random-t5"
        with tempfile.TemporaryDirectory() as tempdir:
            subprocess.run(
                [
                    "optimum-cli",
                    "export",
                    "neuron",
                    "--model",
                    model_id,
                    "--task",
                    "text2text-generation",
                    "--batch_size",
                    "1",
                    "--sequence_length",
                    "18",
                    "--num_beams",
                    "4",
                    "--auto_cast",
                    "matmul",
                    "--auto_cast_type",
                    "bf16",
                    "--output_hidden_states",
                    "--output_attentions",
                    tempdir,
                ],
                shell=False,
                check=True,
            )

    @pytest.mark.skip(reason="Skipping the test since `parallel_model_trace` is deprecated(to fix).")
    @requires_neuronx
    def test_encoder_decoder_tp2(self):
        model_id = "michaelbenayoun/t5-tiny-random"
        with tempfile.TemporaryDirectory() as tempdir:
            subprocess.run(
                [
                    "optimum-cli",
                    "export",
                    "neuron",
                    "--model",
                    model_id,
                    "--task",
                    "text2text-generation",
                    "--tensor_parallel_size",
                    "2",
                    "--batch_size",
                    "1",
                    "--sequence_length",
                    "18",
                    "--num_beams",
                    "4",
                    "--auto_cast",
                    "matmul",
                    "--auto_cast_type",
                    "bf16",
                    tempdir,
                ],
                shell=False,
                check=True,
            )
