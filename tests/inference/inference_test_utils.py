# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import shutil
import tempfile
import unittest
from typing import Dict

from transformers import set_seed

from ..exporters.exporters_utils import EXPORT_MODELS_TINY as MODEL_NAMES
from ..exporters.exporters_utils import SEED


class NeuronModelTestMixin(unittest.TestCase):
    ARCH_MODEL_MAP = {}
    STATIC_INPUTS_SHAPES = {"batch_size": 1, "sequence_length": 32}

    @classmethod
    def setUpClass(cls):
        cls.neuron_model_dirs = {}

    def _setup(self, model_args: Dict):
        """
        Exports the PyTorch models to Neuron models ahead of time to avoid multiple exports during the tests.
        We don't use unittest setUpClass, in order to still be able to run individual tests.
        """
        model_arch = model_args["model_arch"]
        model_arch_and_params = model_args["test_name"]
        dynamic_batch_size = getattr(model_args, "dynamic_batch_size", False)

        if model_arch_and_params not in self.neuron_model_dirs:
            # model_args will contain kwargs to pass to NeuronBaseModel.from_pretrained()
            model_args.pop("test_name")
            model_args.pop("model_arch")
            model_args.pop("dynamic_batch_size", None)

            model_id = (
                self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
            )
            set_seed(SEED)
            neuron_model = self.NEURON_MODEL_CLASS.from_pretrained(
                model_id, **model_args, export=True, dynamic_batch_size=dynamic_batch_size, **self.STATIC_INPUTS_SHAPES
            )

            model_dir = tempfile.mkdtemp(prefix=f"{model_arch_and_params}_{self.TASK}_")
            neuron_model.save_pretrained(model_dir)
            self.neuron_model_dirs[model_arch_and_params] = model_dir

    @classmethod
    def tearDownClass(cls):
        for _, dir_path in cls.neuron_model_dirs.items():
            shutil.rmtree(dir_path)
