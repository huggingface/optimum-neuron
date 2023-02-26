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
import copy
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict
from unittest import TestCase

import pytest
from optimum.neuron.exporter import export, validate_model_outputs
from optimum.neuron.exporter.model_configs import BertNeuronConfig
from parameterized import parameterized
from transformers import AutoModelForSequenceClassification, AutoTokenizer, is_torch_available
from transformers.testing_utils import require_torch, slow


EXPORT_MODELS_TINY = {
    "bert": "hf-internal-testing/tiny-random-BertModel",
}


def _get_models_to_test(export_models_dict: Dict):
    models_to_test = []
    if is_torch_available():
        for model_type, model_id in export_models_dict.items():
            task = "sequence-classification"
            model_type = model_type.replace("_", "-")
            models_to_test.append((f"{model_type}_{task}", model_type, model_id, task))

    return sorted(models_to_test)


class NeuronXExportTestCase(TestCase):
    """
    Integration tests ensuring supported models are correctly exported.
    """

    def _neuronx_export(
        self,
        test_name: str,
        model_type: str,
        model_id: str,
        task: str,
    ):
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        reference_model = copy.deepcopy(model)

        neuron_config = BertNeuronConfig(config=model.config, task=task)

        atol = neuron_config.ATOL_FOR_VALIDATION
        dummy_inputs_shapes = {"batch_size": 2, "sequence_length": 18}

        with NamedTemporaryFile("w") as output:
            try:
                export(
                    model=model,
                    config=neuron_config,
                    output=Path(output.name),
                    input_shapes=dummy_inputs_shapes,
                )

                validate_model_outputs(
                    config=neuron_config,
                    reference_model=reference_model,
                    neuron_model_path=Path(output.name),
                    neuron_named_outputs=["logits"],
                    atol=atol,
                    input_shapes=dummy_inputs_shapes,
                )
            except (RuntimeError, ValueError) as e:
                self.fail(f"{model_type}, {task} -> {e}")

    @parameterized.expand(_get_models_to_test(EXPORT_MODELS_TINY))
    @require_torch
    def test_bert_export(self, test_name, name, model_name, task):
        self._neuronx_export(test_name, name, model_name, task)
