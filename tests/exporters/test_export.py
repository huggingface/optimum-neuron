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

from parameterized import parameterized
from transformers import AutoConfig
from transformers.testing_utils import require_torch

from optimum.exporters.neuron import NeuronConfig, export, validate_model_outputs
from optimum.exporters.neuron.model_configs import *  # noqa: F403
from optimum.exporters.tasks import TasksManager
from optimum.neuron.utils.testing_utils import is_inferentia_test
from optimum.utils import DEFAULT_DUMMY_SHAPES, logging

from .exporters_utils import EXPORT_MODELS_TINY


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _get_models_to_test(export_models_dict: Dict):
    models_to_test = []
    for model_type, model_names_tasks in export_models_dict.items():
        model_type = model_type.replace("_", "-")
        task_config_mapping = TasksManager.get_supported_tasks_for_model_type(model_type, "neuron")

        if isinstance(model_names_tasks, str):  # test export of all tasks on the same model
            tasks = list(task_config_mapping.keys())
            model_tasks = {model_names_tasks: tasks}
        else:
            n_tested_tasks = sum(len(tasks) for tasks in model_names_tasks.values())
            if n_tested_tasks != len(task_config_mapping):
                logger.warning(f"Not all tasks are tested for {model_type}.")
            model_tasks = model_names_tasks  # possibly, test different tasks on different models

        for model_name, tasks in model_tasks.items():
            for task in tasks:
                default_shapes = dict(DEFAULT_DUMMY_SHAPES)
                neuron_config_constructor = TasksManager.get_exporter_config_constructor(
                    model_type=model_type,
                    exporter="neuron",
                    task=task,
                    model_name=model_name,
                    exporter_config_kwargs={**default_shapes},
                )

                models_to_test.append(
                    (f"{model_type}_{task}", model_type, model_name, task, neuron_config_constructor)
                )

    return sorted(models_to_test)


@is_inferentia_test
class NeuronExportTestCase(TestCase):
    """
    Integration tests ensuring supported models are correctly exported.
    """

    def _neuronx_export(
        self, test_name: str, model_type: str, model_name: str, task: str, neuron_config_constructor: "NeuronConfig"
    ):
        model_class = TasksManager.get_model_class_for_task(task, framework="pt")
        config = AutoConfig.from_pretrained(model_name)
        model = model_class.from_config(config)
        reference_model = copy.deepcopy(model)

        neuron_config = neuron_config_constructor(config=model.config, task=task, batch_size=2, sequence_length=18)

        atol = neuron_config.ATOL_FOR_VALIDATION

        with NamedTemporaryFile("w") as output:
            try:
                _, neuron_outputs = export(
                    model=model,
                    config=neuron_config,
                    output=Path(output.name),
                )

                validate_model_outputs(
                    config=neuron_config,
                    reference_model=reference_model,
                    neuron_model_path=Path(output.name),
                    neuron_named_outputs=neuron_outputs,
                    atol=atol,
                )
            except (RuntimeError, ValueError) as e:
                self.fail(f"{model_type}, {task} -> {e}")

    @parameterized.expand(_get_models_to_test(EXPORT_MODELS_TINY))
    @require_torch
    def test_export(self, test_name, name, model_name, task, neuron_config_constructor):
        self._neuronx_export(test_name, name, model_name, task, neuron_config_constructor)
