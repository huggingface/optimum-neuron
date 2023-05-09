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

import random
import subprocess
import tempfile
import unittest
from itertools import product
from typing import Dict, Optional

from parameterized import parameterized

from optimum.exporters.neuron.model_configs import *  # noqa: F403
from optimum.exporters.tasks import TasksManager
from optimum.neuron.utils import is_neuron_available, is_neuronx_available
from optimum.neuron.utils.testing_utils import is_inferentia_test
from optimum.utils import DEFAULT_DUMMY_SHAPES, logging

from ..exporters.exporters_utils import EXPORT_MODELS_TINY


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


_COMMOM_COMMANDS = {
    "--auto_cast": ["none", "matmul", "all"],
    "--auto_cast_type": ["bf16", "fp16"],  # "tf32", "mixed"
}
_NEURON_COMMANDS = {}
_NEURONX_COMMANDS = {}
_DYNAMIC_COMMANDS = {"neuron": ["--disable-fast-relayout"], "neuronx": []}


def _get_models_to_test(export_models_dict: Dict, random_pick: Optional[int] = 1):
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
                TasksManager.get_exporter_config_constructor(
                    model_type=model_type,
                    exporter="neuron",
                    task=task,
                    model_name=model_name,
                    exporter_config_kwargs={**default_shapes},
                )

                models_to_test.append((f"{model_type}_{task}", model_name, task))

    if random_pick is not None:
        return sorted(random.choices(models_to_test, k=random_pick))
    else:
        return sorted(models_to_test)


def _get_commands_to_test(models_to_test):
    commands_to_test = []
    for test_name, model_name, task in models_to_test:
        if is_neuron_available():
            command_items = dict(_COMMOM_COMMANDS, **_NEURON_COMMANDS)
            dynamic_args = _DYNAMIC_COMMANDS["neuron"]
        elif is_neuronx_available():
            command_items = dict(_COMMOM_COMMANDS, **_NEURONX_COMMANDS)
            dynamic_args = _DYNAMIC_COMMANDS["neuronx"]
        else:
            continue

        base_command = f"optimum-cli export neuron --model {model_name} --task {task}"

        for extra_arg_options in product(*command_items.values()):
            extra_command = " ".join(
                [" ".join([arg, option]) for arg, option in zip(command_items, extra_arg_options)]
            )
            extra_command += " " + " ".join(random.choices(dynamic_args, k=random.randint(0, len(dynamic_args))))
            command = base_command + " " + extra_command

            commands_to_test.append((test_name + extra_command.strip(), command))

    return sorted(commands_to_test)


@is_inferentia_test
class TestExportCLI(unittest.TestCase):
    def test_helps_no_raise(self):
        commands = [
            "optimum-cli --help",
            "optimum-cli export --help",
            "optimum-cli export neuron --help",
        ]

        for command in commands:
            subprocess.run(command, shell=True, check=True)

    @parameterized.expand(_get_commands_to_test(_get_models_to_test(EXPORT_MODELS_TINY)), skip_on_empty=True)
    def test_export_commands(self, test_name, command_content):
        with tempfile.TemporaryDirectory() as tempdir:
            command = command_content + f" {tempdir}"

            subprocess.run(command, shell=True, check=True)
