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
import random
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Dict, Optional
from unittest import TestCase

from parameterized import parameterized
from transformers import AutoConfig, set_seed
from transformers.testing_utils import require_vision

from optimum.exporters.neuron import (
    NeuronConfig,
    export,
    export_models,
    get_stable_diffusion_models_for_export,
    validate_model_outputs,
    validate_models_outputs,
)
from optimum.exporters.neuron.model_configs import *  # noqa: F403
from optimum.exporters.tasks import TasksManager
from optimum.neuron.utils import is_neuronx_available
from optimum.neuron.utils.testing_utils import is_inferentia_test
from optimum.utils import DEFAULT_DUMMY_SHAPES, is_diffusers_available, logging
from optimum.utils.testing_utils import require_diffusers

from .exporters_utils import EXPORT_MODELS_TINY, STABLE_DIFFUSION_MODELS_TINY


if is_diffusers_available():
    from diffusers import StableDiffusionPipeline

SEED = 42

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _get_models_to_test(export_models_dict: Dict, random_pick: Optional[int] = None):
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

    if random_pick is not None:
        return sorted(random.choices(models_to_test, k=random_pick))
    else:
        return sorted(models_to_test)


@is_inferentia_test
class NeuronExportTestCase(TestCase):
    """
    Integration tests ensuring supported models are correctly exported.
    """

    def _neuronx_export(
        self,
        test_name: str,
        model_type: str,
        model_name: str,
        task: str,
        neuron_config_constructor: "NeuronConfig",
        dynamic_batch_size: bool = False,
    ):
        model_class = TasksManager.get_model_class_for_task(task, framework="pt")
        config = AutoConfig.from_pretrained(model_name)
        model = model_class.from_config(config)
        reference_model = copy.deepcopy(model)

        mandatory_shapes = {
            name: DEFAULT_DUMMY_SHAPES[name]
            for name in neuron_config_constructor.func.get_mandatory_axes_for_task(task)
        }
        neuron_config = neuron_config_constructor(
            config=model.config, task=task, dynamic_batch_size=dynamic_batch_size, **mandatory_shapes
        )

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
    def test_export(self, test_name, name, model_name, task, neuron_config_constructor):
        self._neuronx_export(test_name, name, model_name, task, neuron_config_constructor)

    @parameterized.expand(_get_models_to_test(EXPORT_MODELS_TINY), skip_on_empty=True)  # , random_pick=1
    def test_export_with_dynamic_batch_size(self, test_name, name, model_name, task, neuron_config_constructor):
        if is_neuronx_available():
            self._neuronx_export(test_name, name, model_name, task, neuron_config_constructor, dynamic_batch_size=True)

    @parameterized.expand(STABLE_DIFFUSION_MODELS_TINY)
    @require_vision
    @require_diffusers
    def test_export_for_stable_diffusion_models(self, model_name):
        set_seed(SEED)

        pipeline = StableDiffusionPipeline.from_pretrained(model_name)
        output_model_names = [
            "text_encoder/model.neuron",
            "unet/model.neuron",
            "vae_decoder/model.neuron",
            "vae_conv/model.neuron",
        ]
        text_encoder_input_shapes = {"batch_size": 2, "sequence_length": 18}
        vae_decoder_input_shapes = unet_input_shapes = vae_post_quant_conv_input_shapes = {
            "batch_size": 2,
            "num_channels": 4,
            "height": 64,
            "width": 64,
        }
        models_and_neuron_configs = get_stable_diffusion_models_for_export(
            pipeline,
            text_encoder_input_shapes,
            vae_decoder_input_shapes,
            unet_input_shapes,
            vae_post_quant_conv_input_shapes,
            dynamic_batch_size=True,
        )

        with TemporaryDirectory() as tmpdirname:
            _, neuron_outputs = export_models(
                models_and_neuron_configs=models_and_neuron_configs,
                output_dir=Path(tmpdirname),
                output_file_names=output_model_names,
            )
            validate_models_outputs(
                models_and_neuron_configs=models_and_neuron_configs,
                neuron_named_outputs=neuron_outputs,
                output_dir=Path(tmpdirname),
                neuron_files_subpaths=output_model_names,
            )
