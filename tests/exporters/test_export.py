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
import os
import random
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Dict, List, Optional

from parameterized import parameterized
from transformers import AutoConfig, AutoModelForSeq2SeqLM, set_seed
from transformers.testing_utils import require_vision

from optimum.exporters.neuron import (
    NeuronDefaultConfig,
    build_stable_diffusion_components_mandatory_shapes,
    export,
    export_models,
    validate_model_outputs,
    validate_models_outputs,
)
from optimum.exporters.neuron.__main__ import _get_submodels_and_neuron_configs
from optimum.exporters.neuron.model_configs import *  # noqa: F403
from optimum.exporters.tasks import TasksManager
from optimum.neuron.utils import is_neuron_available
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx
from optimum.utils import DEFAULT_DUMMY_SHAPES, is_diffusers_available, logging
from optimum.utils.testing_utils import require_diffusers, require_sentence_transformers

from .exporters_utils import (
    ENCODER_DECODER_MODELS_TINY,
    EXPORT_MODELS_TINY,
    SENTENCE_TRANSFORMERS_MODELS,
    STABLE_DIFFUSION_MODELS_TINY,
    WEIGHTS_NEFF_SEPARATION_UNSUPPORTED_ARCH,
)


if is_diffusers_available():
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

SEED = 42

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _get_models_to_test(
    export_models_dict: Dict,
    exclude_model_types: Optional[List[str]] = None,
    library_name: str = "transformers",
):
    models_to_test = []
    for model_type, model_names_tasks in export_models_dict.items():
        model_type = model_type.replace("_", "-")
        if exclude_model_types is None or (model_type not in exclude_model_types):
            task_config_mapping = TasksManager.get_supported_tasks_for_model_type(
                model_type, "neuron", library_name=library_name
            )

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
                        library_name=library_name,
                        task=task,
                        model_name=model_name,
                        exporter_config_kwargs={**default_shapes},
                    )

                    models_to_test.append(
                        (f"{model_type}_{task}", model_type, model_name, task, neuron_config_constructor)
                    )

    random_pick = os.environ.get("MAX_EXPORT_TEST_COMBINATIONS", None)
    if random_pick is not None:
        return sorted(random.choices(models_to_test, k=int(random_pick)))
    else:
        return sorted(models_to_test)


class NeuronExportTestCase(unittest.TestCase):
    """
    Integration tests ensuring supported models are correctly exported.
    """

    if is_neuron_available():
        # Deberta has 'XSoftmax' unsupported on INF1
        for model in ["deberta", "deberta-v2"]:
            EXPORT_MODELS_TINY.pop(model)

    def _neuronx_export(
        self,
        test_name: str,
        model_type: str,
        model_name: str,
        task: str,
        neuron_config_constructor: "NeuronDefaultConfig",
        dynamic_batch_size: bool = False,
        inline_weights_to_neff: bool = True,
    ):
        library_name = TasksManager.infer_library_from_model(model_name)
        if library_name == "sentence_transformers":
            model_class = TasksManager.get_model_class_for_task(task, framework="pt", library=library_name)
            model = model_class(model_name)
            if "clip" in model[0].__class__.__name__.lower():
                config = model[0].model.config
            else:
                config = model[0].auto_model.config
        else:
            model_class = TasksManager.get_model_class_for_task(task, framework="pt")
            config = AutoConfig.from_pretrained(model_name)
            model = model_class.from_config(config)
        reference_model = copy.deepcopy(model)

        mandatory_shapes = {
            name: DEFAULT_DUMMY_SHAPES[name]
            for name in neuron_config_constructor.func.get_mandatory_axes_for_task(task)
        }
        neuron_config = neuron_config_constructor(
            config=config, task=task, dynamic_batch_size=dynamic_batch_size, **mandatory_shapes
        )

        atol = neuron_config.ATOL_FOR_VALIDATION

        with NamedTemporaryFile("w") as output:
            try:
                _, neuron_outputs = export(
                    model=model,
                    config=neuron_config,
                    output=Path(output.name),
                    inline_weights_to_neff=inline_weights_to_neff,
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

    @parameterized.expand(_get_models_to_test(EXPORT_MODELS_TINY, library_name="transformers"))
    @is_inferentia_test
    def test_export(self, test_name, name, model_name, task, neuron_config_constructor):
        self._neuronx_export(test_name, name, model_name, task, neuron_config_constructor)

    @parameterized.expand(
        _get_models_to_test(
            EXPORT_MODELS_TINY,
            exclude_model_types=WEIGHTS_NEFF_SEPARATION_UNSUPPORTED_ARCH,
            library_name="transformers",
        )
    )
    @is_inferentia_test
    @requires_neuronx
    def test_export_separated_weights(self, test_name, name, model_name, task, neuron_config_constructor):
        self._neuronx_export(
            test_name, name, model_name, task, neuron_config_constructor, inline_weights_to_neff=False
        )

    @parameterized.expand(_get_models_to_test(SENTENCE_TRANSFORMERS_MODELS, library_name="sentence_transformers"))
    @is_inferentia_test
    @require_vision
    @require_sentence_transformers
    @requires_neuronx
    def test_export_sentence_transformers(self, test_name, name, model_name, task, neuron_config_constructor):
        self._neuronx_export(test_name, name, model_name, task, neuron_config_constructor)

    @parameterized.expand(_get_models_to_test(EXPORT_MODELS_TINY, library_name="transformers"), skip_on_empty=True)
    @is_inferentia_test
    @requires_neuronx
    def test_export_with_dynamic_batch_size(self, test_name, name, model_name, task, neuron_config_constructor):
        self._neuronx_export(test_name, name, model_name, task, neuron_config_constructor, dynamic_batch_size=True)


@is_inferentia_test
@requires_neuronx
@require_vision
@require_diffusers
class NeuronStableDiffusionExportTestCase(unittest.TestCase):
    """
    Integration tests ensuring stable diffusion models are correctly exported.
    """

    @parameterized.expand(
        [STABLE_DIFFUSION_MODELS_TINY["stable-diffusion"], STABLE_DIFFUSION_MODELS_TINY["latent-consistency"]]
    )
    def test_export_for_stable_diffusion_models(self, model_id):
        set_seed(SEED)

        # prepare neuron config / models
        model = StableDiffusionPipeline.from_pretrained(model_id)
        input_shapes = build_stable_diffusion_components_mandatory_shapes(
            **{"batch_size": 1, "height": 64, "width": 64, "num_images_per_prompt": 4}
        )

        with TemporaryDirectory() as tmpdirname:
            models_and_neuron_configs, output_model_names = _get_submodels_and_neuron_configs(
                model=model,
                input_shapes=input_shapes,
                task="stable-diffusion",
                output=Path(tmpdirname),
                model_name_or_path=model_id,
            )
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

    @parameterized.expand([STABLE_DIFFUSION_MODELS_TINY["stable-diffusion-xl"]])
    def test_export_for_stable_diffusion_xl_models(self, model_id):
        set_seed(SEED)

        # prepare neuron config / models
        model = StableDiffusionXLPipeline.from_pretrained(model_id)
        input_shapes = build_stable_diffusion_components_mandatory_shapes(
            **{"batch_size": 1, "height": 64, "width": 64, "num_images_per_prompt": 4}
        )

        with TemporaryDirectory() as tmpdirname:
            models_and_neuron_configs, output_model_names = _get_submodels_and_neuron_configs(
                model=model,
                input_shapes=input_shapes,
                task="stable-diffusion-xl",
                output=Path(tmpdirname),
                model_name_or_path=model_id,
            )
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


@is_inferentia_test
@requires_neuronx
class NeuronEncoderDecoderExportTestCase(unittest.TestCase):
    """
    Integration tests ensuring encoder-decoder models are correctly exported.
    """

    @parameterized.expand(ENCODER_DECODER_MODELS_TINY.items())
    def test_export_encoder_decoder_models(self, model_name, model_id):
        set_seed(SEED)

        # prepare neuron config / models
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        input_shapes = {"batch_size": 1, "sequence_length": 18, "num_beams": 4}

        with TemporaryDirectory() as tmpdirname:
            models_and_neuron_configs, output_model_names = _get_submodels_and_neuron_configs(
                model=model,
                input_shapes=input_shapes,
                task="text2text-generation",
                output=Path(tmpdirname),
                model_name_or_path=model_id,
                output_attentions=True,
                output_hidden_states=True,
            )
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
