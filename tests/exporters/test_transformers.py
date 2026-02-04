# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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

# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from optimum.exporters.tasks import TasksManager
from optimum.utils import DEFAULT_DUMMY_SHAPES
from optimum.utils.testing_utils import require_sentence_transformers
from parameterized import parameterized
from transformers import AutoConfig, AutoModelForSeq2SeqLM, set_seed
from transformers.testing_utils import slow

from optimum.exporters.neuron import (
    NeuronDefaultConfig,
    export,
    export_models,
    validate_model_outputs,
    validate_models_outputs,
)
from optimum.exporters.neuron.__main__ import get_submodels_and_neuron_configs
from optimum.exporters.neuron.model_configs import *  # noqa: F403
from optimum.neuron.utils import InputShapesArguments
from optimum.neuron.utils.testing_utils import requires_neuronx

from .exporters_utils import (
    ENCODER_DECODER_MODELS_TINY,
    EXPORT_MODELS_TINY,
    EXTRA_DEFAULT_DUMMY_SHAPES,
    SENTENCE_TRANSFORMERS_MODELS,
    WEIGHTS_NEFF_SEPARATION_UNSUPPORTED_ARCH,
    get_models_to_test,
)


SEED = 42


class NeuronExportTestCase(unittest.TestCase):
    """
    Integration tests ensuring supported models are correctly exported.
    """

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
            reference_model = model_class(model_name)
            if "clip" in model[0].__class__.__name__.lower():
                config = model[0].model.config
            else:
                config = model[0].auto_model.config
        else:
            model_class = TasksManager.get_model_class_for_task(task, model_type=model_type, framework="pt")
            config = AutoConfig.from_pretrained(model_name)
            model = model_class.from_config(config)
            reference_model = model_class.from_config(config)

        mandatory_shapes = {
            name: DEFAULT_DUMMY_SHAPES.get(name) or EXTRA_DEFAULT_DUMMY_SHAPES.get(name)
            for name in neuron_config_constructor.func.get_mandatory_axes_for_task(task)
        }
        mandatory_shapes = InputShapesArguments(**mandatory_shapes)
        neuron_config = neuron_config_constructor(
            config=config,
            task=task,
            dynamic_batch_size=dynamic_batch_size,
            input_shapes=mandatory_shapes,
        )

        atol = neuron_config.ATOL_FOR_VALIDATION

        with NamedTemporaryFile("w") as output:
            try:
                _, neuron_outputs = export(
                    model_or_path=model,
                    config=neuron_config,
                    output=Path(output.name),
                    inline_weights_to_neff=inline_weights_to_neff,
                    instance_type="inf2",
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

    @parameterized.expand(get_models_to_test(EXPORT_MODELS_TINY, library_name="transformers"))
    def test_export(self, test_name, name, model_name, task, neuron_config_constructor):
        self._neuronx_export(test_name, name, model_name, task, neuron_config_constructor)

    @parameterized.expand(
        get_models_to_test(
            EXPORT_MODELS_TINY,
            exclude_model_types=WEIGHTS_NEFF_SEPARATION_UNSUPPORTED_ARCH,
            library_name="transformers",
        )
    )
    @slow
    def test_export_separated_weights(self, test_name, name, model_name, task, neuron_config_constructor):
        self._neuronx_export(
            test_name, name, model_name, task, neuron_config_constructor, inline_weights_to_neff=False
        )

    @parameterized.expand(get_models_to_test(SENTENCE_TRANSFORMERS_MODELS, library_name="sentence_transformers"))
    @require_sentence_transformers
    @requires_neuronx
    def test_export_sentence_transformers(self, test_name, name, model_name, task, neuron_config_constructor):
        self._neuronx_export(test_name, name, model_name, task, neuron_config_constructor)

    @parameterized.expand(get_models_to_test(EXPORT_MODELS_TINY, library_name="transformers"), skip_on_empty=True)
    @slow
    @requires_neuronx
    def test_export_with_dynamic_batch_size(self, test_name, name, model_name, task, neuron_config_constructor):
        self._neuronx_export(test_name, name, model_name, task, neuron_config_constructor, dynamic_batch_size=True)


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
            models_and_neuron_configs, output_model_names = get_submodels_and_neuron_configs(
                model=model,
                input_shapes=input_shapes,
                task="text2text-generation",
                output=Path(tmpdirname),
                library_name="transformers",
                model_name_or_path=model_id,
                output_attentions=True,
                output_hidden_states=True,
            )
            _, neuron_outputs = export_models(
                models_and_neuron_configs=models_and_neuron_configs,
                task="text2text-generation",
                output_dir=Path(tmpdirname),
                output_file_names=output_model_names,
            )
            validate_models_outputs(
                models_and_neuron_configs=models_and_neuron_configs,
                neuron_named_outputs=neuron_outputs,
                output_dir=Path(tmpdirname),
                neuron_files_subpaths=output_model_names,
            )
