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
import random
from typing import Dict

from optimum.exporters.tasks import TasksManager
from optimum.utils import DEFAULT_DUMMY_SHAPES, logging

from optimum.neuron.utils import InputShapesArguments


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Transformers

EXPORT_MODELS_TINY = {
    "albert": "hf-internal-testing/tiny-random-AlbertModel",
    "audio-spectrogram-transformer": "Ericwang/tiny-random-ast",
    "beit": "hf-internal-testing/tiny-random-BeitForImageClassification",
    "bert": "hf-internal-testing/tiny-random-BertModel",
    "camembert": "hf-internal-testing/tiny-random-camembert",
    "convbert": "hf-internal-testing/tiny-random-ConvBertModel",
    "convnext": "hf-internal-testing/tiny-random-convnext",
    "convnextv2": "hf-internal-testing/tiny-random-ConvNextV2Model",
    "cvt": "hf-internal-testing/tiny-random-CvTModel",
    "deberta": "hf-internal-testing/tiny-random-DebertaModel",
    "deberta-v2": "hf-internal-testing/tiny-random-DebertaV2Model",
    "deit": "hf-internal-testing/tiny-random-DeiTModel",
    "distilbert": "hf-internal-testing/tiny-random-DistilBertModel",
    "donut-swin": "hf-internal-testing/tiny-random-DonutSwinModel",
    "dpt": "hf-internal-testing/tiny-random-DPTModel",
    "electra": "hf-internal-testing/tiny-random-ElectraModel",
    # "esm": "hf-internal-testing/tiny-random-EsmModel",  # TODO: put the test back, when https://github.com/aws-neuron/aws-neuron-sdk/issues/1081 is solved.
    "flaubert": "flaubert/flaubert_small_cased",
    "hubert": "hf-internal-testing/tiny-random-HubertModel",
    "levit": "hf-internal-testing/tiny-random-LevitModel",
    "mobilebert": "hf-internal-testing/tiny-random-MobileBertModel",
    "mobilenet_v2": "hf-internal-testing/tiny-random-MobileNetV2Model",
    # "mobilevit": "hf-internal-testing/tiny-random-mobilevit",  # blocked since neuron sdk 2.23: timeout
    "modernbert": "hf-internal-testing/tiny-random-ModernBertModel",
    "mpnet": "hf-internal-testing/tiny-random-MPNetModel",
    "phi": "bumblebee-testing/tiny-random-PhiModel",
    "roberta": "hf-internal-testing/tiny-random-RobertaModel",
    "roformer": "hf-internal-testing/tiny-random-RoFormerModel",
    # "sew": "hf-internal-testing/tiny-random-SEWModel",  # blocked
    # "sew-d": "hf-internal-testing/tiny-random-SEWDModel",  # blocked
    "swin": "hf-internal-testing/tiny-random-SwinModel",
    # "unispeech": "hf-internal-testing/tiny-random-unispeech",  # blocked since neuron sdk 2.23: neuronx-cc failed with 70
    # "unispeech-sat": "hf-internal-testing/tiny-random-unispeech-sat",  # blocked since neuron sdk 2.23: neuronx-cc failed with 70
    "vit": "hf-internal-testing/tiny-random-vit",
    "wav2vec2": "hf-internal-testing/tiny-random-Wav2Vec2Model",
    # "wav2vec2-conformer": "hf-internal-testing/tiny-random-wav2vec2-conformer",  # blocked
    # "wavlm": "hf-internal-testing/tiny-random-wavlm",  # blocked since neuron sdk 2.23: neuronx-cc failed with 70
    "xlm": "hf-internal-testing/tiny-random-XLMModel",
    "xlm-roberta": "hf-internal-testing/tiny-xlm-roberta",
    "yolos": "hf-internal-testing/tiny-random-YolosModel",
}

ENCODER_DECODER_MODELS_TINY = {
    "t5": "hf-internal-testing/tiny-random-t5",
}

SENTENCE_TRANSFORMERS_MODELS = {
    "transformer": "sentence-transformers/all-MiniLM-L6-v2",
    "clip": "sentence-transformers/clip-ViT-B-32",
}

WEIGHTS_NEFF_SEPARATION_UNSUPPORTED_ARCH = ["camembert", "roberta", "mobilenet_v2"]

# Diffusers

STABLE_DIFFUSION_MODELS_TINY = {
    "stable-diffusion": "hf-internal-testing/tiny-stable-diffusion-torch",
    "stable-diffusion-xl": "echarlaix/tiny-random-stable-diffusion-xl",
    "latent-consistency": "echarlaix/tiny-random-latent-consistency",
}

LORA_WEIGHTS_TINY = {
    "stable-diffusion": ("Jingya/tiny-stable-diffusion-lora-64", "pytorch_lora_weights.safetensors", "pokemon"),
}

EXTRA_DEFAULT_DUMMY_SHAPES = {
    "text_batch_size": 1,
    "image_batch_size": 1,
}

SEED = 42


def get_models_to_test(
    export_models_dict: Dict,
    exclude_model_types: list[str] | None = None,
    library_name: str = "transformers",
):
    models_to_test = []
    for model_type, model_names_tasks in export_models_dict.items():
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
                random_task = os.environ.get("RANDOM_TASK_PER_MODEL", False)
                if random_task:
                    tasks = random.choices(tasks, k=1)
                for task in tasks:
                    default_shapes = dict(DEFAULT_DUMMY_SHAPES)
                    default_shapes = InputShapesArguments(**default_shapes)
                    neuron_config_constructor = TasksManager.get_exporter_config_constructor(
                        model_type=model_type,
                        exporter="neuron",
                        library_name=library_name,
                        task=task,
                        model_name=model_name,
                        exporter_config_kwargs={"input_shapes": default_shapes},
                    )

                    models_to_test.append(
                        (f"{model_type}_{task}", model_type, model_name, task, neuron_config_constructor)
                    )

    return sorted(models_to_test)
