# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Tests validating that models can be parallelized correctly."""

import unittest
from typing import TYPE_CHECKING, List, Optional, Type, Union

from transformers.models.auto.modeling_auto import (
    MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_BACKBONE_MAPPING_NAMES,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_CTC_MAPPING_NAMES,
    MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES,
    MODEL_FOR_MASKED_LM_MAPPING_NAMES,
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES,
    MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES,
    MODEL_FOR_PRETRAINING_MAPPING_NAMES,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES,
)


if TYPE_CHECKING:
    from transformers import PretrainedConfig


def _generate_supported_model_class_names(
    model_name: Type["PretrainedConfig"],
    supported_tasks: Optional[Union[str, List[str]]] = None,
) -> List[str]:
    task_mapping = {
        # TODO: enable that when base models are supported.
        # "default": MODEL_MAPPING_NAMES,
        "pretraining": MODEL_FOR_PRETRAINING_MAPPING_NAMES,
        "next-sentence-prediction": MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES,
        "masked-lm": MODEL_FOR_MASKED_LM_MAPPING_NAMES,
        "causal-lm": MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
        "seq2seq-lm": MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
        "speech-seq2seq": MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES,
        "multiple-choice": MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES,
        "document-question-answering": MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES,
        "question-answering": MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,
        "sequence-classification": MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
        "token-classification": MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
        "masked-image-modeling": MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES,
        "image-classification": MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,
        "zero-shot-image-classification": MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES,
        "ctc": MODEL_FOR_CTC_MAPPING_NAMES,
        "audio-classification": MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,
        "semantic-segmentation": MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES,
        "backbone": MODEL_FOR_BACKBONE_MAPPING_NAMES,
    }

    if supported_tasks is None:
        supported_tasks = task_mapping.keys()
    if isinstance(supported_tasks, str):
        supported_tasks = [supported_tasks]

    model_class_names = []
    for task in supported_tasks:
        class_name = task_mapping[task].get(model_name, None)
        if class_name:
            model_class_names.append(class_name)

    return model_class_names


MODEL_TYPES_TO_TEST = [
    ("bert", "hf-internal-testing/tiny-random-bert"),
    ("roberta", "hf-internal-testing/tiny-random-roberta"),
    ("gpt_neo", "hf-internal-testing/tiny-random-GPTNeoModel"),
    ("llama", "HuggingFaceM4/tiny-random-LlamaForCausalLM"),
]

MODELS_TO_TEST = []
for model_type, model_name_or_path in MODEL_TYPES_TO_TEST:
    for model_class_name in _generate_supported_model_class_names(model_type):
        MODELS_TO_TEST.append((model_class_name, model_name_or_path))


class ModelParallelizationTestCase(unittest.TestCase):
    def get_parallel_test_python_file_content(
        self,
        model_class: str,
        model_name_or_path: str,
        from_config: bool,
        tp_size: int,
        lazy_load: bool,
    ):
        model_import = f"from transformers import AutoConfig, AutoTokenizer, {model_class}"
        other_imports = (
            "import torch\n"
            "from optimum.neuron.distributed import ParallelizersManager, lazy_load_for_parallelism\n"
            "import neuronx_distributed\n"
            "import os\n"
        )

        initialize_torch_distributed = (
            "if os.environ.get('TORCHELASTIC_RUN_ID'):\n"
            "\timport torch_xla.distributed.xla_backend as xbn\n"
            "if not isinstance(torch.distributed.group.WORLD, xbn.ProcessGroupXla):\n"
            "\ttorch.distributed.init_process_group(backend='xla')\n"
        )

        initialize_tp = f"neuronx_distributed.parallel_layers.parallel_state.initialize_model_parallel(tensor_model_parallel_size={tp_size})"

        config_loading = f"config = AutoConfig.from_pretrained('{model_name_or_path}')"
        preprocessor_loading = f"preprocessor = AutoTokenizer.from_pretrained('{model_name_or_path}')"
        inputs = "inputs = preprocessor('This is a test to check that TP is working.', return_tensors='pt')"

        if from_config:
            model_loading_line = f"model = {model_class}(config)"
            full_model_loading_line = f"full_model = {model_class}(config)"
        else:
            model_loading_line = f"model = {model_class}.from_pretrained('{model_name_or_path}')"
            full_model_loading_line = f"full_model = {model_class}.from_pretrained('{model_name_or_path}')"

        if lazy_load:
            model_loading_block = (
                f"with lazy_load_for_parallelism(tensor_parallel_size={tp_size}):\n" f"    {model_loading_line}"
            )
        else:
            model_loading_block = model_loading_line

        parallel_model_loading = (
            "parallel_model = ParallelizersManager.parallelizer_for_model(model).parallelize(model)"
        )

        inference = (
            "full_model_outputs = full_model(**inputs, return_dict=True)\n"
            "parallel_model_outputs = parallel_model(**inputs, return_dict=True)\n"
            "for name, t in full_model_outputs.items():\n"
            "   torch.testing.assert_close(t, parallel_model_outputs[name])"
        )

        return "\n".join(
            [
                model_import,
                other_imports,
                initialize_torch_distributed,
                initialize_tp,
                config_loading,
                preprocessor_loading,
                inputs,
                full_model_loading_line,
                model_loading_block,
                parallel_model_loading,
                inference,
            ]
        )

    # TODO: enable that when continuing to write tests.
    # @parameterized.expand(MODELS_TO_TEST)
    # def test_model_parallel(self, model_class_name: str, model_name_or_path: str):
    #     python_code = self.get_parallel_test_python_file_content(model_class_name, model_name_or_path, False, 2, False)

    #     with TemporaryDirectory() as tmpdirname:
    #         with open(f"{tmpdirname}/code.py", "w") as fp:
    #             fp.write(python_code)

    #         cmd = ["torchrun", "--nproc_per_node=2", f"{tmpdirname}/code.py"]

    #         p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #         stdout, stderr = p.communicate()
    #         print(stdout.decode("utf-8"))
    #         print("\n" * 10)
    #         print(stderr.decode("utf-8"))
