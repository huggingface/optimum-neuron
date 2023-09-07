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

import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Dict, List, Optional, Type, Union

import torch
from parameterized import parameterized
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_BACKBONE_MAPPING_NAMES,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_CTC_MAPPING_NAMES,
    MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES,
    MODEL_FOR_MASKED_LM_MAPPING_NAMES,
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

from optimum.neuron.utils.cache_utils import get_num_neuron_cores

from ..test_utils import is_trainium_test


if TYPE_CHECKING:
    from transformers import PretrainedConfig


TEMPLATE_FILE_NAME = "model_parallel_test_template.txt"
NUM_NEURON_CORES_AVAILABLE = get_num_neuron_cores()


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
        # Those architectures are more painful to deal with because the input is different.
        # "multiple-choice": MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES,
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
    ("llama", "fxmarty/tiny-llama-fast-tokenizer"),
    ("t5", "patrickvonplaten/t5-tiny-random"),
]

MODELS_TO_TEST = []
for model_type, model_name_or_path in MODEL_TYPES_TO_TEST:
    for model_class_name in _generate_supported_model_class_names(model_type):
        MODELS_TO_TEST.append((model_class_name, model_name_or_path))


@is_trainium_test
class ModelParallelizationTestCase(unittest.TestCase):
    def _test_model_parallel(
        self,
        tp_size: int,
        model_class_name: str,
        model_name_or_path: str,
        from_config: bool,
        with_lazy_load: bool,
        parallelize_embeddings: bool,
        overwrite_model_config: Optional[Dict[str, str]] = None,
        num_neuron_cores: int = NUM_NEURON_CORES_AVAILABLE,
    ):
        if num_neuron_cores < tp_size:
            raise ValueError(
                "The number of Neuron cores available is lower than the TP size, failing since the test might not be "
                "testing what is expected."
            )

        template_content = None
        template_file_path = Path(__file__).parent.resolve() / TEMPLATE_FILE_NAME
        with open(template_file_path, "r") as fp:
            template_content = fp.read()

        specialization_env = {
            "from_config": "true" if from_config else "false",
            "lazy_load": "true" if with_lazy_load else "false",
            **os.environ,
        }
        if overwrite_model_config is not None:
            specialization_env["config_overwrite"] = ",".join(
                f"{key}={value}" for key, value in overwrite_model_config.items()
            )

        with TemporaryDirectory() as tmpdirname:
            specialization_data = {
                "model_class": model_class_name,
                "model_name_or_path": model_name_or_path,
                "parallelize_embeddings": "True" if parallelize_embeddings else "False",
                "tp_size": tp_size,
                "output_path": tmpdirname,
            }
            specialized_content = template_content.format(**specialization_data)
            with open(f"{tmpdirname}/code.py", "w") as fp:
                fp.write(specialized_content)

            cmd = ["torchrun", f"--nproc_per_node={num_neuron_cores}", f"{tmpdirname}/code.py"]

            # Original model.
            p = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={"is_parallel": "false", **specialization_env}
            )
            stdout, stderr = p.communicate()

            stdout = stdout.decode("utf-8")
            stderr = stderr.decode("utf-8")

            if stdout == "":
                stdout = "N/A"
            if stderr == "":
                stderr = "N/A"

            full_output = f"Original model standard output:\n{stdout}\nOriginal model standard error:\n{stderr}"
            print(full_output)

            # Parallel model.
            p = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={"is_parallel": "true", **specialization_env}
            )
            stdout, stderr = p.communicate()

            stdout = stdout.decode("utf-8")
            stderr = stderr.decode("utf-8")

            if stdout == "":
                stdout = "N/A"
            if stderr == "":
                stderr = "N/A"

            full_output = f"Parallel model standard output:\n{stdout}\nParallel model standard error:\n{stderr}"
            print(full_output)

            temporary_dir = Path(tmpdirname)
            original_model_outputs = torch.load(temporary_dir / "original.bin")
            parallel_model_outputs = torch.load(temporary_dir / "parallel.bin")
            for name, t in parallel_model_outputs.items():
                if not isinstance(t, torch.Tensor):
                    continue
                print(t, original_model_outputs[name])
                torch.testing.assert_close(t, original_model_outputs[name], msg=f"Input called {name} do not match.")

    @parameterized.expand(MODELS_TO_TEST)
    def test_model_parallel_from_config_without_lazy_load(self, model_class_name: str, model_name_or_path: str):
        self._test_model_parallel(
            num_neuron_cores=2,
            tp_size=2,
            model_class_name=model_class_name,
            model_name_or_path=model_name_or_path,
            from_config=True,
            with_lazy_load=False,
            parallelize_embeddings=False,  # Should be True once it's working.
        )

    @parameterized.expand(MODELS_TO_TEST)
    def test_model_parallel_from_pretrained_without_lazy_load(self, model_class_name: str, model_name_or_path: str):
        self._test_model_parallel(
            num_neuron_cores=2,
            tp_size=2,
            model_class_name=model_class_name,
            model_name_or_path=model_name_or_path,
            from_config=False,
            with_lazy_load=False,
            parallelize_embeddings=False,  # Should be True once it's working.
        )

    @unittest.skipIf(
        NUM_NEURON_CORES_AVAILABLE < 32,
        f"This test requires 32 Neuron cores, but only {NUM_NEURON_CORES_AVAILABLE} are available",
    )
    def test_llama_v2_gqa_variants(self):
        # MHA setup
        # TP size = 4, num_attention_heads = 8, num_key_value_heads = 8
        self._test_model_parallel(
            tp_size=4,
            model_class_name="LlamaForCausalLM",
            model_name_or_path="anushehchaudry/llama-2-tiny-random",
            from_config=True,
            with_lazy_load=False,
            parallelize_embeddings=False,
            overwrite_model_config={
                "num_hidden_layers": "2",
                "num_attention_heads": "8",
                "num_key_value_heads": "8",
            },
        )

        # GQA setup with num_key_value_heads > tp_size.
        # TP size = 2, num_attention_heads = 8, num_key_value_heads = 4
        self._test_model_parallel(
            tp_size=2,
            model_class_name="LlamaForCausalLM",
            model_name_or_path="anushehchaudry/llama-2-tiny-random",
            from_config=True,
            with_lazy_load=False,
            parallelize_embeddings=False,
            overwrite_model_config={
                "num_hidden_layers": "2",
                "num_attention_heads": "8",
                "num_key_value_heads": "4",
            },
        )

        # GQA setup with num_key_value_heads = tp_size.
        # TP size = 4, num_attention_heads = 8, num_key_value_heads = 4
        self._test_model_parallel(
            tp_size=4,
            model_class_name="LlamaForCausalLM",
            model_name_or_path="anushehchaudry/llama-2-tiny-random",
            from_config=True,
            with_lazy_load=False,
            parallelize_embeddings=False,
            overwrite_model_config={
                "num_hidden_layers": "2",
                "num_attention_heads": "8",
                "num_key_value_heads": "4",
            },
        )

        # GQA setup with num_key_value_heads < tp_size.
        # TP size = 4, num_attention_heads = 8, num_key_value_heads = 2
        self._test_model_parallel(
            tp_size=4,
            model_class_name="LlamaForCausalLM",
            model_name_or_path="anushehchaudry/llama-2-tiny-random",
            from_config=True,
            with_lazy_load=False,
            parallelize_embeddings=False,
            overwrite_model_config={
                "num_hidden_layers": "2",
                "num_attention_heads": "8",
                "num_key_value_heads": "2",
            },
        )

        # MQA setup
        # TP size = 4, num_attention_heads = 8, num_key_value_heads = 1
        self._test_model_parallel(
            tp_size=4,
            model_class_name="LlamaForCausalLM",
            model_name_or_path="anushehchaudry/llama-2-tiny-random",
            from_config=True,
            with_lazy_load=False,
            parallelize_embeddings=False,
            overwrite_model_config={
                "num_hidden_layers": "2",
                "num_attention_heads": "8",
                "num_key_value_heads": "1",
            },
        )

    # TODO: enable that once it's working.
    # @parameterized.expand(MODELS_TO_TEST)
    # def test_model_parallel_without_parallelizing_embeddings(self, model_class_name: str, model_name_or_path: str):
    #     self._test_model_parallel(model_class_name, model_name_or_path, False, False, False)

    # TODO: enable that.
    # @parameterized.expand(MODELS_TO_TEST)
    # def test_model_parallel_from_pretrained_with_lazy_load(self, model_class_name: str, model_name_or_path: str):
    #     self._test_model_parallel(model_class_name, model_name_or_path, False, True)
