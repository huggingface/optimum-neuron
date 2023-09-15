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

from optimum.neuron.utils.cache_utils import get_num_neuron_cores, set_neuron_cache_path
from optimum.neuron.utils.import_utils import is_neuronx_available
from optimum.neuron.utils.runner import run_command_with_realtime_output

from ..test_utils import is_trainium_test


if TYPE_CHECKING:
    from transformers import PretrainedConfig


TEMPLATE_FILE_NAME = "model_parallel_test_template.txt"
if is_neuronx_available():
    NUM_NEURON_CORES_AVAILABLE = get_num_neuron_cores()
else:
    NUM_NEURON_CORES_AVAILABLE = 0


CLASSES_TO_IGNORE = [
    "T5ForSequenceClassification",
]


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
        if class_name is not None and class_name not in CLASSES_TO_IGNORE:
            model_class_names.append(class_name)

    return list(set(model_class_names))


MODEL_TYPES_TO_TEST = [
    ("bert", "hf-internal-testing/tiny-random-bert"),
    ("roberta", "hf-internal-testing/tiny-random-roberta"),
    ("gpt_neo", "hf-internal-testing/tiny-random-GPTNeoModel"),
    ("llama", "anushehchaudry/llama-2-tiny-random"),
    ("t5", "hf-tiny-model-private/tiny-random-T5ForConditionalGeneration", {"d_ff": "64"}),
]

MODELS_TO_TEST = []
for entry in MODEL_TYPES_TO_TEST:
    if len(entry) == 2:
        model_type, model_name_or_path = entry
        config_overwrite = {}
    else:
        model_type, model_name_or_path, config_overwrite = entry
    for model_class_name in _generate_supported_model_class_names(model_type):
        MODELS_TO_TEST.append((model_class_name, model_name_or_path, config_overwrite))


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
        num_neuron_cores: int = NUM_NEURON_CORES_AVAILABLE,
        run_test_in_parallel: bool = False,
        overwrite_model_config: Optional[Dict[str, str]] = None,
    ):
        if num_neuron_cores < tp_size:
            raise ValueError(
                "The number of Neuron cores available is lower than the TP size, failing since the test might not be "
                "testing what is expected."
            )

        if run_test_in_parallel and (NUM_NEURON_CORES_AVAILABLE // num_neuron_cores) < 2:
            raise ValueError(
                "The test cannot be run in parallel because there is not enough Neuron cores available to preserve the "
                f"number of Neuron cores requested ({NUM_NEURON_CORES_AVAILABLE} cores available and {num_neuron_cores} "
                "were requested)"
            )

        template_content = None
        current_directory = Path(__file__).parent.resolve()
        template_file_path = current_directory / TEMPLATE_FILE_NAME
        with open(template_file_path, "r") as fp:
            template_content = fp.read()

        specialization_env = {
            "from_config": "true" if from_config else "false",
            "lazy_load": "true" if with_lazy_load else "false",
            "parallelize_embeddings": "true" if parallelize_embeddings else "false",
            **os.environ,
        }

        # Updating the Python path to be able to use `tests/distributed/utils.py`.
        python_path = specialization_env.get("PYTHONPATH", "")
        python_path = f"{current_directory}:{python_path}"
        specialization_env["PYTHONPATH"] = python_path

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

            # When running the test in parallel, we need 2 rendez-vous endpoints: one for the script running the
            # original model and one for the script running the parallel model.
            rdzv_endpoint_host = "localhost"
            rdzv_endpoint_port = 29400

            orig_neuron_cc_flags = os.environ.get("NEURON_CC_FLAGS", "")
            set_neuron_cache_path(tmpdirname)
            neuron_cc_flags = os.environ["NEURON_CC_FLAGS"]
            os.environ["NEURON_CC_FLAGS"] = orig_neuron_cc_flags

            # Original model.
            env = {"is_parallel": "false", **specialization_env, "NEURON_CC_FLAGS": neuron_cc_flags}
            if run_test_in_parallel:
                # Setting the rendez-vous endpoint for the original model process.
                cmd.insert(1, f"--rdzv_endpoint={rdzv_endpoint_host}:{rdzv_endpoint_port}")
                env["NEURON_RT_VISIBLE_CORES"] = f"0-{num_neuron_cores - 1}"


            # When running tests in parallel, synchronization is done after both processes started.
            if not run_test_in_parallel:
                _, stdout = run_command_with_realtime_output(cmd, env=env)
            else:
                p_original = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)

            # Parallel model.
            env = {"is_parallel": "true", **specialization_env, "NEURON_CC_FLAGS": neuron_cc_flags}
            if run_test_in_parallel:
                # Updating the rendez-vous endpoint for the parallel model process.
                cmd[1] = f"--rdzv_endpoint={rdzv_endpoint_host}:{rdzv_endpoint_port + 1}"
                env["NEURON_RT_VISIBLE_CORES"] = f"{num_neuron_cores}-{2 * num_neuron_cores - 1}"

                p_parallel = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)

                stdout, _ = p_original.communicate()
                stdout = stdout.decode("utf-8")
                full_output = f"Original model standard output:\n{stdout}"
                print(full_output)

                stdout, _ = p_parallel.communicate()
                stdout = stdout.decode("utf-8")
                full_output = f"Parallel model standard output:\n{stdout}"
                print(full_output)

            else:
                _, stdout = run_command_with_realtime_output(cmd, env=env)


            temporary_dir = Path(tmpdirname)
            original_model_outputs = torch.load(temporary_dir / "original.bin")
            parallel_model_outputs = torch.load(temporary_dir / "parallel.bin")
            for name, t in parallel_model_outputs.items():
                if not isinstance(t, torch.Tensor):
                    continue
                original_t = original_model_outputs[name]
                print(f"Testing that {name} match.")
                print(f"Original {name}:\nShape: {original_t.shape}\nValue: {original_t}")
                print(f"Parallel {name}:\nShape: {t.shape}\nValue: {t}")
                print(t, original_t)
                torch.testing.assert_close(t, original_t)
                print("Ok!")

    @parameterized.expand(MODELS_TO_TEST)
    def test_model_parallel_from_config_without_lazy_load(
        self, model_class_name: str, model_name_or_path: str, config_overwrite: Dict[str, str]
    ):
        self._test_model_parallel(
            num_neuron_cores=8,
            tp_size=2,
            run_test_in_parallel=False,
            model_class_name=model_class_name,
            model_name_or_path=model_name_or_path,
            from_config=True,
            with_lazy_load=False,
            parallelize_embeddings=True,
            overwrite_model_config=config_overwrite,
        )

    @parameterized.expand(MODELS_TO_TEST)
    def test_model_parallel_from_pretrained_without_lazy_load(
        self, model_class_name: str, model_name_or_path: str, config_overwrite: Dict[str, str]
    ):
        self._test_model_parallel(
            num_neuron_cores=8,
            tp_size=2,
            model_class_name=model_class_name,
            model_name_or_path=model_name_or_path,
            from_config=False,
            with_lazy_load=False,
            parallelize_embeddings=True,
            overwrite_model_config=config_overwrite,
        )

    @parameterized.expand(MODELS_TO_TEST)
    def test_model_parallel_without_parallelizing_embeddings(
        self, model_class_name: str, model_name_or_path: str, config_overwrite: Dict[str, str]
    ):
        self._test_model_parallel(
            num_neuron_cores=8,
            tp_size=2,
            model_class_name=model_class_name,
            model_name_or_path=model_name_or_path,
            from_config=False,
            with_lazy_load=True,
            parallelize_embeddings=False,
            overwrite_model_config=config_overwrite,
        )

    @unittest.skipIf(
        NUM_NEURON_CORES_AVAILABLE < 32,
        f"This test requires 32 Neuron cores, but only {NUM_NEURON_CORES_AVAILABLE} are available",
    )
    def test_llama_v2_gqa_variants(self):
        llama_v2_model_name = "anushehchaudry/llama-2-tiny-random"
        # MHA setup
        # TP size = 2, num_attention_heads = 8, num_key_value_heads = 8
        self._test_model_parallel(
            num_neuron_cores=8,
            tp_size=2,
            run_test_in_parallel=True,
            model_class_name="LlamaForCausalLM",
            model_name_or_path=llama_v2_model_name,
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
            num_neuron_cores=8,
            tp_size=2,
            run_test_in_parallel=True,
            model_class_name="LlamaForCausalLM",
            model_name_or_path=llama_v2_model_name,
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
        # TP size = 8, num_attention_heads = 16, num_key_value_heads = 8
        self._test_model_parallel(
            num_neuron_cores=8,
            tp_size=8,
            run_test_in_parallel=True,
            model_class_name="LlamaForCausalLM",
            model_name_or_path=llama_v2_model_name,
            from_config=True,
            with_lazy_load=False,
            parallelize_embeddings=False,
            overwrite_model_config={
                "num_hidden_layers": "2",
                "hidden_size": "32",
                "num_attention_heads": "16",
                "num_key_value_heads": "8",
            },
        )

        # GQA setup with num_key_value_heads < tp_size.
        # TP size = 8, num_attention_heads = 16, num_key_value_heads = 2
        self._test_model_parallel(
            num_neuron_cores=8,
            tp_size=8,
            run_test_in_parallel=True,
            model_class_name="LlamaForCausalLM",
            model_name_or_path=llama_v2_model_name,
            from_config=True,
            with_lazy_load=False,
            parallelize_embeddings=False,
            overwrite_model_config={
                "num_hidden_layers": "2",
                "hidden_size": "32",
                "num_attention_heads": "16",
                "num_key_value_heads": "2",
            },
        )

        # MQA setup
        # TP size = 8, num_attention_heads = 16, num_key_value_heads = 1
        self._test_model_parallel(
            num_neuron_cores=8,
            tp_size=8,
            run_test_in_parallel=True,
            model_class_name="LlamaForCausalLM",
            model_name_or_path=llama_v2_model_name,
            from_config=True,
            with_lazy_load=False,
            parallelize_embeddings=False,
            overwrite_model_config={
                "num_hidden_layers": "2",
                "hidden_size": "32",
                "num_attention_heads": "16",
                "num_key_value_heads": "1",
            },
        )
