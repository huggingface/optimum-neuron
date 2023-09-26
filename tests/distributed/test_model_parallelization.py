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

import pytest
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
    ("bert", "hf-internal-testing/tiny-random-bert", {"num_hidden_layers": "2"}),
    ("roberta", "hf-internal-testing/tiny-random-roberta", {"num_hidden_layers": "2"}),
    (
        "gpt_neo",
        "hf-internal-testing/tiny-random-GPTNeoModel",
        {
            "num_layers": "2",
        },
    ),
    ("gpt_neox", "hf-tiny-model-private/tiny-random-GPTNeoXModel", {"num_hidden_layers": "2", "intermediate_size": "36"}),
    ("llama", "yujiepan/llama-2-tiny-3layers-random", {"num_hidden_layers": "2"}),
    (
        "t5",
        "hf-internal-testing/tiny-random-T5Model",
        {"d_ff": "36", "num_layers": "2", "num_decoder_layers": "2"},
    ),
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
    OUTPUTS_TO_IGNORE = {
        # It might not match in the sequence parallel setting because of mistmatched shapes.
        # Since these outputs are not needed during training, we do not want to perform an expensive gather for them.
        "encoder_last_hidden_state",
    }

    def _check_output(self, name: str, original_output, output, lazy_load: bool):
        assert type(original_output) is type(output)
        if isinstance(original_output, (tuple, list, set)):
            for idx, orig_output in enumerate(original_output):
                new_name = f"{name}.{idx}"
                self._check_output(new_name, orig_output, output[idx], lazy_load)
        elif isinstance(original_output, dict):
            for output_name in original_output:
                new_name = f"{name}.{output_name}"
                self._check_output(new_name, original_output[name], output[name], lazy_load)
        elif isinstance(original_output, torch.Tensor):
            print(f"Original {name}:\nShape: {original_output.shape}\nValue: {original_output}")
            print(f"Parallel {name}:\nShape: {output.shape}\nValue: {output}")

            # TODO: Remove that once lazy load initializew the weights the same way as no lazy load.
            if not lazy_load:
                torch.testing.assert_close(original_output, output)
        else:
            assert original_output == output, f"Output named {name} do not match."

    def _test_model_parallel(
        self,
        tp_size: int,
        model_class_name: str,
        model_name_or_path: str,
        from_config: bool,
        with_lazy_load: bool,
        parallelize_embeddings: bool,
        sequence_parallel_enabled: bool,
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
            "sequence_parallel_enabled": "true" if sequence_parallel_enabled else "false",
            # TODO: disable that once that loss computation compilation for LLama does not take forever.
            "computing_loss_is_supported": "true" if not model_class_name.startswith("Llama") else "false",
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
                p_original_returncode, stdout = run_command_with_realtime_output(cmd, env=env)
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
                p_original_returncode = p_original.returncode
                stdout = stdout.decode("utf-8")
                full_output = f"Original model standard output:\n{stdout}"
                print(full_output)

                stdout, _ = p_parallel.communicate()
                p_parallel_returncode = p_parallel.returncode
                stdout = stdout.decode("utf-8")
                full_output = f"Parallel model standard output:\n{stdout}"
                print(full_output)

            else:
                p_parallel_returncode, stdout = run_command_with_realtime_output(cmd, env=env)

            assert p_original_returncode == 0
            assert p_parallel_returncode == 0

            temporary_dir = Path(tmpdirname)
            original_model_outputs = torch.load(temporary_dir / "original.bin")
            parallel_model_outputs = torch.load(temporary_dir / "parallel.bin")
            for name, t in original_model_outputs.items():
                if name in self.OUTPUTS_TO_IGNORE:
                    continue
                print(f"Testing that {name} match.")
                regular_parallel_outputs_error_msg = None
                gathered_parallel_outputs_error_msg = None
                try:
                    self._check_output(name, t, parallel_model_outputs[name], with_lazy_load)
                except AssertionError as e:
                    regular_parallel_outputs_error_msg = str(e)
                if regular_parallel_outputs_error_msg is not None:
                    print("Regular output did not match, testing with the gathered output...")
                    try:
                        self._check_output(name, t, parallel_model_outputs[f"gathered_{name}"], with_lazy_load)
                    except AssertionError as e:
                        gathered_parallel_outputs_error_msg = str(e)
                if regular_parallel_outputs_error_msg is not None and gathered_parallel_outputs_error_msg is not None:
                    msg = (
                        "Output did not matched.\nTest with non-gathered parallel outputs error:\n"
                        f"{regular_parallel_outputs_error_msg}\nTest with gathered parallel outputs error:\n"
                        f"{gathered_parallel_outputs_error_msg}"
                    )
                    raise AssertionError(msg)
                print("Ok!")

    @parameterized.expand(MODELS_TO_TEST)
    def test_model_parallel_from_config_no_lazy_load(
        self, model_class_name: str, model_name_or_path: str, config_overwrite: Dict[str, str]
    ):
        self._test_model_parallel(
            num_neuron_cores=8,
            tp_size=2,
            run_test_in_parallel=True,
            model_class_name=model_class_name,
            model_name_or_path=model_name_or_path,
            from_config=True,
            with_lazy_load=False,
            # TODO: enable once ParallelCrossEntropy works.
            # parallelize_embeddings=True,
            parallelize_embeddings=False,
            sequence_parallel_enabled=True,
            overwrite_model_config=config_overwrite,
        )

    @parameterized.expand(MODELS_TO_TEST)
    def test_model_parallel_from_pretrained_no_lazy_load(
        self, model_class_name: str, model_name_or_path: str, config_overwrite: Dict[str, str]
    ):
        self._test_model_parallel(
            num_neuron_cores=8,
            tp_size=2,
            run_test_in_parallel=True,
            model_class_name=model_class_name,
            model_name_or_path=model_name_or_path,
            from_config=False,
            with_lazy_load=False,
            # TODO: enable once ParallelCrossEntropy works.
            # parallelize_embeddings=True,
            parallelize_embeddings=False,
            sequence_parallel_enabled=True,
            overwrite_model_config=config_overwrite,
        )

    @parameterized.expand(MODELS_TO_TEST)
    def test_model_parallel_lazy_load_without_parallelizing_embeddings(
        self, model_class_name: str, model_name_or_path: str, config_overwrite: Dict[str, str]
    ):
        self._test_model_parallel(
            num_neuron_cores=8,
            tp_size=2,
            run_test_in_parallel=True,
            model_class_name=model_class_name,
            model_name_or_path=model_name_or_path,
            from_config=False,
            with_lazy_load=True,
            parallelize_embeddings=False,
            sequence_parallel_enabled=True,
            overwrite_model_config=config_overwrite,
        )

    @parameterized.expand(MODELS_TO_TEST)
    @pytest.mark.skip("Parallel cross entropy does not work yet.")
    def test_model_parallel_lazy_load_without_sequence_parallel(
        self, model_class_name: str, model_name_or_path: str, config_overwrite: Dict[str, str]
    ):
        self._test_model_parallel(
            num_neuron_cores=8,
            tp_size=2,
            run_test_in_parallel=True,
            model_class_name=model_class_name,
            model_name_or_path=model_name_or_path,
            from_config=False,
            with_lazy_load=True,
            parallelize_embeddings=True,
            sequence_parallel_enabled=False,
            overwrite_model_config=config_overwrite,
        )

    @parameterized.expand(MODELS_TO_TEST)
    @pytest.mark.skip("Parallel cross entropy does not work yet.")
    def test_model_parallel_lazy_load_without_anything(
        self, model_class_name: str, model_name_or_path: str, config_overwrite: Dict[str, str]
    ):
        self._test_model_parallel(
            num_neuron_cores=8,
            tp_size=2,
            run_test_in_parallel=True,
            model_class_name=model_class_name,
            model_name_or_path=model_name_or_path,
            from_config=False,
            with_lazy_load=True,
            parallelize_embeddings=False,
            sequence_parallel_enabled=False,
            overwrite_model_config=config_overwrite,
        )

    @pytest.mark.skipif(
        NUM_NEURON_CORES_AVAILABLE < 32,
        reason=f"This test requires 32 Neuron cores, but only {NUM_NEURON_CORES_AVAILABLE} are available",
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
            sequence_parallel_enabled=False,
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
            sequence_parallel_enabled=False,
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
            sequence_parallel_enabled=False,
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
            sequence_parallel_enabled=False,
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
            sequence_parallel_enabled=False,
            overwrite_model_config={
                "num_hidden_layers": "2",
                "hidden_size": "32",
                "num_attention_heads": "16",
                "num_key_value_heads": "1",
            },
        )
