# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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
"""Tests every (architecture, task) supported pair on 🤗 Transformers training example scripts."""

import json
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Optional, Set, Union
from unittest import TestCase

from huggingface_hub import HfFolder
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_CTC_MAPPING,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
)
from transformers.testing_utils import slow

from optimum.neuron.utils.cache_utils import (
    load_custom_cache_repo_name_from_hf_home,
    set_custom_cache_repo_name_in_hf_home,
    set_neuron_cache_path,
)
from optimum.neuron.utils.testing_utils import is_trainium_test


# Doing it this way to be able to use this file in tools.
path_tests = Path(__file__).parent
sys.path.insert(0, str(path_tests))
from utils import MODELS_TO_TEST_MAPPING  # noqa: E402


TOKEN = HfFolder.get_token()
if os.environ.get("HF_TOKEN_OPTIMUM_NEURON_CI", None) is not None:
    TOKEN = os.environ.get("HF_TOKEN_OPTIMUM_NEURON_CI")

CACHE_REPO_NAME = "optimum-internal-testing/optimum-neuron-cache-for-testing"


def _get_supported_models_for_script(
    models_to_test: Dict[str, str], task_mapping: Dict[str, str], to_exclude: Optional[Set[str]] = None
) -> List[str]:
    """
    Filters models that can perform the task from models_to_test.
    """
    if to_exclude is None:
        to_exclude = set()
    supported_models = []
    for model_type, model_name in models_to_test.items():
        if model_type in to_exclude:
            continue
        if CONFIG_MAPPING[model_type] in task_mapping:
            if model_type == "bart" and task_mapping is not MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING:
                continue
            supported_models.append((model_type, model_name))
    return supported_models


_SCRIPT_TO_MODEL_MAPPING = {
    "run_clm": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING,
        MODEL_FOR_CAUSAL_LM_MAPPING,
        to_exclude={"bart", "bert", "camembert", "electra", "roberta", "xlm-roberta"},
    ),
    "run_mlm": _get_supported_models_for_script(MODELS_TO_TEST_MAPPING, MODEL_FOR_MASKED_LM_MAPPING),
    "run_swag": _get_supported_models_for_script(MODELS_TO_TEST_MAPPING, MODEL_FOR_MULTIPLE_CHOICE_MAPPING),
    "run_qa": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING, MODEL_FOR_QUESTION_ANSWERING_MAPPING, to_exclude={"bart"}
    ),
    "run_summarization": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING, MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING, to_exclude={"marian", "m2m_100"}
    ),
    "run_translation": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING, MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING
    ),
    "run_glue": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING, MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING, to_exclude={"bart", "gpt2", "gpt_neo"}
    ),
    "run_ner": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, to_exclude={"gpt2"}
    ),
    "run_image_classification": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING, MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING
    ),
    "run_audio_classification": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING, MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING
    ),
    "run_speech_recognition_ctc": _get_supported_models_for_script(MODELS_TO_TEST_MAPPING, MODEL_FOR_CTC_MAPPING),
}


class ExampleTestMeta(type):
    """
    Metaclass that takes care of creating the proper example tests for a given task.

    It uses example_name to figure out which models support this task, and create a run example test for each of these
    models.
    """

    def __new__(cls, name, bases, attrs, example_name=None):
        models_to_test = []
        if example_name is not None:
            models_to_test = _SCRIPT_TO_MODEL_MAPPING.get(example_name)
            if models_to_test is None:
                raise AttributeError(f"could not create class because no model was found for example {example_name}")
        for model_type, model_name in models_to_test:
            attrs[f"test_{example_name}_{model_type}"] = cls._create_test(model_type, model_name)
        attrs["EXAMPLE_NAME"] = example_name
        return super().__new__(cls, name, bases, attrs)

    @staticmethod
    def process_class_attribute(attribute: Union[Any, Dict[str, Any]], model_type: str) -> Any:
        if isinstance(attribute, dict):
            return attribute.get(model_type, attribute["default"])
        return attribute

    @classmethod
    def _create_test(cls, model_type: str, model_name: str) -> Callable[["ExampleTesterBase"], None]:
        """
        Creates a test function that runs an example for a model_name.

        Args:
            model_name (`str`): the model_name_or_path.

        Returns:
            `Callable[[ExampleTesterBase], None]`: The test function that runs the example.
        """

        @slow
        @is_trainium_test
        def test(self):
            if self.EXAMPLE_NAME is None:
                raise ValueError("An example name must be provided")
            example_script = Path(self.EXAMPLE_DIR).glob(f"*/{self.EXAMPLE_NAME}.py")
            example_script = list(example_script)
            if len(example_script) == 0:
                raise RuntimeError(f"Could not find {self.EXAMPLE_NAME}.py in examples located in {self.EXAMPLE_DIR}")
            elif len(example_script) > 1:
                raise RuntimeError(f"Found more than {self.EXAMPLE_NAME}.py in examples located in {self.EXAMPLE_DIR}")
            else:
                example_script = example_script[0]

            self._install_requirements(example_script.parent / "requirements.txt")

            do_precompilation = ExampleTestMeta.process_class_attribute(self.DO_PRECOMPILATION, model_type)
            only_precompilation = ExampleTestMeta.process_class_attribute(self.ONLY_PRECOMPILATION, model_type)

            eval_is_supported = ExampleTestMeta.process_class_attribute(self.EVAL_IS_SUPPORTED, model_type)
            eval_score_threshold = ExampleTestMeta.process_class_attribute(self.EVAL_SCORE_THRESHOLD, model_type)

            env = self.get_env(model_type)
            if do_precompilation:
                with TemporaryDirectory(dir=Path(self.EXAMPLE_DIR)) as tmp_dir:
                    cmd_line = self._create_command_line(
                        example_script,
                        model_name,
                        model_type,
                        tmp_dir,
                        is_precompilation=True,
                    )
                    joined_cmd_line = " ".join(cmd_line)
                    print(f"#### Running precompilation... ####\n{joined_cmd_line}\n")
                    p = subprocess.Popen(joined_cmd_line, shell=True, env=env)
                    return_code = p.wait()
                    self.assertEqual(return_code, 0)

            if not only_precompilation:
                with TemporaryDirectory(dir=Path(self.EXAMPLE_DIR)) as tmp_dir:
                    cmd_line = self._create_command_line(
                        example_script,
                        model_name,
                        model_type,
                        tmp_dir,
                    )
                    joined_cmd_line = " ".join(cmd_line)
                    print(f"#### Running command line... ####\n{joined_cmd_line}\n")
                    os.environ["WANDB_NAME"] = f"{self.EXAMPLE_NAME}_{model_type}"
                    p = subprocess.Popen(joined_cmd_line, shell=True, env=env)
                    return_code = p.wait()
                    self.assertEqual(return_code, 0)

                    if eval_is_supported:
                        with open(Path(tmp_dir) / "all_results.json") as fp:
                            results = json.load(fp)
                        threshold_overrides = {}
                        threshold = threshold_overrides.get(model_name, eval_score_threshold)
                        if self.EVAL_SCORE_GREATER_IS_BETTER:
                            self.assertGreaterEqual(float(results[self.SCORE_NAME]), threshold)
                        else:
                            self.assertLessEqual(float(results[self.SCORE_NAME]), threshold)

        return test


class ExampleTesterBase(TestCase):
    """
    Base example tester class.

    Attributes:
        EXAMPLE_DIR (`Union[str, Path]`) -- The directory containing the examples.
        EXAMPLE_NAME (`Optional[str]`) -- The name of the example script without the file extension, e.g. run_qa, run_glue, etc.
        TASK_NAME (`str`) -- The name of the dataset to use.
        EVAL_IS_SUPPORTED (`bool`) -- Whether evaluation is currently supported on AWS Tranium.
            If True, the example will run evaluation, otherwise it will be skipped.
        EVAL_SCORE_THRESHOLD (`float`) -- The score threshold from which training is assumed to have worked.
        SCORE_NAME (`str`) -- The name of the metric to use for checking that the example ran successfully.
        DATASET_PARAMETER_NAME (`str`) -- The argument name to use for the dataset parameter.
            Most of the time it will be "dataset_name", but for some tasks on a benchmark it might be something else.
        TRAIN_BATCH_SIZE (`int`) -- The batch size to give to the example script for training.
        EVAL_BATCH_SIZE (`int`) -- The batch size to give to the example script for evaluation.
        GRADIENT_ACCUMULATION_STEPS (`int`) -- The number of gradient accumulation to use during training.
        DATALOADER_DROP_LAST (`bool`) -- Whether to drop the last batch if it is a remainder batch.
        NPROC_PER_NODE (`int`) -- The number of Neuron cores to use when doing multiple workers training.
        EXTRA_COMMAND_LINE_ARGUMENTS (`Optional[List[str]]`) -- Extra arguments, if needed, to be passed to the command line traning
            script.
    """

    EXAMPLE_DIR = Path(__file__).parent.parent / "examples"
    EXAMPLE_NAME = ""
    TASK_NAME = None
    DATASET_CONFIG_NAME = ""
    EVAL_IS_SUPPORTED = True
    # Camembert is pretrained on French.
    EVAL_SCORE_THRESHOLD = {"default": 0.75, "camembert": 0.5}
    EVAL_SCORE_GREATER_IS_BETTER = True
    SCORE_NAME = "eval_accuracy"
    DATASET_PARAMETER_NAME = "dataset_name"
    NUM_EPOCHS = 1
    MAX_STEPS = None
    LEARNING_RATE = 1e-4
    TRAIN_BATCH_SIZE = 16
    EVAL_BATCH_SIZE = 16
    GRADIENT_ACCUMULATION_STEPS = 16
    NPROC_PER_NODE = 2
    EXTRA_COMMAND_LINE_ARGUMENTS = None
    LOGGING_STEPS = 1
    SAVE_STEPS = 200
    ONLY_PRECOMPILATION = False
    DO_PRECOMPILATION = False
    NEURON_CACHE = None
    MULTI_PROC = os.environ.get("MULTI_PROC", "false")
    BF16 = True

    @classmethod
    def setUpClass(cls):
        cls._create_venv()
        cls._orig_token = HfFolder.get_token()
        cls._orig_cache_repo = load_custom_cache_repo_name_from_hf_home()
        HfFolder.save_token(TOKEN)
        set_custom_cache_repo_name_in_hf_home(CACHE_REPO_NAME)

    @classmethod
    def tearDownClass(cls):
        cls._remove_venv()
        if cls._orig_token is not None:
            HfFolder.save_token(cls._orig_token)
        if cls._orig_cache_repo is not None:
            set_custom_cache_repo_name_in_hf_home(cls._orig_cache_repo)

    def setUp(self):
        set_neuron_cache_path("/var/tmp/neuron-compile-cache")

    def tearDown(self):
        set_neuron_cache_path("/var/tmp/neuron-compile-cache")

        cmd_line = "sudo rmmod neuron".split()
        p = subprocess.Popen(cmd_line)
        return_code = p.wait()
        assert return_code == 0

        cmd_line = "sudo modprobe neuron".split()
        p = subprocess.Popen(cmd_line)
        return_code = p.wait()
        assert return_code == 0

    def get_env(self, model_type: str) -> Dict[str, str]:
        env = dict(os.environ)
        return env

    def _create_command_line(
        self,
        script: str,
        model_name: str,
        model_type: str,
        output_dir: str,
        is_precompilation: bool = False,
    ) -> List[str]:
        # Task related.
        task = ExampleTestMeta.process_class_attribute(self.TASK_NAME, model_type)
        dataset_parameter_name = ExampleTestMeta.process_class_attribute(self.DATASET_PARAMETER_NAME, model_type)
        dataset_config_name = ExampleTestMeta.process_class_attribute(self.DATASET_CONFIG_NAME, model_type)

        # Batch size related.
        train_batch_size = ExampleTestMeta.process_class_attribute(self.TRAIN_BATCH_SIZE, model_type)
        eval_batch_size = ExampleTestMeta.process_class_attribute(self.EVAL_BATCH_SIZE, model_type)
        gradient_accumulation_steps = ExampleTestMeta.process_class_attribute(
            self.GRADIENT_ACCUMULATION_STEPS, model_type
        )

        # Training related.
        learning_rate = ExampleTestMeta.process_class_attribute(self.LEARNING_RATE, model_type)
        eval_is_supported = ExampleTestMeta.process_class_attribute(self.EVAL_IS_SUPPORTED, model_type)
        n_proc_per_node = ExampleTestMeta.process_class_attribute(self.NPROC_PER_NODE, model_type)
        num_train_epochs = ExampleTestMeta.process_class_attribute(self.NUM_EPOCHS, model_type)
        max_steps = ExampleTestMeta.process_class_attribute(self.MAX_STEPS, model_type)
        logging_steps = ExampleTestMeta.process_class_attribute(self.LOGGING_STEPS, model_type)
        save_steps = ExampleTestMeta.process_class_attribute(self.SAVE_STEPS, model_type)

        bf16 = ExampleTestMeta.process_class_attribute(self.BF16, model_type)
        multi_proc = ExampleTestMeta.process_class_attribute(self.MULTI_PROC, model_type)

        # Extra
        extra_command_line_arguments = []
        if self.EXTRA_COMMAND_LINE_ARGUMENTS is not None:
            extra_command_line_arguments = [
                ExampleTestMeta.process_class_attribute(arg, model_type) for arg in self.EXTRA_COMMAND_LINE_ARGUMENTS
            ]

        do_eval = eval_is_supported and not is_precompilation

        do_eval_option = "--do_eval" if do_eval else " "
        task_option = f"--{dataset_parameter_name} {task}" if task else " "

        if multi_proc == "false":
            program = ["venv/bin/python" if self.venv_was_created() else "python"]
        else:
            program = [
                "venv/bin/torchrun" if self.venv_was_created() else "torchrun",
                f"--nproc_per_node={n_proc_per_node}",
            ]

        if is_precompilation:
            neuron_parallel_compile_path = (
                "venv/bin/neuron_parallel_compile" if self.venv_was_created() else "neuron_parallel_compile"
            )
            program = [neuron_parallel_compile_path] + program

        if max_steps is not None:
            max_steps = f"--max_steps {max_steps}"
        else:
            max_steps = ""

        cmd_line = program + [
            f"{script}",
            f"--model_name_or_path {model_name}",
            f"{task_option}",
            "--do_train",
            f"{do_eval_option}",
            f"--output_dir {output_dir}",
            "--overwrite_output_dir true",
            f"--learning_rate {learning_rate}",
            f"--per_device_train_batch_size {train_batch_size}",
            f"--per_device_eval_batch_size {eval_batch_size}",
            f"--gradient_accumulation_steps {gradient_accumulation_steps}",
            "--save_strategy steps",
            f" --num_train_epochs {num_train_epochs}",
            max_steps,
            "--dataloader_num_workers 4",
            f"--save_steps {save_steps}",
            "--save_total_limit 1",
            f"--logging_steps {logging_steps}",
        ]
        if bf16:
            cmd_line.append("--bf16")
        if is_precompilation:
            cmd_line.append("--report_to none")

        if dataset_config_name:
            cmd_line.append(f"--dataset_config_name {dataset_config_name}")

        if extra_command_line_arguments is not None:
            cmd_line += extra_command_line_arguments

        pattern = re.compile(r"([\"\'].+?[\"\'])|\s")
        return [x for y in cmd_line for x in re.split(pattern, y) if x]

    @classmethod
    def venv_was_created(cls):
        return os.environ.get("USE_VENV", "true") == "true" and os.path.isdir("venv")

    @classmethod
    def _create_venv(cls):
        """
        Creates the virtual environment for the example.
        """
        if os.environ.get("USE_VENV", "true") == "true":
            cmd_line = "python -m venv venv".split()
            p = subprocess.Popen(cmd_line)
            return_code = p.wait()
            assert return_code == 0

            # Install pip
            cmd_line = "venv/bin/python -m ensurepip --upgrade".split()
            p = subprocess.Popen(cmd_line)
            return_code = p.wait()
            assert return_code == 0

    @classmethod
    def _remove_venv(cls):
        """
        Removes the virtual environment for the example.
        """
        if cls.venv_was_created():
            cmd_line = "rm -rf venv".split()
            p = subprocess.Popen(cmd_line)
            return_code = p.wait()
            assert return_code == 0

    def _install_requirements(self, requirements_filename: Union[str, os.PathLike]):
        """
        Installs the necessary requirements to run the example if the provided file exists, otherwise does nothing.
        """

        pip_name = "venv/bin/pip" if self.venv_was_created() else "pip"

        # Update pip
        cmd_line = f"{pip_name} install --upgrade pip".split()
        p = subprocess.Popen(cmd_line)
        return_code = p.wait()
        self.assertEqual(return_code, 0)

        # Set pip repository pointing to the Neuron repository
        cmd_line = f"{pip_name} config set global.extra-index-url https://pip.repos.neuron.amazonaws.com".split()
        p = subprocess.Popen(cmd_line)
        return_code = p.wait()
        self.assertEqual(return_code, 0)

        # Install wget, awscli, Neuron Compiler and Neuron Framework
        cmd_line = f"{pip_name} freeze | grep torch-neuronx".split()
        p = subprocess.Popen(cmd_line, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        outs, _ = p.communicate()
        if outs is not None:
            cmd_line = f"{pip_name} install wget awscli neuronx-cc==2.* torch-neuronx torchvision".split()
            p = subprocess.Popen(cmd_line)
            return_code = p.wait()
            self.assertEqual(return_code, 0)

        cmd_line = f"{pip_name} install -e {Path(__file__).parent.parent}".split()
        p = subprocess.Popen(cmd_line)
        return_code = p.wait()
        self.assertEqual(return_code, 0)

        # Install requirements
        if Path(requirements_filename).exists():
            cmd_line = f"{pip_name} install -r {requirements_filename}".split()
            p = subprocess.Popen(cmd_line)
            return_code = p.wait()
            self.assertEqual(return_code, 0)

        # TODO: remove that as soon as possible.
        cmd_line = f"{pip_name} install numpy==1.20.3".split()
        p = subprocess.Popen(cmd_line)
        return_code = p.wait()
        self.assertEqual(return_code, 0)

        # Potentially install WANDB
        wandb_token = os.environ.get("WANDB_TOKEN", None)
        if wandb_token is not None:
            cmd_line = f"{pip_name} install wandb".split()
            p = subprocess.Popen(cmd_line)
            return_code = p.wait()
            self.assertEqual(return_code, 0)

            env_with_updated_path = dict(os.environ, PATH=f"/home/ubuntu/.local/bin:{os.environ['PATH']}")

            wandb_name = "venv/bin/wandb" if self.venv_was_created() else "wandb"
            cmd_line = f"{wandb_name} login --relogin {wandb_token}".split()
            p = subprocess.Popen(cmd_line, env=env_with_updated_path)
            self.assertEqual(return_code, 0)

            wandb_project_name = os.environ.get("WANDB_PROJECT", "aws-neuron-tests")
            today = date.today().strftime("%d%m%Y")
            wandb_project_name = f"{wandb_project_name}-{today}"
            os.environ["WANDB_PROJECT"] = wandb_project_name
            cmd_line = f"{wandb_name} init -p {wandb_project_name}".split()
            p = subprocess.Popen(cmd_line, env=env_with_updated_path)
            self.assertEqual(return_code, 0)


class CausalLMExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_clm"):
    TASK_NAME = "wikitext"
    DATASET_CONFIG_NAME = "wikitext-2-raw-v1"
    NUM_EPOCHS = 1
    TRAIN_BATCH_SIZE = {"default": 2, "gpt2": 1}
    EVAL_BATCH_SIZE = 2
    SCORE_NAME = "random_test"
    EVAL_SCORE_THRESHOLD = 35
    EVAL_SCORE_GREATER_IS_BETTER = False


class TextClassificationExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_glue"):
    TASK_NAME = "sst2"
    DATASET_PARAMETER_NAME = "task_name"
    NUM_EPOCHS = 1


class TokenClassificationExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_ner"):
    TASK_NAME = "conll2003"
    TRAIN_BATCH_SIZE = {"default": 4, "distilbert": 6}
    EVAL_BATCH_SIZE = {"default": 4, "distilbert": 6}
    NUM_EPOCHS = 1
    EXTRA_COMMAND_LINE_ARGUMENTS = [
        "--max_seq_length 384",
    ]


class MultipleChoiceExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_swag"):
    EVAL_SCORE_THRESHOLD = {"default": 0.75, "camembert": 0.5, "distilbert": 0.645}
    TRAIN_BATCH_SIZE = {"default": 2, "distilbert": 3}
    EVAL_BATCH_SIZE = {"default": 2, "distilbert": 3}
    MAX_STEPS = 100
    NUM_EPOCHS = 1
    EXTRA_COMMAND_LINE_ARGUMENTS = [
        "--max_eval_samples 840",
        "--max_seq_length 512",
    ]


class QuestionAnsweringExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_qa"):
    TASK_NAME = "squad"
    SCORE_NAME = "eval_f1"
    NUM_EPOCHS = 1


class SummarizationExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_summarization"):
    TASK_NAME = "cnn_dailymail"
    DATASET_CONFIG_NAME = "3.0.0"
    TRAIN_BATCH_SIZE = 1
    EVAL_BATCH_SIZE = 1
    MAX_STEPS = 200
    EVAL_IS_SUPPORTED = False
    EVAL_SCORE_THRESHOLD = 30
    SCORE_NAME = "eval_rougeLsum"
    EXTRA_COMMAND_LINE_ARGUMENTS = [
        "--prediction_loss_only",
        "--pad_to_max_length",
        "--max_target_length 200",
        {"default": "--max_source_length 1024", "t5": "--max_source_length 768"},
        {"default": "", "t5": "--source_prefix 'summarize: '"},
    ]

    def _create_command_line(
        self,
        script: str,
        model_name: str,
        model_type: str,
        output_dir: str,
        is_precompilation: bool = False,
    ) -> List[str]:
        extra_command_line_arguments = [
            ExampleTestMeta.process_class_attribute(arg, model_type) for arg in self.EXTRA_COMMAND_LINE_ARGUMENTS
        ]
        if extra_command_line_arguments is None:
            extra_command_line_arguments = []
        return super()._create_command_line(
            script,
            model_name,
            model_type,
            output_dir,
            is_precompilation=is_precompilation,
        )


class TranslationExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_translation"):
    TASK_NAME = "wmt16"
    DATASET_CONFIG_NAME = "ro-en"
    TRAIN_BATCH_SIZE = 1
    EVAL_BATCH_SIZE = 1
    MAX_STEPS = 200
    EVAL_IS_SUPPORTED = False
    EVAL_SCORE_THRESHOLD = 22
    SCORE_NAME = "eval_bleu"
    EXTRA_COMMAND_LINE_ARGUMENTS = [
        "--source_lang ro",
        "--target_lang en",
        "--pad_to_max_length",
        {"default": "--max_source_length 512", "m2m_100": "--max_source_length 128"},
        {"default": "--max_target_length 512", "m2m_100": "--max_target_length 128"},
        # "--max_source_length 512",
        # "--max_target_length 512",
        "--prediction_loss_only",
    ]

    def _create_command_line(
        self,
        script: str,
        model_name: str,
        model_type: str,
        output_dir: str,
        is_precompilation: bool = False,
    ) -> List[str]:
        extra_command_line_arguments = [
            ExampleTestMeta.process_class_attribute(arg, model_type) for arg in self.EXTRA_COMMAND_LINE_ARGUMENTS
        ]
        if extra_command_line_arguments is None:
            extra_command_line_arguments = []
        if "t5" in model_name:
            extra_command_line_arguments.append("--source_prefix 'translate English to Romanian: '")
        return super()._create_command_line(
            script,
            model_name,
            model_type,
            output_dir,
            is_precompilation=is_precompilation,
        )


class ImageClassificationExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_image_classification"
):
    TASK_NAME = "cifar10"
    NUM_EPOCHS = 2
    EXTRA_COMMAND_LINE_ARGUMENTS = [
        "--remove_unused_columns false",
        "--dataloader_drop_last true",
        "--ignore_mismatched_sizes",
    ]


# class AudioClassificationExampleTester(
#     ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_audio_classification"
# ):
#     TASK_NAME = "superb"
#     DATASET_CONFIG_NAME = "ks"
#     GRADIENT_ACCUMULATION_STEPS = 16
#     EXTRA_COMMAND_LINE_ARGUMENTS = ["--max_length_seconds 1", "--attention_mask False"]
#     LEARNING_RATE = 3e-5
#
#
# class SpeechRecognitionExampleTester(
#     ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_speech_recognition_ctc"
# ):
#     TASK_NAME = "common_voice"
#     DATASET_CONFIG_NAME = "tr"
#     TRAIN_BATCH_SIZE = 1
#     GRADIENT_ACCUMULATION_STEPS = 8
#     EVAL_BATCH_SIZE = 1
#     NUM_EPOCHS = 15
#     # Here we are evaluating against the loss because it can take a long time to have wer < 1.0
#     SCORE_NAME = "eval_loss"
#     EVAL_SCORE_THRESHOLD = 4
#     EVAL_SCORE_GREATER_IS_BETTER = False
#     EXTRA_COMMAND_LINE_ARGUMENTS = [
#         "--learning_rate 3e-4",
#         "--warmup_steps 400",
#         "--mask_time_prob 0.0",
#         "--layerdrop 0.0",
#         "--freeze_feature_encoder",
#         "--text_column_name sentence",
#         "--length_column_name input_length",
#         '--chars_to_ignore , ? . ! - \\; \\: \\" “ % ‘ ” � ',
#     ]
