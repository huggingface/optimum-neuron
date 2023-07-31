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
"""Utilities to be able to perform model compilation easily."""

import os
import re
import subprocess
from enum import Enum
from pathlib import Path
from subprocess import PIPE
from tempfile import TemporaryDirectory
from typing import List, Optional, Tuple, Union

import requests
from huggingface_hub import HfFolder

from ...utils import logging
from .cache_utils import get_hf_hub_cache_repos, has_write_access_to_repo, load_custom_cache_repo_name_from_hf_home


logger = logging.get_logger()

_BASE_RAW_FILES_PATH_IN_GH_REPO = "https://raw.githubusercontent.com/huggingface/optimum-neuron/"
_GH_REPO_EXAMPLE_FOLDERS = [
    "audio-classification",
    "image-classification",
    "language-modeling",
    "multiple-choice",
    "question-answering",
    "summarization",
    "text-classification",
    "token-classification",
    "translation",
]

_TASK_TO_EXAMPLE_SCRIPT = {
    "causal-lm": "run_clm",
    "masked-lm": "run_mlm",
    "text-classification": "run_glue",
    "token-classification": "run_ner",
    "multiple-choice": "run_swag",
    "question-answering": "run_qa",
    "summarization": "run_summarization",
    "translation": "run_translation",
    "image-classification": "run_image_classification",
    "audio-classification": "run_audio_classification",
    "speech-recognition": "run_speech_recognition_ctc",
}


def download_example_script_from_github(task_name: str, target_directory: Path, revision: str = "main") -> Path:
    # TODO: test that every existing task can be downloaded.
    script_name = f"{_TASK_TO_EXAMPLE_SCRIPT[task_name]}.py"
    example_script_path = target_directory / script_name
    was_saved = False
    for folder in _GH_REPO_EXAMPLE_FOLDERS:
        url = f"{_BASE_RAW_FILES_PATH_IN_GH_REPO}/{revision}/examples/{folder}/{script_name}"
        r = requests.get(url)
        if r.status_code != 200:
            continue
        with open(example_script_path, "w") as fp:
            fp.write(r.text)
        was_saved = True
    if not was_saved:
        raise FileNotFoundError(f"Could not find an example script for the task {task_name} on the GitHub repo")

    return example_script_path


class Precision(str, Enum):
    fp = "fp"
    bf16 = "bf16"


class ExampleRunner:
    _TASK_TO_COMMAND_ARGUMENTS = {
        "masked-lm": {
            "dataset_name": "wikitext",
            "dataset_config_name": "wikitext-2-raw-v1",
            "set_max_length": True,
            "extra_command_line_arguments": [
                "--pad_to_max_length",
            ],
        },
        "causal-lm": {
            "dataset_name": "wikitext",
            "dataset_config_name": "wikitext-2-raw-v1",
        },
        "text-classification": {
            "task_name": "sst2",
        },
        "token-classification": {
            "dataset_name": "conll2003",
            "set_max_length": True,
            "extra_command_line_arguments": [
                "--pad_to_max_length",
                "--ignore_mismatched_sizes",
            ],
        },
        "multiple-choice": {
            "set_max_length": True,
            "extra_command_line_arguments": [
                "--pad_to_max_length",
                # "--ignore_mismatched_sizes",
            ],
        },
        "question-answering": {
            "dataset_name": "squad",
            # It is already the case, but just to make sure if it ever changes.
            "set_max_length": True,
            "extra_command_line_arguments": [
                "--pad_to_max_length",
            ],
        },
        "summarization": {
            "dataset_name": "cnn_dailymail",
            "dataset_config_name": "3.0.0",
            "set_max_source_length": True,
            "set_max_target_length": True,
            "extra_command_line_arguments": [
                "--pad_to_max_length",
            ],
        },
        "translation": {
            "dataset_name": "wmt16",
            "dataset_config_name": "ro-en",
            "set_max_source_length": True,
            "set_max_target_length": True,
            "extra_command_line_arguments": [
                "--source_lang ro",
                "--target_lang en",
                "--pad_to_max_length",
                "--prediction_loss_only",
            ],
        },
        "image-classification": {
            "dataset_name": "cifar10",
            "extra_command_line_arguments": [
                "--remove_unused_columns false",
                "--ignore_mismatched_sizes",
            ],
        },
    }

    def __init__(
        self, model_name_or_path: str, task: str, example_dir: Optional[Union[str, Path]] = None, use_venv: bool = True
    ):
        self.model_name_or_path = model_name_or_path

        if task not in _TASK_TO_EXAMPLE_SCRIPT:
            supported_tasks = ", ".join(_TASK_TO_EXAMPLE_SCRIPT.keys())
            raise ValueError(f"Unknown task named {task}, supported tasks are: {supported_tasks}")
        self.task = task

        self.example_dir = example_dir
        if example_dir is None:
            example_dir = Path(__file__).parent.parent.parent.parent / "examples"
            if not example_dir.exists():
                logger.info(
                    f"Could not find the example script for the task {task} locally. Please provide the path manually "
                    "or install `optimum-neuron` from sources. Otherwise the example will be downloaded from the "
                    "GitHub repo."
                )
            else:
                self.example_dir = example_dir

        self.use_venv = use_venv
        self.venv_dir = TemporaryDirectory()
        self.python_name = "python"
        self.pip_name = "pip"
        self.torchrun_name = "torchrun"
        if use_venv:
            self.create_venv(self.venv_dir.name)

        self._installed_requirements = False

    def create_venv(self, venv_path: str):
        """
        Creates the virtual environment for the example.
        """
        cmd_line = f"python -m venv {venv_path} --system-site-packages".split()
        p = subprocess.Popen(cmd_line, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            stdout = stdout.decode("utf-8")
            stderr = stderr.decode("utf-8")
            raise RuntimeError(
                f"Could not create the virtual environment to run the example. Full error:\nStandard output:\n{stdout}\n"
                f"Standard error:\n{stderr}"
            )

        # Install pip
        cmd_line = f"{venv_path}/bin/python -m ensurepip --upgrade".split()
        p = subprocess.Popen(cmd_line, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            stdout = stdout.decode("utf-8")
            stderr = stderr.decode("utf-8")
            raise RuntimeError(
                f"Could not create the virtual environment to run the example. Full error:\nStandard output:\n{stdout}\n"
                f"Standard error:\n{stderr}"
            )

        self.python_name = f"{venv_path}/bin/python"
        self.pip_name = f"{venv_path}/bin/pip"
        self.torchrun_name = f"{venv_path}/bin/torchrun"

    def maybe_remove_venv(self):
        """
        Removes the virtual environment for the example if it exists.
        """
        self.venv_dir.cleanup()

    def install_requirements(self, requirements_filename: Union[str, Path]):
        """
        Installs the necessary requirements to run the example if the provided file exists, otherwise does nothing.
        """
        if self._installed_requirements:
            return
        if self.use_venv:
            # Update pip
            cmd_line = f"{self.pip_name} install --upgrade pip".split()
            p = subprocess.Popen(cmd_line)
            returncode = p.wait()
            assert returncode == 0

            # Set pip repository pointing to the Neuron repository
            cmd_line = (
                f"{self.pip_name} config set global.extra-index-url https://pip.repos.neuron.amazonaws.com".split()
            )
            p = subprocess.Popen(cmd_line)
            returncode = p.wait()
            assert returncode == 0

            # Install wget, awscli, Neuron Compiler and Neuron Framework
            cmd_line = f"{self.pip_name} freeze".split()
            p = subprocess.Popen(cmd_line, stdout=PIPE)
            cmd_line = "grep torch-neuronx".split()
            p = subprocess.Popen(cmd_line, stdin=p.stdout, stdout=PIPE)
            stdout, _ = p.communicate()
            if not stdout:
                cmd_line = f"{self.pip_name} install wget awscli neuronx-cc==2.* torch-neuronx torchvision".split()
                p = subprocess.Popen(cmd_line)
                returncode = p.wait()
                assert returncode == 0

        # Install requirements
        if isinstance(requirements_filename, str):
            requirements_filename = Path(requirements_filename)

        if requirements_filename.exists():
            cmd_line = f"{self.pip_name} install -r {requirements_filename.as_posix()}".split()
            p = subprocess.Popen(cmd_line)
            returncode = p.wait()
            assert returncode == 0
            self._installed_requirements = True

        if self.use_venv or requirements_filename.exists():
            # TODO: remove that as soon as possible.
            cmd_line = f"{self.pip_name} install numpy==1.21.6".split()
            p = subprocess.Popen(cmd_line)
            returncode = p.wait()
            assert returncode == 0

    def check_user_logged_in_and_cache_repo_is_set(self):
        token = HfFolder.get_token()
        if not token:
            raise RuntimeError(
                "You need to log in the Hugging Face Hub otherwise you will not be able to push anything. "
                "Please run the following command: huggingface-cli login"
            )
        saved_custom_cache_repo = load_custom_cache_repo_name_from_hf_home()
        custom_cache_repo = os.environ.get("CUSTOM_CACHE_REPO", None)
        if saved_custom_cache_repo is None and custom_cache_repo is None:
            logger.warning(
                "No custom Neuron cache repo set which means that the official Neuron cache repo will be used. If "
                "you are not a member of the Optimum Neuron Team, this means that you will not be able to push to the "
                "Hub. Follow the instructions here to set you custom Neuron cache: "
                "https://huggingface.co/docs/optimum-neuron/guides/cache_system#how-to-use-a-private-trainium-model-cache"
            )

        main_repo = get_hf_hub_cache_repos()[0]
        has_write_access = has_write_access_to_repo(main_repo)
        if not has_write_access:
            raise RuntimeError(
                f"You do not have write access to {main_repo}. Please log in and/or use a custom Tranium cache repo."
            )

    def run(
        self,
        num_cores: int,
        precision: Union[str, Precision],
        train_batch_size: int,
        sequence_length: Optional[Union[int, Tuple[int, int], List[int]]] = None,
        do_eval: bool = False,
        eval_batch_size: Optional[int] = None,
        gradient_accumulation_steps: int = 1,
        num_epochs: int = 1,
        max_steps: Optional[int] = None,
        logging_steps: int = 1,
        save_steps: int = -1,
        learning_rate: float = 1e-4,
    ) -> Tuple[int, str, str]:
        if num_cores <= 0 or num_cores > 32:
            raise ValueError("The number of Neuron cores to use must be between 1 and 32.")
        if isinstance(precision, str) and not isinstance(precision, Precision):
            precision = Precision(precision)
        if sequence_length is None and self.task != "image-classification":
            raise ValueError(f"You must provide sequence_length for task {self.task}.")

        self.check_user_logged_in_and_cache_repo_is_set()

        tmpdir = TemporaryDirectory()

        if self.example_dir is None:
            script_path = download_example_script_from_github(self.task, Path(tmpdir.name))
        else:
            script_name = _TASK_TO_EXAMPLE_SCRIPT[self.task]
            candidates = Path(self.example_dir).glob(f"*/{script_name}.py")
            candidates = list(candidates)
            if len(candidates) == 0:
                raise RuntimeError(f"Could not find {script_name}.py in examples located in {self.example_dir}")
            elif len(candidates) > 1:
                raise RuntimeError(f"Found more than {script_name}.py in examples located in {self.example_dir}")
            else:
                script_path = candidates[0]

        # Installing requirements if needed.
        self.install_requirements(script_path.parent / "requirements.txt")

        cmd = []

        cmd.append(self.python_name if num_cores == 1 else f"{self.torchrun_name} --nproc_per_node {num_cores}")
        cmd.append(script_path.as_posix())
        cmd.append(f"--model_name_or_path {self.model_name_or_path}")

        # Training steps and batch sizes.
        cmd.append(f"--num_train_epochs {num_epochs}")
        if max_steps is not None:
            cmd.append(f"--max_steps {max_steps}")
        cmd.append("--do_train")
        if do_eval:
            cmd.append("--do_eval")
        cmd.append(f"--learning_rate {learning_rate}")
        cmd.append(f"--per_device_train_batch_size {train_batch_size}")
        if do_eval:
            if eval_batch_size is None:
                raise ValueError("eval_batch_size must be specified if do_eval=True")
            else:
                cmd.append(f"--per_device_eval_batch_size {eval_batch_size}")
        cmd.append(f"--gradient_accumulation_steps {gradient_accumulation_steps}")

        # Logging and saving
        cmd.append("--report_to none")
        cmd.append(f"--logging_steps {logging_steps}")
        cmd.append("--save_strategy steps")
        cmd.append(f"--save_steps {save_steps}")
        cmd.append("--save_total_limit 1")

        if precision is Precision.bf16:
            cmd.append("--bf16")

        # Dataset
        arguments = self._TASK_TO_COMMAND_ARGUMENTS[self.task]
        for name, value in arguments.items():
            if name == "set_max_length":
                if isinstance(sequence_length, (tuple, list)):
                    raise ValueError(f"Only one sequence length needs to be specified for {self.task}.")
                cmd.append(f"--max_seq_length {sequence_length}")
            elif name in ["set_max_source_length", "set_max_target_length"]:
                if isinstance(sequence_length, int):
                    raise ValueError(
                        f"A sequence length for the encoder and for the decoder need to be specified for {self.task}, "
                        "but only one was specified here."
                    )
                if name == "set_max_source_length":
                    cmd.append(f"--max_source_length {sequence_length[0]}")
                else:
                    cmd.append(f"--max_target_length {sequence_length[1]}")
            elif name == "extra_command_line_arguments":
                for argument in value:
                    cmd.append(argument)
            else:
                cmd.append(f"--{name} {value}")

        def split_args_and_value_in_command(cmd: List[str]) -> List[str]:
            pattern = re.compile(r"([\"\'].+?[\"\'])|\s")
            return [x for y in cmd for x in re.split(pattern, y) if x]

        with TemporaryDirectory() as tmpdirname:
            cmd.append(f"--output_dir {tmpdirname}")

            cmd = split_args_and_value_in_command(cmd)

            print(f"RUNNING COMMAND:\n{' '.join(cmd)}")

            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = proc.communicate()
            stdout = stdout.decode("utf-8")
            stderr = stderr.decode("utf-8")

        tmpdir.cleanup()

        return proc.returncode, stdout, stderr
