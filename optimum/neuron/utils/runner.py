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

import codecs
import os
import re
import subprocess
import sys
from enum import Enum
from pathlib import Path
from subprocess import PIPE
from tempfile import TemporaryDirectory
from typing import Any

import requests
from huggingface_hub import (
    HfApi,
    get_token,
    snapshot_download,
)
from transformers import AutoConfig

from ...utils import logging
from .cache_utils import get_hf_hub_cache_repos, has_write_access_to_repo, load_custom_cache_repo_name_from_hf_home


logger = logging.get_logger()

_GH_REPO_RAW_URL = "https://raw.githubusercontent.com/huggingface/optimum-neuron"
_GH_REPO_URL = "https://github.com/huggingface/optimum-neuron"
_GH_REPO_EXAMPLE_FOLDERS = [
    # "audio-classification",
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
    # "audio-classification": "run_audio_classification",
    "speech-recognition": "run_speech_recognition_ctc",
}


def list_filenames_in_github_repo_directory(
    github_repo_directory_url: str, only_files: bool = False, only_directories: bool = False
) -> list[str]:
    """
    Lists the content of a repository on GitHub.
    """
    if only_files and only_directories:
        raise ValueError("Either `only_files` or `only_directories` can be set to True.")

    response = requests.get(github_repo_directory_url)

    if response.status_code != 200:
        raise ValueError(f"Could not fetch the content of the page: {github_repo_directory_url}.")

    # Here we use regex instead of beautiful soup to not rely on yet another library.
    table_regex = r"\<table aria-labelledby=\"folders-and-files\".*\<\/table\>"
    filename_column_regex = r"\<div class=\"react-directory-filename-cell\".*?\<\/div>"
    if only_files:
        filename_regex = r"\<a .* aria-label=\"([\w\.]+), \(File\)\""
    elif only_directories:
        filename_regex = r"\<a .* aria-label=\"([\w\.]+), \(Directory\)\""
    else:
        filename_regex = r"\<a .* aria-label=\"([\w\.]+)"

    filenames = []

    table_match = re.search(table_regex, response.text)
    if table_match is not None:
        table_content = response.text[table_match.start(0) : table_match.end(0)]
        for column in re.finditer(filename_column_regex, table_content):
            match = re.search(filename_regex, column.group(0))
            if match:
                filenames.append(match.group(1))

    return list(set(filenames))


def download_example_script_from_github(task_name: str, target_directory: Path, revision: str = "main") -> Path:
    was_saved = False
    script_name = f"{_TASK_TO_EXAMPLE_SCRIPT[task_name]}.py"
    example_script_path = target_directory
    for folder in _GH_REPO_EXAMPLE_FOLDERS:
        raw_url_folder = f"{_GH_REPO_RAW_URL}/refs/heads/{revision}/examples/{folder}"
        url_folder = f"{_GH_REPO_URL}/tree/{revision}/examples/{folder}"
        filenames_for_example = list_filenames_in_github_repo_directory(url_folder, only_files=True)
        if script_name not in filenames_for_example:
            continue
        for filename in filenames_for_example:
            r = requests.get(f"{raw_url_folder}/{filename}")
            if r.status_code != 200:
                continue
            local_path = target_directory / filename
            with open(local_path, "w") as fp:
                fp.write(r.text)
            if filename == script_name:
                was_saved = True
                example_script_path = local_path
        if was_saved:
            break
    if not was_saved:
        raise FileNotFoundError(f"Could not find an example script for the task {task_name} on the GitHub repo")
    return example_script_path


class Precision(str, Enum):
    fp = "fp"
    bf16 = "bf16"


def run_command_with_realtime_output(cmd: list[str], **popen_kwargs) -> tuple[int, str]:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **popen_kwargs)
    stdout = []
    decoder = codecs.getincrementaldecoder("utf-8")()
    while True:
        output = proc.stdout.read(1)
        output = decoder.decode(output)
        if output == "" and proc.poll() is not None:
            break
        if output != "":
            stdout.append(output)
            print(output, end="")
    stdout = "".join(stdout)
    return proc.returncode, stdout


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
            "dataset_name": "bnsapa/cybersecurity-ner",
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
                "--prediction_loss_only",
                {"t5": "--source_prefix 'summarize: '"},
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
                {"t5": "--source_prefix 'Translate Romanian to English: '"},
            ],
        },
        "image-classification": {
            "dataset_name": "mnist",
            "extra_command_line_arguments": [
                "--remove_unused_columns false",
                "--ignore_mismatched_sizes",
            ],
        },
    }

    def __init__(
        self,
        model_name_or_path: str,
        task: str,
        example_dir: str | Path | None = None,
        config_overrides: dict[str, Any | None] = None,
        use_venv: bool = False,
        install_requirements: bool = True,
    ):
        self.model_name_or_path = model_name_or_path
        self.config_overrides = config_overrides

        if task not in _TASK_TO_EXAMPLE_SCRIPT:
            supported_tasks = ", ".join(_TASK_TO_EXAMPLE_SCRIPT.keys())
            raise ValueError(f"Unknown task named {task}, supported tasks are: {supported_tasks}")
        self.task = task

        self.example_dir = example_dir

        if use_venv:
            raise NotImplementedError("use_venv=True is not supported yet.")
        self.use_venv = use_venv
        self.should_install_requirements = install_requirements
        self.venv_dir = TemporaryDirectory()
        self.python_name = sys.executable
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

    def install_requirements(self, requirements_filename: str | Path):
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

            # set pip repository pointing to the Neuron repository
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

    def check_user_logged_in_and_cache_repo_is_set(self):
        token = get_token()
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
                f"You do not have write access to {main_repo}. Please log in and/or use a custom Neuron cache repo."
            )

    def download_model_repo_and_override_config(
        self, model_name_or_path: str, config_overrides: dict[str, Any], output_dir: str | Path
    ) -> str | Path:
        if not config_overrides:
            return model_name_or_path

        filenames = HfApi().list_repo_files(repo_id=model_name_or_path, token=get_token())
        safetensors_model_file_pattern = re.compile(r"\w+(-[0-9]*-of-[0-9]*)?\.safetensors")
        allow_patterns = ["*.json", "*.txt"]
        if any(re.match(safetensors_model_file_pattern, filename) for filename in filenames):
            # Not downloading PyTorch checkpoints if safetensors checkpoints are available.
            allow_patterns.append("*.safetensors")
        else:
            allow_patterns.append("*.bin")

        directory = Path(output_dir) / model_name_or_path.split("/")[-1]

        # local_dir_use_symlinks = "auto" will download big files (>= 5MB) in the cache and create symlinks in
        # local_dir, while creating copies in local_dir for small files.
        # Here the goal is to edit the config of the model so this solution seems optimal.
        snapshot_download(
            model_name_or_path, allow_patterns=allow_patterns, local_dir=directory, local_dir_use_symlinks="auto"
        )

        config = AutoConfig.from_pretrained(directory)

        for name, value in config_overrides.items():
            type_of_attribute = type(getattr(config, name))
            if type(value) is not type_of_attribute:
                value = type_of_attribute(value)
            setattr(config, name, value)

        config.save_pretrained(directory)

        return directory

    def run(
        self,
        num_cores: int,
        precision: str | Precision,
        train_batch_size: int,
        sequence_length: int | tuple[int, int | None, list[int]] = None,
        do_eval: bool = False,
        eval_batch_size: int | None = None,
        gradient_accumulation_steps: int = 1,
        num_epochs: int = 1,
        max_steps: int | None = None,
        max_eval_samples: int | None = None,
        logging_steps: int = 1,
        save_steps: int = -1,
        save_total_limit: int = -1,
        learning_rate: float = 1e-4,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        disable_embedding_parallelization: bool = False,
        zero_1: bool = False,
        output_dir: Path | str | None = None,
        do_precompilation: bool = False,
        print_outputs: bool = False,
        resume_from_checkpoint: str | Path | None = None,
        _disable_is_private_model_repo_check: bool = False,
    ) -> tuple[int, str]:
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
        if self.should_install_requirements:
            self.install_requirements(script_path.parent / "requirements.txt")

        def compute_max_train_samples(
            max_steps: int,
            num_cores: int,
            tensor_parallel_size: int,
            pipeline_parallel_size: int,
            per_device_train_batch_size: int,
        ) -> int:
            number_of_cores_per_replicas = tensor_parallel_size * pipeline_parallel_size
            total_batch_size = (num_cores // number_of_cores_per_replicas) * per_device_train_batch_size
            total_num_samples = max_steps * total_batch_size
            # Adding 10% more examples just to make sure.
            return int(total_num_samples * 1.1)

        cmd = []

        cmd.append(self.python_name if num_cores == 1 else f"{self.torchrun_name} --nproc_per_node {num_cores}")
        cmd.append(script_path.as_posix())

        model_name_or_path = self.model_name_or_path
        if self.config_overrides is not None:
            model_name_or_path = self.download_model_repo_and_override_config(
                self.model_name_or_path, self.config_overrides, tmpdir.name
            )
        cmd.append(f"--model_name_or_path {model_name_or_path}")

        # Training steps and batch sizes.
        cmd.append(f"--num_train_epochs {num_epochs}")
        max_steps_idx = -1
        if max_steps is not None:
            cmd.append(f"--max_steps {max_steps}")
            max_steps_idx = len(cmd) - 1
            max_train_samples = compute_max_train_samples(
                max_steps, num_cores, tensor_parallel_size, pipeline_parallel_size, train_batch_size
            )
            cmd.append(f"--max_train_samples {max_train_samples}")

        cmd.append("--do_train")
        if do_eval:
            cmd.append("--do_eval")
            if max_eval_samples is not None:
                cmd.append(f"--max_eval_samples {max_eval_samples}")
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
        cmd.append(f"--save_total_limit {save_total_limit}")

        # Parallelism
        if tensor_parallel_size > 1:
            cmd.append(f"--tensor_parallel_size {tensor_parallel_size}")
        if pipeline_parallel_size > 1:
            cmd.append(f"--pipeline_parallel_size {pipeline_parallel_size}")
        if disable_embedding_parallelization:
            cmd.append("--disable_embedding_parallelization")
        if zero_1:
            cmd.append("--zero_1")

        if precision is Precision.bf16:
            cmd.append("--bf16")

        # Dataset
        arguments = self._TASK_TO_COMMAND_ARGUMENTS[self.task]
        model_type = AutoConfig.from_pretrained(model_name_or_path).model_type
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
                    if isinstance(argument, dict):
                        argument = argument.get(model_type, argument.get("default", None))
                    if argument is not None:
                        cmd.append(argument)
            else:
                cmd.append(f"--{name} {value}")

        def split_args_and_value_in_command(cmd: list[str]) -> list[str]:
            pattern = re.compile(r"([\"\'].+?[\"\'])|\s")
            return [x for y in cmd for x in re.split(pattern, y) if x]

        with TemporaryDirectory() as tmpdirname:
            if output_dir is None:
                cmd.append(f"--output_dir {tmpdirname}")
            else:
                cmd.append(f"--output_dir {output_dir}")

            if resume_from_checkpoint is not None:
                cmd.append(f"--resume_from_checkpoint {resume_from_checkpoint}")

            env = dict(os.environ)
            if _disable_is_private_model_repo_check:
                env["OPTIMUM_NEURON_DISABLE_IS_PRIVATE_REPO_CHECK"] = "true"

            if do_precompilation:
                # We need to update both the number of steps and the output directory specifically for the
                # precompilation step.
                with TemporaryDirectory() as precompilation_tmpdirname:
                    precompilation_cmd = list(cmd)
                    precompilation_cmd.pop(-1)  # Removing the --output_dir argument.
                    max_steps_cmd_str = "--max_steps 10"
                    max_train_samples = compute_max_train_samples(
                        10, num_cores, tensor_parallel_size, pipeline_parallel_size, train_batch_size
                    )
                    max_train_samples_cmd = f"--max_train_samples {max_train_samples}"
                    if max_steps_idx >= 0:
                        precompilation_cmd[max_steps_idx] = max_steps_cmd_str
                        precompilation_cmd[max_steps_idx + 1] = max_train_samples_cmd
                    else:
                        precompilation_cmd.append(max_steps_cmd_str)
                        precompilation_cmd.append(max_train_samples_cmd)

                    precompilation_cmd.append(f"--output_dir {precompilation_tmpdirname}")
                    precompilation_cmd = ["neuron_parallel_compile"] + precompilation_cmd

                    precompilation_cmd = split_args_and_value_in_command(precompilation_cmd)

                    print(f"RUNNING PRECOMPILATION COMMAND:\n{' '.join(precompilation_cmd)}")

                    if print_outputs:
                        returncode, stdout = run_command_with_realtime_output(precompilation_cmd)
                    else:
                        proc = subprocess.Popen(
                            precompilation_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
                        )
                        stdout, _ = proc.communicate()
                        stdout = stdout.decode("utf-8")
                        returncode = proc.returncode

            cmd = split_args_and_value_in_command(cmd)
            print(f"RUNNING COMMAND:\n{' '.join(cmd)}")

            if print_outputs:
                returncode, stdout = run_command_with_realtime_output(cmd)
            else:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
                stdout, _ = proc.communicate()
                stdout = stdout.decode("utf-8")
                returncode = proc.returncode

        tmpdir.cleanup()

        return returncode, stdout
