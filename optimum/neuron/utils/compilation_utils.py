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

import subprocess
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Tuple, Union

import requests

from ...utils import logging


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
            ],
        },
        "multiple-choice": {
            "set_max_length": True,
            "extra_command_line_arguments": [
                "--pad_to_max_length",
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
                "--pad_to_max_length",
            ],
        },
        "image-classification": {
            "dataset_name": "cifar10",
            "extra_command_line_arguments": [
                "--remove_unused_columns",
                "--dataloader_drop_last",
                "--ignore_mismatched_sizes",
            ],
        },
    }

    def __init__(self, model_name_or_path: str, task: str, example_dir: Optional[Union[str, Path]] = None):
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

    def run(
        self,
        num_cores: int,
        precision: Union[str, Precision],
        train_batch_size: int,
        sequence_length: Union[int, Tuple[int, int], List[int]],
        do_eval: bool = False,
        eval_batch_size: Optional[int] = None,
        gradient_accumulation_steps: int = 1,
        num_epochs: int = 1,
        max_steps: Optional[int] = None,
        logging_steps: int = 1,
        save_steps: int = -1,
        learning_rate: float = 1e-4,
    ):
        if num_cores <= 0 or num_cores > 32:
            raise ValueError("The number of Neuron cores to use must be between 1 and 32.")
        if isinstance(precision, str) and not isinstance(precision, Precision):
            precision = Precision(precision)

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

        cmd = []

        cmd.append("python" if num_cores == 1 else f"torchrun --nproc_per_node {num_cores}")
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

        with TemporaryDirectory() as tmpdirname:
            cmd.append(f"--output_dir {tmpdirname}")

            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = proc.communicate()
            stdout = stdout.decode("utf-8")
            stderr = stderr.decode("utf-8")

        tmpdir.cleanup()

        return stdout, stderr
