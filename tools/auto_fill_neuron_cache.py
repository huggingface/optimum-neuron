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
"""Tools that fills the neuron cache with common models for the supported tasks."""

import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from optimum.neuron.utils.cache_utils import get_num_neuron_cores


# Important to do it before importing the tests.
os.environ["RUN_SLOW"] = "1"

path_tests = Path(__file__).parent.parent / "tests"
sys.path.insert(0, str(path_tests))

if TYPE_CHECKING:
    from test_examples import ExampleTesterBase  # noqa: E402

from test_examples import (  # noqa: E402
    ExampleTestMeta,
    ImageClassificationExampleTester,
    MultipleChoiceExampleTester,
    QuestionAnsweringExampleTester,
    SummarizationExampleTester,
    TextClassificationExampleTester,
    TokenClassificationExampleTester,
    TranslationExampleTester,
)


TESTER_CLASSES = {
    "sequence-classification": TextClassificationExampleTester,
    "token-classification": TokenClassificationExampleTester,
    "multiple-choice": MultipleChoiceExampleTester,
    "question-answering": QuestionAnsweringExampleTester,
    "summarization": SummarizationExampleTester,
    "translation": TranslationExampleTester,
    "image-classification": ImageClassificationExampleTester,
}


ARCHITECTURES_TO_COMMON_PRETRAINED_WEIGHTS = {
    "bart": {
        "facebook/bart-base": {
            "translation": {"batch_size": 8, "source_sequence_length": 512, "target_sequence_length": 512},
            "summarization": {"batch_size": 8, "source_sequence_length": 200, "target_sequence_length": 1024},
        },
        "facebook/bart-large": {
            "translation": {"batch_size": 8, "source_sequence_length": 512, "target_sequence_length": 512},
            "summarization": {"batch_size": 8, "source_sequence_length": 200, "target_sequence_length": 1024},
        },
    },
    "bert": {
        "bert-base-uncased": {
            "default": {"batch_size": 16, "sequence_length": 128},
            "token-classification": {"batch_size": 8, "sequence_length": 512},
            "multiple-choice": {"batch_size": 8, "sequence_length": 512},
        },
        "bert-large-uncased": {
            "default": {"batch_size": 8, "sequence_length": 128},
            "token-classification": {"batch_size": 4, "sequence_length": 512},
            "multiple-choice": {"batch_size": 4, "sequence_length": 512},
        },
    },
    "camembert": {
        "camembert-base": {
            "default": {"batch_size": 16, "sequence_length": 128},
            "token-classification": {"batch_size": 8, "sequence_length": 512},
            "multiple-choice": {"batch_size": 8, "sequence_length": 512},
        },
        "camembert/camembert-large": {
            "default": {"batch_size": 8, "sequence_length": 128},
            "token-classification": {"batch_size": 4, "sequence_length": 512},
            "multiple-choice": {"batch_size": 4, "sequence_length": 512},
        },
    },
    "distilbert": {
        "distilbert-base-uncased": {
            "default": {"batch_size": 16, "sequence_length": 128},
            "token-classification": {"batch_size": 8, "sequence_length": 512},
            "multiple-choice": {"batch_size": 8, "sequence_length": 512},
        },
    },
    "electra": {
        "google/electra-small-discriminator": {
            "default": {"batch_size": 16, "sequence_length": 128},
            "token-classification": {"batch_size": 8, "sequence_length": 512},
            "multiple-choice": {"batch_size": 8, "sequence_length": 512},
        },
        "google/electra-base-discriminator": {
            "default": {"batch_size": 16, "sequence_length": 128},
            "token-classification": {"batch_size": 8, "sequence_length": 512},
            "multiple-choice": {"batch_size": 8, "sequence_length": 512},
        },
        "google/electra-large-discriminator": {
            "default": {"batch_size": 16, "sequence_length": 128},
            "token-classification": {"batch_size": 4, "sequence_length": 512},
            "multiple-choice": {"batch_size": 4, "sequence_length": 512},
        },
    },
    "gpt2": {
        "gpt2": {
            "default": {"batch_size": 16, "sequence_length": 128},
        },
    },
    "marian": {
        "Helsinki-NLP/opus-mt-en-es": {
            "translation": {"batch_size": 4, "source_sequence_length": 512, "target_sequence_length": 512},
        },
        "Helsinki-NLP/opus-mt-en-hi": {
            "translation": {"batch_size": 4, "source_sequence_length": 512, "target_sequence_length": 512},
        },
        "Helsinki-NLP/opus-mt-es-en": {
            "translation": {"batch_size": 4, "source_sequence_length": 512, "target_sequence_length": 512},
        },
    },
    "roberta": {
        "roberta-base": {
            "default": {"batch_size": 16, "sequence_length": 128},
            "token-classification": {"batch_size": 8, "sequence_length": 512},
            "multiple-choice": {"batch_size": 8, "sequence_length": 512},
        },
        "roberta-large": {
            "default": {"batch_size": 8, "sequence_length": 128},
            "token-classification": {"batch_size": 4, "sequence_length": 512},
            "multiple-choice": {"batch_size": 4, "sequence_length": 512},
        },
    },
    "t5": {
        "t5-small": {
            "translation": {"batch_size": 8, "source_sequence_length": 512, "target_sequence_length": 512},
            "summarization": {"batch_size": 8, "source_sequence_length": 200, "target_sequence_length": 512},
        },
        "t5-base": {
            "translation": {"batch_size": 4, "source_sequence_length": 512, "target_sequence_length": 512},
            "summarization": {"batch_size": 4, "source_sequence_length": 200, "target_sequence_length": 512},
        },
    },
    "vit": {
        "google/vit-base-patch16-224": {"default": {"batch_size": 16}},
        "google/vit-base-patch16-224-in21k": {"default": {"batch_size": 16}},
        "google/vit-large-patch16-224-in21k": {"default": {"batch_size": 8}},
    },
    "xlm-roberta": {
        "xlm-roberta-base": {
            "default": {"batch_size": 16, "sequence_length": 128},
            "token-classification": {"batch_size": 8, "sequence_length": 512},
            "multiple-choice": {"batch_size": 8, "sequence_length": 512},
        },
        "xlm-roberta-large": {
            "default": {"batch_size": 16, "sequence_length": 128},
            "token-classification": {"batch_size": 8, "sequence_length": 512},
            "multiple-choice": {"batch_size": 8, "sequence_length": 512},
        },
    },
}


def get_testers_for_model_type(model_type: str) -> list["ExampleTesterBase"]:
    testers = []
    for task, cls in TESTER_CLASSES.items():
        for attr_name in dir(cls):
            if attr_name.startswith("test") and model_type in attr_name:
                testers.append((task, cls(), attr_name))
    return testers


def remove_extra_command_line_argument(command_prefix: str, extra_command_line_arguments: list[str | dict[str, str]]):
    argument_idx = None
    for idx, cmd_line_argument in enumerate(extra_command_line_arguments):
        if isinstance(cmd_line_argument, dict):
            starts_with_command_prefix = any(v.startswith(command_prefix) for v in cmd_line_argument.values())
        else:
            starts_with_command_prefix = cmd_line_argument.startswith(command_prefix)
        if starts_with_command_prefix:
            argument_idx = idx
    if argument_idx is not None:
        extra_command_line_arguments.pop(argument_idx)


def run_auto_fill_cache_for_model_name(
    model_type: str,
    model_name: str,
    shape_values_for_task: dict[str, int],
    tester: "ExampleTesterBase",
    method_name: str,
    neuron_cache: str,
    num_neuron_cores: int,
    bf16: bool,
):
    batch_size = shape_values_for_task.get("batch_size")
    sequence_length = shape_values_for_task.get("sequence_length")
    source_sequence_length = shape_values_for_task.get("source_sequence_length")
    target_sequence_length = shape_values_for_task.get("target_sequence_length")

    extra_command_line_arguments = tester.EXTRA_COMMAND_LINE_ARGUMENTS
    if extra_command_line_arguments is None:
        extra_command_line_arguments = []

    if batch_size is not None:
        if not bf16:
            batch_size = min(int(batch_size / 2), 1)
        tester.TRAIN_BATCH_SIZE = batch_size
        tester.EVAL_BATCH_SIZE = batch_size

    if sequence_length is not None:
        remove_extra_command_line_argument("--max_seq_length", extra_command_line_arguments)
        extra_command_line_arguments.append(f"--max_seq_length {sequence_length}")

    if source_sequence_length is not None:
        remove_extra_command_line_argument("--max_source_length", extra_command_line_arguments)
        extra_command_line_arguments.append(f"--max_source_length {source_sequence_length}")

    if target_sequence_length is not None:
        remove_extra_command_line_argument("--max_target_length", extra_command_line_arguments)
        extra_command_line_arguments.append(f"--max_target_length {target_sequence_length}")

    tester.EXTRA_COMMAND_LINE_ARGUMENTS = extra_command_line_arguments
    tester.NEURON_CACHE = neuron_cache
    tester.DO_PRECOMPILATION = False
    tester.MAX_STEPS = 200
    tester.SAVE_STEPS = 100
    tester.NPROC_PER_NODE = num_neuron_cores
    tester.BF16 = bf16
    tester.EVAL_IS_SUPPORTED = False
    tester.GRADIENT_ACCUMULATION_STEPS = 1

    setattr(tester, method_name, ExampleTestMeta._create_test(model_type, model_name))
    getattr(tester, method_name)(tester)


def parse_args():
    parser = ArgumentParser(description="Tool that runs precompilation to fill the Neuron Cache.")
    parser.add_argument(
        "--cache-path",
        type=Path,
        default="neuron-cache",
        help="The directory in which all the precompiled neff files will be stored.",
    )
    parser.add_argument("--models", type=str, default="all", nargs="+", help="The models to precompile.")
    parser.add_argument("--tasks", type=str, default="all", nargs="+", help="The tasks to precompile.")
    return parser.parse_args()


def open_and_append_to_file(filename: str, new_line_to_append: str):
    with open(filename, "w+") as fp:
        content = fp.read()
        new_content = f"{content}\n{new_line_to_append}"
        fp.write(new_content)


def main():
    args = parse_args()

    now = datetime.now()
    now_str = now.strftime("%d%m%Y_%H_%M")
    success_filename = f"success_{now_str}.txt"
    failure_filename = f"failure_{now_str}.txt"

    if args.models == "all":
        models = list(ARCHITECTURES_TO_COMMON_PRETRAINED_WEIGHTS.keys())
    else:
        models = args.models

    # Test examples are slow, this allows us to run them.
    os.environ["RUN_SLOW"] = "1"

    total_num_cores = get_num_neuron_cores()
    num_cores_configurations = [1, 2, 8, total_num_cores]
    num_cores_configurations = sorted(set(num_cores_configurations))

    for model_type in models:
        testers = get_testers_for_model_type(model_type)
        for task, tester, method_name in testers:
            if args.tasks != "all" and task not in args.tasks:
                continue
            for model_name, shape_values in ARCHITECTURES_TO_COMMON_PRETRAINED_WEIGHTS[model_type].items():
                print(f"Running precompilation for {model_name} on {task}...")
                for num_cores in num_cores_configurations:
                    print(f"For {num_cores} neuron cores")
                    if num_cores == 1:
                        tester.MULTI_PROC = "false"
                    else:
                        tester.MULTI_PROC = "true"
                    start = time.time()
                    shape_values_for_task = shape_values.get(task)
                    if shape_values_for_task is None:
                        shape_values_for_task = shape_values["default"]
                    example = ["*" * 20]
                    example.append(f"Model:\t{model_name}")
                    example.append("Shapes:")
                    for name, value in shape_values_for_task.items():
                        example.append(f"\t{name} = {value}")
                    example.append(f"Num cores:\t{num_cores}")
                    example_str = ""
                    try:
                        start = time.time()
                        example.append("Precision:\tBF16")
                        example.append("*" * 20)
                        example_str = "\n".join(example)
                        run_auto_fill_cache_for_model_name(
                            model_type,
                            model_name,
                            shape_values_for_task,
                            tester,
                            method_name,
                            args.cache_path,
                            num_cores,
                            True,
                        )
                    except Exception as e:
                        print(e)
                        open_and_append_to_file(failure_filename, example_str)
                    else:
                        open_and_append_to_file(success_filename, example_str)

                    end = time.time()
                    print(f"Done! Duration: {end - start:.3f}.")
                    try:
                        start = time.time()
                        example.append("Precision:\tFull-precision")
                        example.append("*" * 20)
                        example_str = "\n".join(example)
                        run_auto_fill_cache_for_model_name(
                            model_type,
                            model_name,
                            shape_values_for_task,
                            tester,
                            method_name,
                            args.cache_path,
                            num_cores,
                            False,
                        )
                    except Exception:
                        open_and_append_to_file(failure_filename, example_str)
                    else:
                        open_and_append_to_file(success_filename, example_str)
                    end = time.time()
                    print(f"Done! Duration: {end - start:.3f}.")

    os.environ["RUN_SLOW"] = "0"


if __name__ == "__main__":
    main()
