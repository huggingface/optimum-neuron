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
"""Tools that fills the neuron cache with common models for the supported tasks."""

import time
from argparse import ArgumentParser
from pathlib import Path
from typing import TYPE_CHECKING, List, Dict

# TODO: find a cleaner solution to this.
import sys
path_tests = Path(__file__).parent.parent / "tests"
sys.path.insert(0, str(path_tests))

from test_examples import (
        TextClassificationExampleTester,
        TokenClassificationExampleTester,
        MultipleChoiceExampleTester,
        QuestionAnsweringExampleTester,
        SummarizationExampleTester,
        TranslationExampleTester,
        ImageClassificationExampleTester,
        ExampleTestMeta,
)

if TYPE_CHECKING:
    from test_examples import ExampleTesterBase

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
    "albert": {
        "albert-base-v2": {
            "default": {"batch_size": 16, "sequence_length": 128}, 
            "token-classification":  {"batch_size": 2, "sequence_length": 512},
            "multiple-choice":  {"batch_size": 2, "sequence_length": 512},
        },
        "albert-large-v2": {
            "default": {"batch_size": 16, "sequence_length": 128}, 
            "token-classification":  {"batch_size": 2, "sequence_length": 512},
            "multiple-choice":  {"batch_size": 2, "sequence_length": 512},
        },
    },
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
            "token-classification":  {"batch_size": 2, "sequence_length": 512},
            "multiple-choice":  {"batch_size": 2, "sequence_length": 512},
        },
        "bert-large-uncased": {
            "default": {"batch_size": 16, "sequence_length": 128}, 
            "token-classification":  {"batch_size": 2, "sequence_length": 512},
            "multiple-choice":  {"batch_size": 2, "sequence_length": 512},
        },
    },
    "camembert": {
        "camembert-base": {
            "default": {"batch_size": 16, "sequence_length": 128}, 
            "token-classification":  {"batch_size": 2, "sequence_length": 512},
            "multiple-choice":  {"batch_size": 2, "sequence_length": 512},
        },
        "camembert/camembert-large": {
            "default": {"batch_size": 16, "sequence_length": 128}, 
            "token-classification":  {"batch_size": 2, "sequence_length": 512},
            "multiple-choice":  {"batch_size": 2, "sequence_length": 512},
        },
    },
    "distilbert": {
        "distilbert-base-uncased": {
            "default": {"batch_size": 16, "sequence_length": 128}, 
            "token-classification":  {"batch_size": 2, "sequence_length": 512},
            "multiple-choice":  {"batch_size": 2, "sequence_length": 512},
        },
    },
    "electra": {
        "google/electra-small-discriminator": {
            "default": {"batch_size": 16, "sequence_length": 128}, 
            "token-classification":  {"batch_size": 2, "sequence_length": 512},
            "multiple-choice":  {"batch_size": 2, "sequence_length": 512},
        },
        "google/electra-base-discriminator": {
            "default": {"batch_size": 16, "sequence_length": 128}, 
            "token-classification":  {"batch_size": 2, "sequence_length": 512},
            "multiple-choice":  {"batch_size": 2, "sequence_length": 512},
        },
        "google/electra-large-discriminator": {
            "default": {"batch_size": 16, "sequence_length": 128}, 
            "token-classification":  {"batch_size": 2, "sequence_length": 512},
            "multiple-choice":  {"batch_size": 2, "sequence_length": 512},
        },
    },
    "gpt2": {
        "gpt2": {
            "default": {"batch_size": 16, "sequence_length": 128}, 
        },
        # "gpt2-large": {
        #     "default": {"batch_size": 16, "sequence_length": 128}, 
        # },
    },
    "gpt-neo": {
        "EleutherAI/gpt-neo-125M": {
            "default": {"batch_size": 16, "sequence_length": 128}, 
        },
    },
    "marian": {
        "Helsinki-NLP/opus-mt-en-es": {
            "translation": {"batch_size": 8, "source_sequence_length": 512, "target_sequence_length": 512},
        },
        "Helsinki-NLP/opus-mt-en-hi": {
            "translation": {"batch_size": 8, "source_sequence_length": 512, "target_sequence_length": 512},
        },
        "Helsinki-NLP/opus-mt-es-en": {
            "translation": {"batch_size": 8, "source_sequence_length": 512, "target_sequence_length": 512},
        },
    },
    "roberta": {
        "roberta-base": {
            "default": {"batch_size": 16, "sequence_length": 128}, 
            "token-classification":  {"batch_size": 2, "sequence_length": 512},
            "multiple-choice":  {"batch_size": 2, "sequence_length": 512},
        },
        "roberta-large": {
            "default": {"batch_size": 16, "sequence_length": 128}, 
            "token-classification":  {"batch_size": 2, "sequence_length": 512},
            "multiple-choice":  {"batch_size": 2, "sequence_length": 512},
        },
    },
    "t5": {
        "t5-small": {
            "translation": {"batch_size": 8, "source_sequence_length": 512, "target_sequence_length": 512},
            "summarization": {"batch_size": 8, "source_sequence_length": 200, "target_sequence_length": 1024},
        },
        "t5-base": {
            "translation": {"batch_size": 8, "source_sequence_length": 512, "target_sequence_length": 512},
            "summarization": {"batch_size": 8, "source_sequence_length": 200, "target_sequence_length": 1024},
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
            "token-classification":  {"batch_size": 2, "sequence_length": 512},
            "multiple-choice":  {"batch_size": 2, "sequence_length": 512},
        },
        "xlm-roberta-large": {
            "default": {"batch_size": 16, "sequence_length": 128}, 
            "token-classification":  {"batch_size": 2, "sequence_length": 512},
            "multiple-choice":  {"batch_size": 2, "sequence_length": 512},
        },
    },

}


def get_testers_for_model_type(model_type: str) -> List["ExampleTesterBase"]:
    testers = []
    for task, cls in TESTER_CLASSES.items():
        for attr_name in dir(cls):
            if attr_name.startswith("test") and model_type in attr_name:
                testers.append((task, cls(), attr_name))
    return testers


def remove_extra_command_line_argument(command_prefix: str, extra_command_line_arguments: List[str]):
    argument_idx = None
    for idx, cmd_line_argument in enumerate(extra_command_line_arguments):
        if cmd_line_argument.startswith(command_prefix):
            argument_idx = idx
    if argument_idx is not None:
        extra_command_line_arguments.pop(argument_idx)

def run_auto_fill_cache_for_model_name(model_type: str, model_name: str, shape_values_for_task: Dict[str, int], tester: "ExampleTesterBase", method_name: str, neuron_cache: str):
    
    batch_size = shape_values_for_task.get("batch_size")
    sequence_length = shape_values_for_task.get("sequence_length")
    source_sequence_length = shape_values_for_task.get("source_sequence_length")
    target_sequence_length = shape_values_for_task.get("target_sequence_length")
    
    extra_command_line_arguments = tester.EXTRA_COMMAND_LINE_ARGUMENTS
    if extra_command_line_arguments is None:
        extra_command_line_arguments = []
    
    if batch_size is not None:
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
    tester.ONLY_PRECOMPILATION = True
    
    setattr(tester, method_name, ExampleTestMeta._create_test(model_type, model_name))
    getattr(tester, method_name)()



def parse_args():
    parser = ArgumentParser(description="Tool that runs precompilation to fill the Neuron Cache.")
    parser.add_argument("--cache-path", type=Path, default="neuron-cache", help="The directory in which all the precompiled neff files will be stored.")
    parser.add_argument("--models", type=str, default="all", nargs="+",help="The models to precompile.")
    parser.add_argument("--tasks", type=str, default="all", nargs="+",help="The tasks to precompile.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.models == "all":
        models = list(ARCHITECTURES_TO_COMMON_PRETRAINED_WEIGHTS.keys())
    else:
        models = args.models
    
    for model_type in models:
        testers = get_testers_for_model_type(model_type)
        for task, tester, method_name in testers:
            if args.tasks != "all" and task not in args.tasks:
                continue
            for model_name, shape_values in ARCHITECTURES_TO_COMMON_PRETRAINED_WEIGHTS[model_type].items():
                print(f"Running precompilation for {model_name} on {task}...")
                start = time.time()
                shape_values_for_task = shape_values.get(task, shape_values["default"])
                run_auto_fill_cache_for_model_name(model_type, model_name, shape_values_for_task, tester, method_name, args.cache_path)
                end = time.time()
                print(f"Done! Duration: {end - start:.3f}.")

