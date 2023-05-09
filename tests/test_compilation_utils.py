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
"""Tests for the compilation utilities."""

from unittest import TestCase
from parameterized import parameterized

from optimum.neuron.utils.testing_utils import is_trainium_test
from optimum.neuron.utils.compilation_utils import ExampleRunner

_TINY_BERT_MODEL_NAME = "hf-internal-testing/tiny-random-bert"
_TINY_GPT_NEO_MODEL_NAME = "hf-internal-testing/tiny-random-GPTNeoForCausalLM"
_TINY_BART_MODEL_NAME = "hf-internal-testing/tiny-random-BartForConditionalGeneration"
_TINY_VIT_MODEL_NAME = "hf-internal-testing/tiny-random-ViTForImageClassification"

TO_TEST = [
    ("masked-lm", _TINY_BERT_MODEL_NAME, 128),
    ("causal-lm", _TINY_GPT_NEO_MODEL_NAME, 128),
    ("text-classification", _TINY_BERT_MODEL_NAME, 128),
    ("token-classification", _TINY_BERT_MODEL_NAME, 384),
    ("multiple-choice", _TINY_BERT_MODEL_NAME, 384),
    ( "question-answering", _TINY_BERT_MODEL_NAME, 384),
    ("summarization", _TINY_BART_MODEL_NAME, [128, 128]),
    ("translation", _TINY_BART_MODEL_NAME, [128, 128]),
    ("image-classification", _TINY_VIT_MODEL_NAME, None),
]


@is_trainium_test
class TestExampleRunner(TestCase):

    @parameterized.expand(TO_TEST)
    def test_run_example(self, task, model_name_or_path, sequence_length):
        runner = ExampleRunner(model_name_or_path, task)

        def dummy_check_user_logged_in_and_cache_repo_is_set(self):
            pass

        # Doing this to avoid having to log in. We just test on one step so it should not be an issue.
        runner.check_user_logged_in_and_cache_repo_is_set = dummy_check_user_logged_in_and_cache_repo_is_set.__get__(
            runner
        )

        returncode, stdout, stderr = runner.run(1, "bf16", 1, sequence_length=sequence_length, max_steps=1)
        print(f"Standard output:\n{stdout}")
        print("=" * 50)
        print(f"Standard error:\n{stderr}")
        if returncode != 0:
            self.fail(f"ExampleRunner failed for task {task}.\nStandard output:\n{stdout}\nStandard error:\b{stderr}")
