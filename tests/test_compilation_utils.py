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
"""Tests for the compilation utilities."""

import os
from unittest import TestCase

from huggingface_hub import HfFolder
from parameterized import parameterized

from optimum.neuron.utils.cache_utils import (
    load_custom_cache_repo_name_from_hf_home,
    set_custom_cache_repo_name_in_hf_home,
)
from optimum.neuron.utils.compilation_utils import ExampleRunner
from optimum.neuron.utils.testing_utils import is_trainium_test


_TINY_BERT_MODEL_NAME = "hf-internal-testing/tiny-random-bert"
_TINY_GPT_NEO_MODEL_NAME = "hf-internal-testing/tiny-random-GPTNeoForCausalLM"
_TINY_BART_MODEL_NAME = "hf-internal-testing/tiny-random-BartForConditionalGeneration"
_TINY_VIT_MODEL_NAME = "hf-internal-testing/tiny-random-ViTForImageClassification"

TO_TEST = [
    ("masked-lm", _TINY_BERT_MODEL_NAME, 128),
    ("causal-lm", _TINY_GPT_NEO_MODEL_NAME, 128),
    ("text-classification", _TINY_BERT_MODEL_NAME, 128),
    ("token-classification", _TINY_BERT_MODEL_NAME, 384),
    ("multiple-choice", "hf-internal-testing/tiny-random-BertForMultipleChoice", 384),
    ("question-answering", _TINY_BERT_MODEL_NAME, 384),
    ("summarization", _TINY_BART_MODEL_NAME, [10, 10]),
    ("translation", _TINY_BART_MODEL_NAME, [10, 10]),
    ("image-classification", _TINY_VIT_MODEL_NAME, None),
]


@is_trainium_test
class TestExampleRunner(TestCase):
    CACHE_REPO_NAME = "optimum-internal-testing/optimum-neuron-cache-for-testing"

    @classmethod
    def setUpClass(cls) -> None:
        cls._token = HfFolder.get_token()
        cls._cache_repo_name = load_custom_cache_repo_name_from_hf_home()
        if os.environ.get("HF_TOKEN_OPTIMUM_NEURON_CI", None) is not None:
            token = os.environ.get("HF_TOKEN_OPTIMUM_NEURON_CI")
            set_custom_cache_repo_name_in_hf_home(cls.CACHE_REPO_NAME)
            HfFolder.save_token(token)
        else:
            raise RuntimeError("Please specify the token via the HF_TOKEN_OPTIMUM_NEURON_CI environment variable.")

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._token is not None:
            HfFolder.save_token(cls._token)
        if cls._cache_repo_name is not None:
            set_custom_cache_repo_name_in_hf_home(cls._cache_repo_name)

    @parameterized.expand(TO_TEST)
    def test_run_example(self, task, model_name_or_path, sequence_length):
        runner = ExampleRunner(model_name_or_path, task)
        returncode, stdout, stderr = runner.run(
            1, "bf16", 1, sequence_length=sequence_length, max_steps=10, save_steps=5
        )
        print(f"Standard output:\n{stdout}")
        print("=" * 50)
        print(f"Standard error:\n{stderr}")
        if returncode != 0:
            self.fail(f"ExampleRunner failed for task {task}.\nStandard output:\n{stdout}\nStandard error:\b{stderr}")
