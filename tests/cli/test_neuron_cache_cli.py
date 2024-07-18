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

import os
import random
import string
import subprocess
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from typing import Optional

from huggingface_hub import HfApi, create_repo, delete_repo
from huggingface_hub.utils import RepositoryNotFoundError
from transformers import BertConfig, BertModel, BertTokenizer, T5Tokenizer, T5Config, T5Model
from transformers.testing_utils import is_staging_test

from optimum.neuron.utils.cache_utils import (
    CACHE_REPO_NAME,
    load_custom_cache_repo_name_from_hf_home,
)
from optimum.neuron.utils.testing_utils import is_trainium_test
from optimum.utils.testing_utils import USER

from ..utils import StagingTestMixin


# Taken from https://pynative.com/python-generate-random-string/
def get_random_string(length: int) -> str:
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))


@is_trainium_test
class TestNeuronCacheCLI:
    def _optimum_neuron_cache_create(self, cache_repo_id: Optional[str] = None, public: bool = False):
            name_str = f"--name {cache_repo_id}" if cache_repo_id is not None else ""
            public_str = "--public" if public else ""
            command = f"optimum-cli neuron cache create {name_str} {public_str}".split()
            p = subprocess.Popen(command)
            _ = p.wait()
            try:
                repo_id = cache_repo_id if cache_repo_id is not None else CACHE_REPO_NAME
                info = HfApi().repo_info(repo_id, repo_type="model")
                assert info.private == (not public), "The privacy of the repo should match the presence of the --public flag."
                
            except RepositoryNotFoundError:
                pytest.fail("The repo was not created.")
            finally:
                delete_repo(repo_id)

            assert repo_id == load_custom_cache_repo_name_from_hf_home(), f"Saved local Neuron cache name should be equal to {repo_id}."

    def test_optimum_neuron_cache_create_with_custom_name(self, hub_test):
        seed = random.randint(0, 100)
        repo_id = f"{hub_test}-{seed}"
        return self._optimum_neuron_cache_create(cache_repo_id=repo_id)

    def test_optimum_neuron_cache_create_public_with_custom_name(self, hub_test):
        seed = random.randint(0, 100)
        repo_id = f"{hub_test}-{seed}"
        return self._optimum_neuron_cache_create(cache_repo_id=repo_id, public=True)

    def test_optimum_neuron_cache_set(self, hub_test):
        repo_id = hub_test
        command = f"optimum-cli neuron cache set {repo_id}".split()
        p = subprocess.Popen(command)
        returncode = p.wait()
        assert returncode == 0
        assert repo_id == load_custom_cache_repo_name_from_hf_home(), f"Saved local Neuron cache name should be equal to {repo_id}."

    def test_optimum_neuron_cache_add(self, hub_test):
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create dummy BERT model.
            bert_model_name = tmpdir / "bert_model"
            config = BertConfig()

            config.num_hidden_layers = 2
            config.num_attention_heads = 2
            config.vocab_size = 100

            mandatory_tokens = ["[UNK]", "[SEP]", "[CLS]"]

            with open(tmpdir / "bert_vocab.txt", "w") as fp:
                fp.write("\n".join([get_random_string(random.randint(10, 20))] + mandatory_tokens))

            tokenizer = BertTokenizer((tmpdir / "bert_vocab.txt").as_posix())
            tokenizer.save_pretrained(bert_model_name)

            model = BertModel(config)
            model.save_pretrained(bert_model_name)

            # Create dummy T5 model.
            t5_model_name = tmpdir / "t5_model"
            config = T5Config()

            config.num_hidden_layers = 2
            config.num_attention_heads = 2
            config.vocab_size = 100

            mandatory_tokens = ["[UNK]", "[SEP]", "[CLS]"]

            with open(tmpdir / "t5_vocab.txt", "w") as fp:
                fp.write("\n".join([get_random_string(random.randint(10, 20))] + mandatory_tokens))

            tokenizer = T5Tokenizer((tmpdir / "t5_vocab.txt").as_posix())
            tokenizer.save_pretrained(t5_model_name)

            model = T5Model(config)
            model.save_pretrained(t5_model_name)

            env = dict(os.environ)
            env["OPTIMUM_NEURON_DISABLE_IS_PRIVATE_REPO_CHECK"] = "1"

            # With wrong precision value, it should fail.
            command = (
                f"optimum-cli neuron cache add -m  {bert_model_name} --task text-classification --train_batch_size 1 "
                "--precision wrong --num_cores 2 --sequence_length 128"
            ).split()
            p = subprocess.Popen(command, env=env)
            returncode = p.wait()
            assert returncode != 0

            # With wrong num_cores value, it should fail.
            command = (
                f"optimum-cli neuron cache add -m {bert_model_name} --task text-classification --train_batch_size 1 "
                "--precision bf16 --num_cores 999 --sequence_length 128"
            ).split()
            p = subprocess.Popen(command, env=env)
            returncode = p.wait()
            assert returncode != 0

            # Non seq2seq model.
            command = (
                f"optimum-cli neuron cache add -m {bert_model_name} --task text-classification --train_batch_size 1 "
                "--precision bf16 --num_cores 2 --sequence_length 128"
            ).split()
            p = subprocess.Popen(command, stdout=subprocess.PIPE, env=env)
            stdout, _ = p.communicate()
            print(stdout)
            returncode = p.returncode
            assert returncode ==  0

            # seq2seq model.
            command = (
                f"optimum-cli neuron cache add -m {t5_model_name} --task translation --train_batch_size 1 --precision bf16 "
                "--num_cores 2 --encoder_sequence_length 12 --decoder_sequence_length 12"
            ).split()
            p = subprocess.Popen(command, env=env)
            returncode = p.wait()
            assert returncode ==  0
