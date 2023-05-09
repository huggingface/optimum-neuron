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

import os
import subprocess
from subprocess import PIPE
from tempfile import TemporaryDirectory
from unittest import TestCase

from huggingface_hub import HfApi, create_repo, delete_repo
from huggingface_hub.utils import RepositoryNotFoundError
from transformers.testing_utils import is_staging_test

from optimum.neuron.utils.cache_utils import (
    CACHE_REPO_FILENAME,
    CACHE_REPO_NAME,
    load_custom_cache_repo_name_from_hf_home,
)

from ..utils import USER, StagingTestMixin


@is_staging_test
class TestNeuronCacheCLI(StagingTestMixin, TestCase):
    def setUp(self):
        self._hf_home = os.environ.get("HF_HOME", "")

        self.repo_name = "blabla"
        self.repo_id = f"{USER}/{self.repo_name}"

        self.default_repo_name = CACHE_REPO_NAME
        self.default_repo_id = f"{USER}/{self.default_repo_name}"

    def tearDown(self):
        os.environ["HF_HOME"] = self._hf_home

        try:
            delete_repo(self.default_repo_id, repo_type="model")
        except RepositoryNotFoundError:
            pass

        try:
            delete_repo(self.repo_id, repo_type="model")
        except RepositoryNotFoundError:
            pass

    def _optimum_neuron_cache_create(self, default_name: bool = True, public: bool = False):
        with TemporaryDirectory() as tmpdirname:
            os.environ["HF_HOME"] = tmpdirname

            repo_id = self.default_repo_id if default_name else self.repo_id

            env = dict(self._env, HF_HOME=tmpdirname)

            command = f"huggingface-cli login --token {self._staging_token}".split()
            p = subprocess.Popen(command, env=env)
            returncode = p.wait()
            self.assertEqual(returncode, 0)

            name_str = f"--name {self.repo_name}" if not default_name else ""
            public_str = "--public" if public else ""
            command = f"optimum-cli neuron cache create {name_str} {public_str}".split()
            p = subprocess.Popen(command, env=env)
            returncode = p.wait()

            try:
                info = HfApi().repo_info(repo_id, repo_type="model")
                self.assertEqual(
                    info.private, not public, "The privacy of the repo should match the presence of the --public flag."
                )
            except RepositoryNotFoundError:
                self.fail("The repo was not created.")

            hf_home_cache_repo_file = f"{tmpdirname}/{CACHE_REPO_FILENAME}"
            self.assertEqual(
                repo_id,
                load_custom_cache_repo_name_from_hf_home(hf_home_cache_repo_file),
                f"Saved local Trainium cache name should be equal to {repo_id}.",
            )

    def test_optimum_neuron_cache_create_with_default_name(self):
        return self._optimum_neuron_cache_create(public=False)

    def test_optimum_neuron_cache_create_public_with_default_name(self):
        return self._optimum_neuron_cache_create(public=True)

    def test_optimum_neuron_cache_create_with_custom_name(self):
        return self._optimum_neuron_cache_create(default_name=False)

    def test_optimum_neuron_cache_set(self):
        with TemporaryDirectory() as tmpdirname:
            os.environ["HF_HOME"] = tmpdirname

            create_repo(self.repo_name, repo_type="model")

            command = f"optimum-cli neuron cache set --name {self.repo_id}".split()
            env = dict(self._env, HF_HOME=tmpdirname)
            p = subprocess.Popen(command, env=env)
            returncode = p.wait()
            self.assertEqual(returncode, 0)

            hf_home_cache_repo_file = f"{tmpdirname}/{CACHE_REPO_FILENAME}"
            self.assertEqual(
                self.repo_id,
                load_custom_cache_repo_name_from_hf_home(hf_home_cache_repo_file),
                f"Saved local Trainium cache name should be equal to {self.repo_id}.",
            )

    def test_optimum_neuron_cache_add(self):
        # TODO: activate those later.
        # Without any sequence length, it should fail.
        # command = (
        #     "optimum-cli neuron cache add -m bert-base-uncased --task text-classification --train_batch_size 16 "
        #     "--precision bf16 --num_cores 2"
        # ).split()
        # p = subprocess.Popen(command, stderr=PIPE)
        # _, stderr = p.communicate()
        # stderr = stderr.decode("utf-8")
        # self.assertIn("either sequence_length or encoder_sequence and decoder_sequence_length", stderr)

        # Without both encoder and decoder sequence lengths, it should fail.
        # command = (
        #     "optimum-cli neuron cache add -m t5-small --task translation --train_batch_size 16 --precision bf16 "
        #     "--num_cores 2 --encoder_sequence_length 512"
        # ).split()
        # p = subprocess.Popen(command, stderr=PIPE)
        # _, stderr = p.communicate()
        # stderr = stderr.decode("utf-8")
        # self.assertIn("Both the encoder_sequence and decoder_sequence_length", stderr)

        # With wrong precision value, it should fail.
        command = (
            "optimum-cli neuron cache add -m bert-base-uncased --task text-classification --train_batch_size 1 "
            "--precision wrong --num_cores 2 --sequence_length 128"
        ).split()
        p = subprocess.Popen(command)
        returncode = p.wait()
        self.assertNotEqual(returncode, 0)

        # With wrong num_cores value, it should fail.
        command = (
            "optimum-cli neuron cache add -m bert-base-uncased --task text-classification --train_batch_size 1 "
            "--precision bf16 --num_cores 999 --sequence_length 128"
        ).split()
        p = subprocess.Popen(command)
        returncode = p.wait()
        self.assertNotEqual(returncode, 0)


        # Non seq2seq model.
        command = (
            "optimum-cli neuron cache add -m bert-base-uncased --task text-classification --train_batch_size 1 "
            "--precision bf16 --num_cores 2 --sequence_length 128"
        ).split()
        p = subprocess.Popen(command)
        returncode = p.wait()
        self.assertEqual(returncode, 0)

        # seq2seq model.
        command = (
            "optimum-cli neuron cache add -m t5-small --task translation --train_batch_size 1 --precision bf16 "
            "--num_cores 2 --encoder_sequence_length 12 --decoder_sequence_length 12"
        ).split()
        p = subprocess.Popen(command)
        returncode = p.wait()
        self.assertEqual(returncode, 0)
