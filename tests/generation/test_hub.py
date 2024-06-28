# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import re

from huggingface_hub import HfApi
from transformers.testing_utils import ENDPOINT_STAGING

from optimum.neuron import NeuronModelForSeq2SeqLM
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx
from optimum.utils.testing_utils import TOKEN, USER


def _test_push_to_hub(model, model_path, repo_id, ignore_patterns=[]):
    model.push_to_hub(model_path, repo_id, use_auth_token=TOKEN, endpoint=ENDPOINT_STAGING)
    api = HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)
    try:
        hub_files_path = api.list_repo_files(repo_id)
        for path, _, files in os.walk(model_path):
            for name in files:
                local_file_path = os.path.join(path, name)
                hub_file_path = os.path.relpath(local_file_path, model_path)
                excluded = False
                for pattern in ignore_patterns:
                    if re.compile(pattern).match(hub_file_path) is not None:
                        excluded = True
                        break
                assert excluded or hub_file_path in hub_files_path
    finally:
        api.delete_repo(repo_id)


def neuron_push_model_id(model_id):
    model_name = model_id.split("/")[-1]
    repo_id = f"{USER}/{model_name}-neuronx"
    return repo_id


@is_inferentia_test
@requires_neuronx
def test_seq2seq_model_from_hub():
    model = NeuronModelForSeq2SeqLM.from_pretrained(
        "Jingya/tiny-random-t5-neuronx", revision="43ea08d54b0a972e74b5bd22bc8112de021ece0c"
    )
    return model


@is_inferentia_test
@requires_neuronx
def test_push_seq2seq_to_hub(neuron_seq2seq_greedy_path, neuron_push_seq2seq_id):
    model = NeuronModelForSeq2SeqLM.from_pretrained(neuron_seq2seq_greedy_path)
    _test_push_to_hub(model, neuron_seq2seq_greedy_path, neuron_push_seq2seq_id)
