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

from generation_utils import check_neuron_model
from huggingface_hub import HfApi
from transformers.testing_utils import ENDPOINT_STAGING

from optimum.neuron import NeuronModelForCausalLM, NeuronModelForSeq2SeqLM
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx
from optimum.utils.testing_utils import TOKEN


@is_inferentia_test
@requires_neuronx
def test_model_from_hub():
    model = NeuronModelForCausalLM.from_pretrained(
        "dacorvo/tiny-random-gpt2-neuronx", revision="6cb671b50db5cecb7abead9e2ec7099d4bab44a8"
    )
    check_neuron_model(model, batch_size=16, sequence_length=512, num_cores=2, auto_cast_type="fp32")


@is_inferentia_test
@requires_neuronx
def test_push_to_hub(neuron_decoder_path, neuron_push_decoder_id):
    model = NeuronModelForCausalLM.from_pretrained(neuron_decoder_path)
    model.push_to_hub(neuron_decoder_path, neuron_push_decoder_id, use_auth_token=TOKEN, endpoint=ENDPOINT_STAGING)
    api = HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)
    try:
        hub_files_info = api.list_files_info(neuron_push_decoder_id)
        hub_files_path = [info.rfilename for info in hub_files_info]
        for path, _, files in os.walk(neuron_decoder_path):
            for name in files:
                local_file_path = os.path.join(path, name)
                hub_file_path = os.path.relpath(local_file_path, neuron_decoder_path)
                assert hub_file_path in hub_files_path
    finally:
        api.delete_repo(neuron_push_decoder_id)


@is_inferentia_test
@requires_neuronx
def test_seq2seq_model_from_hub():
    model = NeuronModelForSeq2SeqLM.from_pretrained(
        "Jingya/tiny-random-t5-neuronx", revision="ce617676ce12a19df7c6bd523c69b83447fa036b"
    )
    return model


@is_inferentia_test
@requires_neuronx
def test_push_seq2seq_to_hub(neuron_seq2seq_greedy_path, neuron_push_seq2seq_id):
    model = NeuronModelForSeq2SeqLM.from_pretrained(neuron_seq2seq_greedy_path)
    model.push_to_hub(
        neuron_seq2seq_greedy_path, neuron_push_seq2seq_id, use_auth_token=TOKEN, endpoint=ENDPOINT_STAGING
    )
    api = HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)
    try:
        hub_files_info = api.list_files_info(neuron_push_seq2seq_id)
        hub_files_path = [info.rfilename for info in hub_files_info]
        for path, _, files in os.walk(neuron_seq2seq_greedy_path):
            for name in files:
                local_file_path = os.path.join(path, name)
                hub_file_path = os.path.relpath(local_file_path, neuron_seq2seq_greedy_path)
                assert hub_file_path in hub_files_path
    finally:
        api.delete_repo(neuron_push_seq2seq_id)
