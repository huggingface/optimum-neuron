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

from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx
from optimum.utils.testing_utils import TOKEN


@is_inferentia_test
@requires_neuronx
def test_model_from_hub():
    model = NeuronModelForCausalLM.from_pretrained(
        "dacorvo/tiny-random-gpt2-neuronx", revision="b8f1aec89f9b278721068bfe616fa9227c1d0238"
    )
    check_neuron_model(model, batch_size=16, num_cores=2, auto_cast_type="fp32")


@is_inferentia_test
@requires_neuronx
def test_push_to_hub(neuron_model_path, neuron_push_id):
    model = NeuronModelForCausalLM.from_pretrained(neuron_model_path)
    model.push_to_hub(neuron_model_path, neuron_push_id, use_auth_token=TOKEN, endpoint=ENDPOINT_STAGING)
    api = HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)
    try:
        hub_files_info = api.list_files_info(neuron_push_id)
        hub_files_path = [info.rfilename for info in hub_files_info]
        for path, _, files in os.walk(neuron_model_path):
            for name in files:
                local_file_path = os.path.join(path, name)
                hub_file_path = os.path.relpath(local_file_path, neuron_model_path)
                assert hub_file_path in hub_files_path
    finally:
        api.delete_repo(neuron_push_id)
