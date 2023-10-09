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
from tempfile import TemporaryDirectory

import pytest
from transformers import AutoTokenizer

from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.utils.testing_utils import requires_neuronx
from optimum.utils.testing_utils import USER


DECODER_MODEL_ARCHITECTURES = ["gpt2", "llama"]
DECODER_MODEL_NAMES = {
    "gpt2": "hf-internal-testing/tiny-random-gpt2",
    "llama": "dacorvo/tiny-random-llama",
}


@pytest.fixture(scope="module", params=[DECODER_MODEL_NAMES[model_arch] for model_arch in DECODER_MODEL_ARCHITECTURES])
def export_model_id(request):
    return request.param


@pytest.fixture(scope="module")
@requires_neuronx
def neuron_model_path(export_model_id):
    model = NeuronModelForCausalLM.from_pretrained(
        export_model_id, export=True, batch_size=1, sequence_length=128, num_cores=2
    )
    model_dir = TemporaryDirectory()
    model_path = model_dir.name
    model.save_pretrained(model_path)
    del model
    tokenizer = AutoTokenizer.from_pretrained(export_model_id)
    tokenizer.save_pretrained(model_path)
    del tokenizer
    # Yield instead of returning to keep a reference to the temporary directory.
    # It will go out of scope and be released only once all tests needing the fixture
    # have been completed.
    yield model_path


@pytest.fixture(scope="module")
def neuron_push_id(export_model_id):
    model_name = export_model_id.split("/")[-1]
    repo_id = f"{USER}/{model_name}-neuronx"
    return repo_id


def check_neuron_model(neuron_model, batch_size=None, sequence_length=None, num_cores=None, auto_cast_type=None):
    neuron_config = getattr(neuron_model.config, "neuron", None)
    assert neuron_config
    if batch_size:
        assert neuron_config["batch_size"] == batch_size
    if sequence_length:
        assert neuron_config["sequence_length"] == sequence_length
    if num_cores:
        assert neuron_config["num_cores"] == num_cores
    if auto_cast_type:
        assert neuron_config["auto_cast_type"] == auto_cast_type
