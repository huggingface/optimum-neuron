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
from tempfile import TemporaryDirectory

import pytest
import torch
from huggingface_hub import HfApi
from transformers import AutoTokenizer
from transformers.testing_utils import ENDPOINT_STAGING

from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx
from optimum.utils import logging
from optimum.utils.testing_utils import TOKEN, USER

from .exporters.exporters_utils import EXPORT_MODELS_TINY


logger = logging.get_logger()


DECODER_MODEL_ARCHITECTURES = ["gpt2"]


@pytest.fixture(scope="module", params=[EXPORT_MODELS_TINY[model_arch] for model_arch in DECODER_MODEL_ARCHITECTURES])
def export_model_id(request):
    return request.param


@pytest.fixture(scope="module")
def neuron_model_path(export_model_id):
    # For now we need to use a batch_size of 2 because it fails with batch_size == 1
    model = NeuronModelForCausalLM.from_pretrained(export_model_id, export=True, batch_size=2)
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


def _check_neuron_model(neuron_model, batch_size=None, num_cores=None, auto_cast_type=None):
    neuron_config = getattr(neuron_model.config, "neuron", None)
    assert neuron_config
    if batch_size:
        assert neuron_config["batch_size"] == batch_size
    if num_cores:
        assert neuron_config["num_cores"] == num_cores
    if auto_cast_type:
        assert neuron_config["auto_cast_type"] == auto_cast_type


@is_inferentia_test
@requires_neuronx
@pytest.mark.parametrize(
    "batch_size, num_cores, auto_cast_type",
    [
        pytest.param(1, 2, "f32", marks=pytest.mark.skip(reason="Does not work with batch_size 1")),
        [2, 2, "f32"],
        [2, 2, "bf16"],
    ],
)
def test_model_export(export_model_id, batch_size, num_cores, auto_cast_type):
    model = NeuronModelForCausalLM.from_pretrained(
        export_model_id, export=True, batch_size=batch_size, num_cores=num_cores, auto_cast_type=auto_cast_type
    )
    _check_neuron_model(model, batch_size, num_cores, auto_cast_type)


@is_inferentia_test
@requires_neuronx
def test_model_from_path(neuron_model_path):
    model = NeuronModelForCausalLM.from_pretrained(neuron_model_path)
    _check_neuron_model(model)


@is_inferentia_test
@requires_neuronx
def test_model_from_hub():
    model = NeuronModelForCausalLM.from_pretrained("dacorvo/tiny-random-gpt2-neuronx")
    _check_neuron_model(model)


def _test_model_generation(model, tokenizer, batch_size, length, **gen_kwargs):
    prompt_text = "Hello, I'm a language model,"
    prompts = [prompt_text for _ in range(batch_size)]
    tokens = tokenizer(prompts, return_tensors="pt")
    with torch.inference_mode():
        sample_output = model.generate(**tokens, min_length=length, max_length=length, **gen_kwargs)
        assert sample_output.shape[0] == batch_size
        assert sample_output.shape[1] == length


@is_inferentia_test
@requires_neuronx
@pytest.mark.parametrize(
    "gen_kwargs", [{"do_sample": True}, {"do_sample": True, "temperature": 0.7}], ids=["sample", "sample-with-temp"]
)
def test_model_generation(neuron_model_path, gen_kwargs):
    model = NeuronModelForCausalLM.from_pretrained(neuron_model_path)
    tokenizer = AutoTokenizer.from_pretrained(neuron_model_path)
    # Using static model parameters
    _test_model_generation(model, tokenizer, model.batch_size, model.max_length, **gen_kwargs)
    # Using a lower max length
    _test_model_generation(model, tokenizer, model.batch_size, model.max_length // 2, **gen_kwargs)
    # Using an incompatible batch_size
    with pytest.raises(ValueError, match="The specified batch_size"):
        _test_model_generation(model, tokenizer, model.batch_size + 1, model.max_length, **gen_kwargs)
    # Using an incompatible generation length
    with pytest.raises(ValueError, match="The current sequence length"):
        _test_model_generation(model, tokenizer, model.batch_size, model.max_length * 2, **gen_kwargs)


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
