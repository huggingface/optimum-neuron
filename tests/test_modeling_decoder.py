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
import torch
from transformers import AutoTokenizer

from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx
from optimum.utils import logging

from .exporters.exporters_utils import EXPORT_MODELS_TINY as MODEL_NAMES


logger = logging.get_logger()


DECODER_MODEL_ARCHITECTURES = ["gpt2"]


@pytest.fixture(scope="module", params=[MODEL_NAMES[model_arch] for model_arch in DECODER_MODEL_ARCHITECTURES])
def model_id(request):
    return request.param


@pytest.fixture(scope="module")
def neuron_model_path(model_id):
    # For now we need to use a batch_size of 2 because it fails with batch_size == 1
    model = NeuronModelForCausalLM.from_pretrained(model_id, export=True, batch_size=2)
    model_dir = TemporaryDirectory()
    model_path = model_dir.name
    model.save_pretrained(model_path)
    del model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(model_path)
    del tokenizer
    # Yield instead of returning to keep a reference to the temporary directory.
    # It will go out of scope and be released only once all tests needing the fixture
    # have been completed.
    yield model_path


def _check_neuron_model(neuron_model, batch_size=None, seq_length=None, tp_degree=None, amp=None):
    neuron_config = getattr(neuron_model.config, "neuron", None)
    assert neuron_config
    neuron_kwargs = neuron_config.get("neuron_kwargs", None)
    assert neuron_kwargs
    if batch_size:
        assert neuron_kwargs["batch_size"] == batch_size
    if seq_length:
        assert neuron_kwargs["n_positions"] == seq_length
    if tp_degree:
        assert neuron_kwargs["tp_degree"] == tp_degree
    if amp:
        assert neuron_kwargs["amp"] == amp


@is_inferentia_test
@requires_neuronx
@pytest.mark.parametrize(
    "batch_size, seq_length, tp_degree, amp",
    [
        pytest.param(1, 128, 2, "f32", marks=pytest.mark.skip(reason="Does not work with batch_size 1")),
        [2, 128, 2, "f32"],
        [2, 32, 2, "bf16"],
    ],
)
def test_model_from_hub(model_id, batch_size, seq_length, tp_degree, amp):
    model = NeuronModelForCausalLM.from_pretrained(
        model_id, export=True, batch_size=batch_size, n_positions=seq_length, tp_degree=tp_degree, amp=amp
    )
    _check_neuron_model(model, batch_size, seq_length, tp_degree, amp)


@is_inferentia_test
@requires_neuronx
def test_model_from_path(neuron_model_path):
    model = NeuronModelForCausalLM.from_pretrained(neuron_model_path)
    _check_neuron_model(model)


def _test_model_generation(model, tokenizer, batch_size, max_length, **gen_kwargs):
    prompt_text = "Hello, I'm a language model,"
    prompts = [prompt_text for _ in range(batch_size)]
    tokens = tokenizer(prompts, return_tensors="pt")
    model.reset_generation()
    with torch.inference_mode():
        sample_output = model.generate(**tokens, max_length=max_length, **gen_kwargs)
        assert sample_output.shape[0] == batch_size
        assert sample_output.shape[1] == max_length


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
