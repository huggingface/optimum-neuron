# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import pytest
import torch
from transformers import AutoTokenizer

from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


def _test_model_generation(model, tokenizer, batch_size, input_length, **gen_kwargs):
    input_ids = torch.ones((batch_size, input_length), dtype=torch.int64)
    with torch.inference_mode():
        sample_output = model.generate(input_ids, **gen_kwargs)
        assert sample_output.shape[0] == batch_size


@pytest.mark.parametrize(
    "gen_kwargs",
    [
        {"do_sample": True},
        {"do_sample": True, "temperature": 0.7},
        {"do_sample": False},
        {"do_sample": False, "repetition_penalty": 1.2},
    ],
    ids=["sample", "sample-with-temp", "greedy", "greedy_no-repeat"],
)
@is_inferentia_test
@requires_neuronx
def test_model_generation(neuron_model_path, gen_kwargs):
    model = NeuronModelForCausalLM.from_pretrained(neuron_model_path)
    tokenizer = AutoTokenizer.from_pretrained(neuron_model_path)
    _test_model_generation(model, tokenizer, model.batch_size, 10, **gen_kwargs)


@is_inferentia_test
@requires_neuronx
def test_model_generation_input_dimensions(neuron_model_path):
    model = NeuronModelForCausalLM.from_pretrained(neuron_model_path)
    tokenizer = AutoTokenizer.from_pretrained(neuron_model_path)
    # Using valid input dimensions
    _test_model_generation(model, tokenizer, model.batch_size, model.max_length // 2)
    # Using an incompatible batch_size
    with pytest.raises(ValueError, match="The specified batch_size"):
        _test_model_generation(model, tokenizer, model.batch_size + 1, model.max_length)
    # Using an incompatible input length
    with pytest.raises(ValueError, match="The input sequence length"):
        _test_model_generation(model, tokenizer, model.batch_size, input_length=model.max_length * 2)
