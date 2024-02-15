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
from transformers.generation import StoppingCriteria

from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


def _test_generation(model, batch_size, input_length, **gen_kwargs):
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
def test_decoder_generation(neuron_decoder_path, gen_kwargs):
    model = NeuronModelForCausalLM.from_pretrained(neuron_decoder_path)
    _test_generation(model, model.batch_size, 10, **gen_kwargs)


@is_inferentia_test
@requires_neuronx
def test_model_generation_input_dimensions(neuron_decoder_path):
    model = NeuronModelForCausalLM.from_pretrained(neuron_decoder_path)
    AutoTokenizer.from_pretrained(neuron_decoder_path)
    # Using valid input dimensions
    _test_generation(model, model.batch_size, model.max_length // 2)
    # Using an incompatible batch_size
    with pytest.raises(ValueError, match="The specified batch_size"):
        _test_generation(model, model.batch_size + 1, model.max_length)
    # Using an incompatible input length
    with pytest.raises(ValueError, match="The input sequence length"):
        _test_generation(model, model.batch_size, input_length=model.max_length * 2)


@is_inferentia_test
@requires_neuronx
def test_decoder_generation_custom_stopping_criteria(neuron_decoder_path):
    model = NeuronModelForCausalLM.from_pretrained(neuron_decoder_path)

    class CustomStoppingCriteria(StoppingCriteria):
        def __init__(self):
            self.called = False

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            self.called = True
            return True

    criteria = CustomStoppingCriteria()
    model.generate(input_ids=torch.ones([1, 10], dtype=torch.int64), stopping_criteria=[criteria])
    assert criteria.called, "Custom StoppingCriteria should have been called"


@is_inferentia_test
@requires_neuronx
def test_decoder_generation_padded_inputs(neuron_decoder_path):
    model = NeuronModelForCausalLM.from_pretrained(neuron_decoder_path)
    assert model.batch_size >= 2
    tokenizer = AutoTokenizer.from_pretrained(neuron_decoder_path)
    prompt = (
        "It was a bright cold day in April, and the clocks were striking thirteen."
        " Winston Smith, his chin nuzzled into his breast in an effort to escape the"
        " vile wind, slipped quickly through the glass doors of Victory Mansions,"
    )
    first_input = tokenizer(prompt)
    first_ids = first_input["input_ids"]
    first_mask = first_input["attention_mask"]
    max_padding = 12
    input_len = len(first_ids)
    for i in range(max_padding):
        second_ids = [tokenizer.eos_token_id] * i + first_ids[: input_len - i]
        second_mask = [0] * i + [1] * (input_len - i)
        input_ids = torch.tensor([first_ids, second_ids], dtype=torch.int64)
        attention_mask = torch.tensor([first_mask, second_mask], dtype=torch.int64)
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, do_sample=False)
        # Verify we did not generate any unknown token
        assert torch.all(outputs[:, -1] != 0)
