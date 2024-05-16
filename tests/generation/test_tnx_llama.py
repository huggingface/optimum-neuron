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

import copy

import pytest
import torch
from transformers import AutoTokenizer

from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


@pytest.fixture(scope="module")
def neuron_model_config():
    model_id = "NousResearch/Hermes-2-Pro-Llama-3-8B"
    model_kwargs = {"batch_size": 4, "sequence_length": 4096, "auto_cast_type": "f16", "num_cores": 2}
    model = NeuronModelForCausalLM.from_pretrained(model_id, export=True, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    yield (model, tokenizer)


@is_inferentia_test
@requires_neuronx
def test_generation_llama_padded_inputs(neuron_model_config):
    model, tokenizer = neuron_model_config
    prompt = "One of my fondest memory is of my grandmother making homemade bread"
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
        outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, do_sample=False, max_new_tokens=10
        )
        # Verify we did not generate any unknown token
        assert torch.all(outputs[:, -1] != 0)


@is_inferentia_test
@requires_neuronx
def test_decoder_generation_multiple_eos_token_ids(neuron_model_config):
    model, tokenizer = neuron_model_config
    prompt = "Name three fruits:"
    tokens = tokenizer(prompt, return_tensors="pt")
    generation_config = copy.deepcopy(model.generation_config)
    if not isinstance(generation_config, list):
        generation_config.eos_token_id = [generation_config.eos_token_id]
    generation_config.max_new_tokens = model.max_length - tokens["input_ids"].shape[-1]
    # Generate and verify we stopped on an eos_token_id, and not on max_new_tokens
    outputs = model.generate(**tokens, do_sample=True, generation_config=generation_config)
    assert outputs.shape[-1] < model.max_length
    assert outputs[0, -1].numpy() in generation_config.eos_token_id
    # Extract the last non-eos generated token and use it as a fake eos_token_id
    fake_eos_token_id = outputs[0, -2]
    generation_config.eos_token_id.append(fake_eos_token_id)
    # Generate againg an verify we stopped on that id
    outputs = model.generate(**tokens, do_sample=True, generation_config=generation_config)
    assert outputs[0, -1] == fake_eos_token_id
