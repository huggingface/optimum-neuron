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
from transformers.generation import StoppingCriteria

from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


@pytest.fixture(scope="module")
def model_and_tokenizer(neuron_decoder_path):
    model = NeuronModelForCausalLM.from_pretrained(neuron_decoder_path)
    tokenizer = AutoTokenizer.from_pretrained(neuron_decoder_path)
    yield (model, tokenizer)


def _test_generation(model, batch_size, input_length, **gen_kwargs):
    input_ids = torch.ones((batch_size, input_length), dtype=torch.int64)
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
def test_decoder_generation_base(model_and_tokenizer, gen_kwargs):
    model = model_and_tokenizer[0]
    _test_generation(model, model.batch_size, 10, **gen_kwargs)


@is_inferentia_test
@requires_neuronx
def test_decoder_generation_input_dimensions(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
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
def test_decoder_generation_custom_stopping_criteria(model_and_tokenizer):
    model = model_and_tokenizer[0]

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
def test_decoder_generation_padded_inputs(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
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
def test_decoder_generation_greedy_expectations(neuron_decoder_config):
    neuron_decoder_path = neuron_decoder_config["neuron_model_path"]
    model = NeuronModelForCausalLM.from_pretrained(neuron_decoder_path)
    tokenizer = AutoTokenizer.from_pretrained(neuron_decoder_path)
    prompt = "What is Deep Learning?"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, do_sample=False, max_new_tokens=17)
    expectations = {
        "gpt2": "\n\nDeep learning is a new field of research that has been around for a while",
        "llama": " and How Does it Work?\nDeep learning is a subset of machine learning that uses artificial",
        "mistral": "\nWhat is Deep Learning?\nDeep Learning is a type of machine learning that",
        "mixtral": "_+Azure marineictions spoonニolare又 Movement@Export좌╗personE przASS",  # This model has random weights
        "qwen2": " - Part 1\n\nDeep Learning is a subset of Machine Learning that is based on",
        "granite": "\n\nDeep Learning is a subset of Machine Learning, which is a branch of Art",
        "phi": "\n\nDeep learning is a subset of machine learning that uses neural networks with many",
    }
    config_name = neuron_decoder_config["name"]
    assert tokenizer.decode(outputs[0]).endswith(expectations[config_name])


@is_inferentia_test
@requires_neuronx
def test_decoder_generation_multiple_eos_token_ids(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    prompt = "Name three fruits:"
    tokens = tokenizer(prompt, return_tensors="pt")
    generation_config = copy.deepcopy(model.generation_config)
    if not isinstance(generation_config.eos_token_id, list):
        generation_config.eos_token_id = [generation_config.eos_token_id]
    generation_config.max_new_tokens = 256
    outputs = model.generate(**tokens, do_sample=True, generation_config=generation_config)
    # Extract the last non-eos generated token and use it as a fake eos_token_id
    fake_eos_token_id = outputs[0, -2]
    generation_config.eos_token_id.append(fake_eos_token_id)
    # Generate again an verify we stopped on that id
    outputs = model.generate(**tokens, do_sample=True, generation_config=generation_config)
    assert outputs[0, -1] == fake_eos_token_id


@is_inferentia_test
@requires_neuronx
def test_decoder_generation_stop_strings(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    prompt = "Name three fruits:"
    tokens = tokenizer(prompt, return_tensors="pt")
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.max_new_tokens = model.max_length - tokens["input_ids"].shape[-1]
    # Generate once
    outputs = model.generate(**tokens, do_sample=False, generation_config=generation_config)
    output_string = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    # Now create a generation_config with stop_strings corresponding to the beginning of the outputs
    sos = len(prompt)
    stop_string = output_string[sos : sos + 10]
    generation_config.stop_strings = [stop_string]
    # Generate and verify we stopped on the stop string
    outputs = model.generate(**tokens, do_sample=False, generation_config=generation_config, tokenizer=tokenizer)
    new_output_string = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    # Verify we stopped on the stop string
    assert len(new_output_string) < len(output_string)
    # Verify the stop string is in the generated string (but not necessarily exactly at the end because of tokenization)
    assert stop_string in output_string


@is_inferentia_test
@requires_neuronx
def test_continuous_batching_two_requests(model_and_tokenizer):
    """This test verifies that it is possible to:
    - prefill a first input at a first index,
    - decode a few tokens,
    - prefill a second input at a different index,
    - resume decoding for both inputs.
    Both generated tokens must match since we are using greedy.
    """
    model, tokenizer = model_and_tokenizer
    if not model.model.neuron_config.continuous_batching:
        pytest.skip("Model does not support continuous batching")

    # We need at least three inputs
    assert model.config.neuron["batch_size"] >= 3

    inputs = tokenizer("Once upon a time", return_tensors="pt")

    # A few helper functions we need while prefilling and decoding
    def get_next_tokens(input_ids, cache_ids, start_ids):
        outputs = model.forward(input_ids, cache_ids, start_ids)
        return outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    def increase_attention_mask(attention_mask):
        batch_size = attention_mask.shape[0]
        attention_mask_lengths = torch.sum(attention_mask, dim=1)
        attention_mask_length = attention_mask_lengths.max().item()
        new_attention_mask = torch.zeros(batch_size, attention_mask_length + 1, dtype=torch.int64)
        for i in range(batch_size):
            attention_mask_length = attention_mask_lengths[i].item() + 1
            new_attention_mask[i, :attention_mask_length] = 1
        return new_attention_mask

    # Prefill a single input at index 0, remembering the generated token
    first_inputs = model.prepare_inputs_for_prefill(**inputs, seq_ids=torch.tensor([0]))
    first_generated_tokens = get_next_tokens(**first_inputs)
    # Decode a few tokens
    next_tokens = first_generated_tokens
    attention_mask = inputs.attention_mask
    for _ in range(5):
        # For decode we can only pass the next token, but we need to pass the full attention mask
        attention_mask = increase_attention_mask(attention_mask)
        decode_inputs = model.prepare_inputs_for_decode(next_tokens, attention_mask)
        next_tokens = get_next_tokens(**decode_inputs)
        first_generated_tokens = torch.cat([first_generated_tokens, next_tokens], dim=-1)
    # Prefill a second input at index 2
    second_inputs = model.prepare_inputs_for_prefill(**inputs, seq_ids=torch.tensor([2]))
    second_generated_tokens = get_next_tokens(**second_inputs)
    # Concatenate the last decode token from the first input and the prefill token from the second
    next_tokens = torch.cat([next_tokens, second_generated_tokens], dim=0)
    second_attention_mask = torch.zeros_like(attention_mask)
    second_attention_mask[:, : inputs.attention_mask.shape[1]] = 1
    attention_mask = torch.cat([attention_mask, second_attention_mask], dim=0)
    # Decode more tokens
    for _ in range(10):
        attention_mask = increase_attention_mask(attention_mask)
        decode_inputs = model.prepare_inputs_for_decode(next_tokens, attention_mask, seq_ids=torch.tensor([0, 2]))
        next_tokens = get_next_tokens(**decode_inputs)
        first_generated_tokens = torch.cat([first_generated_tokens, next_tokens[0:1, :]], dim=-1)
        second_generated_tokens = torch.cat([second_generated_tokens, next_tokens[1:, :]], dim=-1)
    assert torch.equal(second_generated_tokens, first_generated_tokens[:, : second_generated_tokens.shape[1]])
