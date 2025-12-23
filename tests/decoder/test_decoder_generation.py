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
from optimum.neuron.models.inference.backend.modules.generation.generation_utils import prepare_sampling_params
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


@pytest.fixture(scope="module")
def model_and_tokenizer(base_neuron_llm_path):
    model = NeuronModelForCausalLM.from_pretrained(base_neuron_llm_path)
    tokenizer = AutoTokenizer.from_pretrained(base_neuron_llm_path)
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
    _test_generation(model, model.neuron_config.batch_size, 10, **gen_kwargs)


@is_inferentia_test
@requires_neuronx
def test_decoder_generation_input_dimensions(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    batch_size = model.neuron_config.batch_size
    sequence_length = model.neuron_config.sequence_length
    # Using valid input dimensions
    _test_generation(model, batch_size, sequence_length // 2)
    # Using an incompatible batch_size
    with pytest.raises(ValueError, match="The specified batch_size"):
        _test_generation(model, batch_size + 1, sequence_length)
    # Using an incompatible input length
    with pytest.raises(ValueError, match="The input sequence length"):
        _test_generation(model, batch_size, input_length=sequence_length * 2)


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
def test_decoder_generation_greedy_expectations(neuron_llm_config):
    neuron_llm_path = neuron_llm_config["neuron_model_path"]
    model = NeuronModelForCausalLM.from_pretrained(neuron_llm_path)
    tokenizer = AutoTokenizer.from_pretrained(neuron_llm_path)
    prompt = "What is Deep Learning?"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, do_sample=False, max_new_tokens=17)
    target = model.neuron_config.target
    expectations = {
        "llama": " and how does it work?\nDeep learning is a subset of machine learning that uses artificial",
        "qwen2": " - Part 1\n\nDeep Learning is a subset of Machine Learning"
        + (". It is a" if target == "trn2" else " that is based on"),
        "qwen3": " What is the difference between Deep Learning and Machine Learning?\n\nDeep Learning is a subset of",
        "granite": "\n\nDeep Learning is a subset of machine learning that is inspired by the structure and",
        "phi": "\n\nDeep learning is a subfield of machine learning that focuses on creating",
        "smollm3": " Deep learning is a subset of machine learning that uses neural networks with many layers to learn",
    }
    config_name = neuron_llm_config["name"]
    generated_text = tokenizer.decode(outputs[0])
    expected_text = expectations[config_name]
    assert generated_text.endswith(expected_text)


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
    outputs = model.generate(**tokens, do_sample=False, generation_config=generation_config)
    # Extract the last non-eos generated token and use it as a fake eos_token_id
    fake_eos_token_id = outputs[0, -2]
    generation_config.eos_token_id.append(fake_eos_token_id)
    # Generate again an verify we stopped on that id
    outputs = model.generate(**tokens, do_sample=False, generation_config=generation_config)
    assert outputs[0, -1] == fake_eos_token_id


@is_inferentia_test
@requires_neuronx
def test_decoder_generation_stop_strings(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    prompt = "Name three fruits:"
    tokens = tokenizer(prompt, return_tensors="pt")
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.max_new_tokens = model.neuron_config.sequence_length - tokens["input_ids"].shape[-1]
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
    if not model.neuron_config.continuous_batching:
        pytest.skip("Model does not support continuous batching")

    # Assume by default that we are not doing on-device sampling
    on_device_sampling = getattr(model.neuron_config, "on_device_sampling", False)

    # We need at least three inputs
    assert model.neuron_config.batch_size >= 3

    inputs = tokenizer("Once upon a time", return_tensors="pt")

    # A few helper functions we need while prefilling and decoding
    def get_next_tokens(**kwargs):
        outputs = model.forward(**kwargs)
        if on_device_sampling:
            # on-device sampling directly returns the next token
            return outputs.hidden_states.unsqueeze(1)
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
    first_inputs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "seq_ids": torch.tensor([0]),
    }
    if on_device_sampling:
        first_inputs["sampling_params"] = prepare_sampling_params(batch_size=1)
    first_generated_tokens = get_next_tokens(**model.prepare_inputs_for_prefill(**first_inputs))
    # Decode a few tokens
    first_inputs["input_ids"] = first_generated_tokens
    for _ in range(5):
        # For decode we can only pass the next token, but we need to pass the full attention mask
        first_inputs["attention_mask"] = increase_attention_mask(first_inputs["attention_mask"])
        next_tokens = get_next_tokens(**model.prepare_inputs_for_decode(**first_inputs))
        first_generated_tokens = torch.cat([first_generated_tokens, next_tokens], dim=-1)
        first_inputs["input_ids"] = next_tokens
    # Prefill a second input at index 2
    second_inputs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "seq_ids": torch.tensor([2]),
    }
    if on_device_sampling:
        second_inputs["sampling_params"] = prepare_sampling_params(batch_size=1)
    second_generated_tokens = get_next_tokens(**model.prepare_inputs_for_prefill(**second_inputs))
    # Resize the second request attention mask to the size of the first request
    second_attention_mask = torch.zeros_like(first_inputs["attention_mask"])
    second_attention_mask[:, : second_inputs["attention_mask"].shape[1]] = 1
    # Concatenate the last decode token from the first input and the prefill token from the second
    two_requests_inputs = {
        "input_ids": torch.cat([first_generated_tokens[:, -1:], second_generated_tokens], dim=0),
        "attention_mask": torch.cat([first_inputs["attention_mask"], second_attention_mask], dim=0),
        "seq_ids": torch.tensor([0, 2]),
    }
    if on_device_sampling:
        two_requests_inputs["sampling_params"] = prepare_sampling_params(batch_size=2)
    # Decode more tokens
    for _ in range(10):
        two_requests_inputs["attention_mask"] = increase_attention_mask(two_requests_inputs["attention_mask"])
        next_tokens = get_next_tokens(**model.prepare_inputs_for_decode(**two_requests_inputs))
        first_generated_tokens = torch.cat([first_generated_tokens, next_tokens[0:1, :]], dim=-1)
        second_generated_tokens = torch.cat([second_generated_tokens, next_tokens[1:, :]], dim=-1)
        two_requests_inputs["input_ids"] = next_tokens
    assert torch.equal(second_generated_tokens, first_generated_tokens[:, : second_generated_tokens.shape[1]])


@is_inferentia_test
@requires_neuronx
def test_generation_assisted_decoding(speculation):
    model_path, draft_model_path = speculation
    model = NeuronModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    assistant_model = NeuronModelForCausalLM.from_pretrained(draft_model_path)
    prompt = "What is Deep Learning?"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, do_sample=False, max_new_tokens=17, assistant_model=assistant_model)
    generated_text = tokenizer.decode(outputs[0])
    expected_text = " and How Does it Work?\nDeep learning is a subset of machine learning that uses artificial neural"
    assert generated_text.endswith(expected_text)
