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
import re

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    model_id = neuron_llm_config["model_id"]
    model = AutoModelForCausalLM.from_pretrained(model_id)
    neuron_llm_path = neuron_llm_config["neuron_model_path"]
    neuron_model = NeuronModelForCausalLM.from_pretrained(neuron_llm_path)
    tokenizer = AutoTokenizer.from_pretrained(neuron_llm_path)
    prompt = "What is Deep Learning?"
    inputs = tokenizer(prompt, return_tensors="pt")
    max_new_tokens = 17
    outputs = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    neuron_outputs = neuron_model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    if not torch.equal(neuron_outputs, outputs):
        config_name = neuron_llm_config["name"]
        generated_text = tokenizer.decode(neuron_outputs[0])
        known_different_generations = {
            "qwen3": " What are the key features of Deep Learning? What are the applications of Deep Learning?",
        }
        if config_name in known_different_generations:
            assert generated_text.endswith(known_different_generations[config_name])
            pytest.xfail(f"Known different generations for {config_name}")
        else:
            expected_text = tokenizer.decode(outputs[0])
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
@pytest.mark.parametrize(
    "max_new_tokens",
    [17, 30],
    ids=["shorter", "short"],
)
def test_speculation_same_model(caplog, speculation, max_new_tokens):
    """Test the generation from a model using the same model as an assistant for speculation.
    We check that the number of speculated tokens logged correspond to what we expect,
    and that the final generated tokens are as expected.
    Since the assistant model is identical to the main model, the number of accepted speculated tokens
    should always be equal to the speculation length or the remainder of the expected number of tokens.
    It is not always true and small changes may happen because the speculation and decode graphs
    are subtly different, and may result in small changes in predictions, for example in capitalization.
    The prompt and generated text in this test have been chosen to minimize this risk.
    """
    model_path, draft_model_path = speculation
    model = NeuronModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    assistant_model = NeuronModelForCausalLM.from_pretrained(draft_model_path)
    prompt = "One of my fondest memories is of my grandmother's kitchen"
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_length = inputs["input_ids"].shape[1]
    with caplog.at_level("INFO"):
        # Generate and capture the logged speculated tokens
        outputs = model.generate(
            **inputs, do_sample=False, max_new_tokens=max_new_tokens, assistant_model=assistant_model
        )
        generated_tokens = outputs[0, prompt_length:].tolist()
        # Verify first we generated the expected number of tokens, otherwise we cannot check the expected text
        assert len(generated_tokens) == max_new_tokens
        expected_text = (
            ", where I spent countless hours helping her in the kitchen."
            " She was a master baker, and her kitchen was always filled with the most wonderful aromas"
        )
        generated_text = tokenizer.decode(generated_tokens)
        assert expected_text.startswith(generated_text)
        # Check the logged speculated tokens
        remaining_speculated_tokens = max_new_tokens - 1  # First token is not speculated
        for record in caplog.records:
            msg = record.msg
            prefix = "Assisted decoding: accepted"
            if msg.startswith(prefix):
                speculated_tokens = int(re.findall(r"\b\d+\b", msg)[0])
                # We expect either the full speculation length or the remaining tokens to generate
                expected_speculated_tokens = min(model.neuron_config.speculation_length, remaining_speculated_tokens)
                assert speculated_tokens == expected_speculated_tokens, (
                    f"Expected {expected_speculated_tokens} speculated tokens, got {speculated_tokens}"
                )
                remaining_speculated_tokens -= speculated_tokens
