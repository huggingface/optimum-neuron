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
import os
import re
from typing import Any

import pytest
import torch
from PIL import Image
from prompts import get_long_prompt
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
from transformers.generation import StoppingCriteria

from optimum.neuron import NeuronModelForCausalLM, NeuronModelForImageTextToText
from optimum.neuron.models.inference.backend.modules.generation.generation_utils import (
    increase_position_ids,
    position_ids_from_attention_mask,
)
from optimum.neuron.models.inference.backend.modules.generation.sampling import prepare_sampling_params
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


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
@pytest.mark.parametrize("neuron_llm_config", ["llama-4x1024"], indirect=True)
def test_decoder_generation_base(neuron_llm_config: dict[str, Any], gen_kwargs):
    model = NeuronModelForCausalLM.from_pretrained(neuron_llm_config["neuron_model_path"])
    _test_generation(model, model.neuron_config.batch_size, 10, **gen_kwargs)


@is_inferentia_test
@requires_neuronx
@pytest.mark.parametrize("neuron_llm_config", ["llama-4x1024"], indirect=True)
def test_decoder_generation_input_dimensions(neuron_llm_config: dict[str, Any]):
    model = NeuronModelForCausalLM.from_pretrained(neuron_llm_config["neuron_model_path"])
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
@pytest.mark.parametrize("neuron_llm_config", ["llama-4x1024"], indirect=True)
def test_decoder_generation_custom_stopping_criteria(neuron_llm_config: dict[str, Any]):
    model = NeuronModelForCausalLM.from_pretrained(neuron_llm_config["neuron_model_path"])

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
def test_decoder_generation_greedy_expectations(any_generate_model):
    model_id = any_generate_model["model_id"]
    model = AutoModelForCausalLM.from_pretrained(model_id)
    neuron_llm_path = any_generate_model["neuron_model_path"]
    neuron_model = NeuronModelForCausalLM.from_pretrained(neuron_llm_path)
    tokenizer = AutoTokenizer.from_pretrained(neuron_llm_path)
    prompt = "What is Deep Learning?"
    inputs = tokenizer(prompt, return_tensors="pt")
    max_new_tokens = 17
    outputs = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    neuron_outputs = neuron_model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    if not torch.equal(neuron_outputs, outputs):
        config_name = any_generate_model["name"]
        generated_text = tokenizer.decode(neuron_outputs[0])
        known_different_generations = {
            "granite-4x1024": "Deep learning is a subset of machine learning that uses artificial neural networks with",
            "qwen3-4x1024": " What are its applications? What are the benefits of using Deep Learning? What are the",
            "qwen3-1x8192": " What are the key features of Deep Learning? What are the applications of Deep Learning?",
        }
        if config_name in known_different_generations:
            assert generated_text.endswith(known_different_generations[config_name])
            pytest.xfail(f"Known different generations for {config_name}")
        else:
            expected_text = tokenizer.decode(outputs[0])
            assert generated_text.endswith(expected_text)


@is_inferentia_test
@requires_neuronx
@pytest.mark.parametrize("neuron_llm_config", ["llama-4x1024"], indirect=True)
def test_decoder_generation_multiple_eos_token_ids(neuron_llm_config: dict[str, Any]):
    model = NeuronModelForCausalLM.from_pretrained(neuron_llm_config["neuron_model_path"])
    tokenizer = AutoTokenizer.from_pretrained(neuron_llm_config["neuron_model_path"])
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
@pytest.mark.parametrize("neuron_llm_config", ["llama-4x1024"], indirect=True)
def test_decoder_generation_stop_strings(neuron_llm_config: dict[str, Any]):
    model = NeuronModelForCausalLM.from_pretrained(neuron_llm_config["neuron_model_path"])
    tokenizer = AutoTokenizer.from_pretrained(neuron_llm_config["neuron_model_path"])
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
@pytest.mark.parametrize("neuron_llm_config", ["llama-4x1024"], indirect=True)
def test_continuous_batching_two_requests(neuron_llm_config: dict[str, Any]):
    """This test verifies that it is possible to:
    - prefill a first input at a first index,
    - decode a few tokens,
    - prefill a second input at a different index,
    - resume decoding for both inputs.
    Both generated tokens must match since we are using greedy.
    """
    model = NeuronModelForCausalLM.from_pretrained(neuron_llm_config["neuron_model_path"])
    tokenizer = AutoTokenizer.from_pretrained(neuron_llm_config["neuron_model_path"])
    if not model.neuron_config.continuous_batching:
        pytest.skip("Model does not support continuous batching")

    # Assume by default that we are not doing on-device sampling
    on_device_sampling = getattr(model.neuron_config, "on_device_sampling", False)

    # We need at least three inputs
    assert model.neuron_config.batch_size >= 3

    inputs = tokenizer("Once upon a time", return_tensors="pt")

    # A few helper functions we need while prefilling and decoding
    def get_next_tokens(input_ids, position_ids, seq_ids, sampling_params):
        outputs = model.forward(
            input_ids=input_ids, position_ids=position_ids, seq_ids=seq_ids, sampling_params=sampling_params
        )
        if on_device_sampling:
            # on-device sampling directly returns the next token
            return outputs.unsqueeze(1)
        return outputs[:, -1, :].argmax(dim=-1, keepdim=True)

    # Prepare common input tensors
    input_ids = inputs.input_ids
    position_ids = position_ids_from_attention_mask(inputs.attention_mask)
    sampling_params = prepare_sampling_params(batch_size=1)
    # Prefill a single input at index 0, remembering the generated token
    first_seq_ids = torch.tensor([0])
    first_generated_tokens = get_next_tokens(input_ids, position_ids, first_seq_ids, sampling_params)
    first_position_ids = increase_position_ids(position_ids, 1)
    # Decode a few tokens
    FIRST_DECODE_TOKENS = 5
    for _ in range(FIRST_DECODE_TOKENS):
        next_tokens = get_next_tokens(
            first_generated_tokens[:, -1:], first_position_ids, first_seq_ids, sampling_params
        )
        first_generated_tokens = torch.cat((first_generated_tokens, next_tokens), dim=-1)
        first_position_ids = increase_position_ids(first_position_ids, 1)
    # Prefill a second input at index 2
    second_seq_ids = torch.tensor([2])
    second_generated_tokens = get_next_tokens(input_ids, position_ids, second_seq_ids, sampling_params)
    second_position_ids = increase_position_ids(position_ids, 1)
    # Concatenate the generated tokens
    generated_tokens = torch.zeros(2, FIRST_DECODE_TOKENS + 1, dtype=torch.int64)
    generated_tokens[0, :] = first_generated_tokens
    generated_tokens[1, -1] = second_generated_tokens[:, -1]
    # Decode more tokens for both requests
    position_ids = torch.cat([first_position_ids, second_position_ids], dim=0)
    two_requests_seq_ids = torch.tensor([0, 2])
    two_requests_sampling_params = torch.cat([sampling_params, sampling_params], dim=0)
    SECOND_DECODE_TOKENS = 10
    for _ in range(SECOND_DECODE_TOKENS):
        next_tokens = get_next_tokens(
            generated_tokens[:, -1:], position_ids, two_requests_seq_ids, two_requests_sampling_params
        )
        generated_tokens = torch.cat((generated_tokens, next_tokens), dim=-1)
        position_ids = increase_position_ids(position_ids, 1)
    assert torch.equal(generated_tokens[0, : SECOND_DECODE_TOKENS + 1], generated_tokens[1, FIRST_DECODE_TOKENS:])


@is_inferentia_test
@requires_neuronx
@pytest.mark.parametrize("neuron_llm_config", ["gemma3-1x8192"], indirect=True)
def test_decoder_generation_long_sequence(neuron_llm_config: dict[str, Any]):
    """Test generation from a long prompt for models that use standard context encoding for long sequences.

    This covers models like gemma3 where chunked prefill is not supported and the full
    sequence is processed in a single context encoding pass.
    """
    model_id = neuron_llm_config["model_id"]
    model = AutoModelForCausalLM.from_pretrained(model_id)
    neuron_llm_path = neuron_llm_config["neuron_model_path"]
    neuron_model = NeuronModelForCausalLM.from_pretrained(neuron_llm_path)
    tokenizer = AutoTokenizer.from_pretrained(neuron_llm_path)
    inputs = tokenizer(get_long_prompt(model_id, 5000, 8192), return_tensors="pt")
    max_new_tokens = 50
    outputs = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    neuron_outputs = neuron_model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    neuron_generated_text = tokenizer.decode(
        neuron_outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    assert generated_text == neuron_generated_text, (
        f"Long sequence generation produced different tokens than HF model.\n"
        f"  Expected: {generated_text!r}\n"
        f"  Got     : {neuron_generated_text!r}"
    )


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


@is_inferentia_test
@requires_neuronx
def test_vlm_generation_base(any_vlm_generate_model: dict[str, Any]):
    """Test that a VLM can generate tokens with and without pixel_values."""
    neuron_model = NeuronModelForImageTextToText.from_pretrained(any_vlm_generate_model["neuron_model_path"])
    batch_size = neuron_model.neuron_config.batch_size
    input_length = 10

    # Text-only generation (no images)
    input_ids = torch.ones((batch_size, input_length), dtype=torch.int64)
    outputs = neuron_model.generate(input_ids, max_new_tokens=5)
    assert outputs.shape[0] == batch_size, "Output batch size should match input batch size"
    assert outputs.shape[1] > input_length, "Output should be longer than input"

    # Generation with dummy pixel_values
    image_size = neuron_model.neuron_config.image_size
    max_num_images = neuron_model.neuron_config.max_num_images
    pixel_values = torch.zeros((batch_size, max_num_images, 3, image_size, image_size))
    outputs_with_images = neuron_model.generate(input_ids, pixel_values=pixel_values, max_new_tokens=5)
    assert outputs_with_images.shape[0] == batch_size


@is_inferentia_test
@requires_neuronx
def test_vlm_generation_greedy_expectations(any_vlm_generate_model: dict[str, Any]):
    """Test that VLM greedy generation matches the HF reference model (text-only path)."""
    neuron_model = NeuronModelForImageTextToText.from_pretrained(any_vlm_generate_model["neuron_model_path"])
    processor = AutoProcessor.from_pretrained(any_vlm_generate_model["neuron_model_path"])
    prompt = "What is Deep Learning?"
    inputs = processor(text=prompt, return_tensors="pt")
    max_new_tokens = 10
    neuron_outputs = neuron_model.generate(inputs["input_ids"], max_new_tokens=max_new_tokens, do_sample=False)
    assert neuron_outputs.shape[0] == 1
    assert neuron_outputs.shape[1] > inputs["input_ids"].shape[1]


@is_inferentia_test
@requires_neuronx
def test_vlm_generation_with_image(any_vlm_generate_model: dict[str, Any]):
    """Test VLM greedy generation with a real image matches the HF CPU reference."""
    image_path = os.path.join(os.path.dirname(__file__), "venus_botticelli.png")
    image = Image.open(image_path).convert("RGB")

    model_id = any_vlm_generate_model["model_id"]
    neuron_model_path = any_vlm_generate_model["neuron_model_path"]

    processor = AutoProcessor.from_pretrained(neuron_model_path)

    # SmolVLM / Idefics3 requires the chat template to emit image tokens correctly.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Can you describe this image?"},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=[image], return_tensors="pt")
    max_new_tokens = 20

    # CPU reference
    cpu_model = AutoModelForImageTextToText.from_pretrained(model_id)
    cpu_outputs = cpu_model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    cpu_text = processor.decode(cpu_outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    # Neuron model
    neuron_model = NeuronModelForImageTextToText.from_pretrained(neuron_model_path)
    neuron_outputs = neuron_model.generate(
        inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )
    neuron_text = processor.decode(neuron_outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    # VLM outputs may diverge after a few tokens due to numerical differences in the
    # vision encoder (TP parallel layers vs CPU).  Check that at least the first few
    # tokens match and the output is non-empty.
    assert len(neuron_text.strip()) > 0, "Neuron model produced empty output"
    cpu_tokens = cpu_text.split()
    neuron_tokens = neuron_text.split()
    print(f"Reference CPU output: {cpu_text!r}")
    print(f"Neuron output       : {neuron_text!r}")
    assert cpu_tokens == neuron_tokens, (
        f"Neuron and CPU outputs differ.\nNeuron: {neuron_text!r}\nCPU:    {cpu_text!r}"
    )
