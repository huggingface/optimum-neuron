# ruff: noqa: E402 ignore imports not at top-level
from typing import Any

import pytest


# Do not collect tests from this file if vllm is not installed
pytest.importorskip("vllm")

from vllm import LLM, RequestOutput, SamplingParams
from vllm.platforms import current_platform

from optimum.neuron.utils.instance import current_instance_type
from optimum.neuron.vllm.platform import OptimumNeuronPlatform


def test_vllm_optimum_neuron_platform():
    assert isinstance(current_platform, OptimumNeuronPlatform)


def _test_vllm_generation(llm):
    prompts = ["One of my fondest memory is"]

    print(f"Prompt: {prompts[0]}")
    # First generation using greedy sampling
    sampling_params = [SamplingParams(top_k=1, max_tokens=10)]

    outputs = llm.generate(prompts, sampling_params)
    assert outputs is not None and len(outputs) == 1
    assert isinstance(outputs[0], RequestOutput)
    first_token_ids = outputs[0].outputs[0].token_ids
    print(f"Greedy output #1: {outputs[0].outputs[0].text}")
    assert len(first_token_ids) == 10

    # Second generation, still using greedy sampling
    outputs = llm.generate(prompts, sampling_params)
    second_token_ids = outputs[0].outputs[0].token_ids
    print(f"Greedy output #2: {outputs[0].outputs[0].text}")
    assert first_token_ids == second_token_ids

    # Third generation, now using top-k sampling
    sampling_params = [SamplingParams(top_k=10, max_tokens=10)]
    outputs = llm.generate(prompts, sampling_params)
    third_token_ids = outputs[0].outputs[0].token_ids
    print(f"Sample output: {outputs[0].outputs[0].text}")
    assert first_token_ids != third_token_ids


@pytest.mark.parametrize("neuron_llm_config", ["llama-4x4096"], indirect=True)
def test_vllm_from_neuron_model(neuron_llm_config: dict[str, Any]):
    """Test vLLm generation on a single model exported locally."""
    neuron_llm_path = neuron_llm_config["neuron_model_path"]
    llm = LLM(model=neuron_llm_path)
    _test_vllm_generation(llm)


def test_vllm_from_hub_model(any_generate_model: dict[str, Any]):
    """Test vLLm generation on all cached test models from the hub."""
    model_id = any_generate_model["model_id"]
    export_kwargs = any_generate_model["export_kwargs"]
    llm = LLM(
        model=model_id,
        max_num_seqs=export_kwargs["batch_size"],
        max_model_len=export_kwargs["sequence_length"],
        tensor_parallel_size=export_kwargs["tensor_parallel_size"],
    )
    _test_vllm_generation(llm)


@pytest.mark.parametrize("neuron_llm_config", ["llama-4x4096"], indirect=True)
def test_vllm_greedy_expectations(neuron_llm_config: dict[str, Any]):
    """Test vLLm greedy sampling on a single model exported locally."""
    neuron_llm_path = neuron_llm_config["neuron_model_path"]
    batch_size = neuron_llm_config["export_kwargs"]["batch_size"]
    llm = LLM(model=neuron_llm_path, max_num_seqs=batch_size)
    # Send more prompts than the compiled batch size (4) and request
    # varying generation lengths to test continuous batching.
    prompts = [
        "The president of the United States is",
        "The capital of France is",
        "It was a bright cold day in April, and the clocks were striking thirteen.",
        "I believe the meaning of life is",
        "The colour of the sky is",
        "One of my fondest memory is",
    ]

    sampling_params = [
        SamplingParams(top_k=1, max_tokens=10),
        SamplingParams(top_k=1, max_tokens=20),
        SamplingParams(top_k=1, max_tokens=30),
        SamplingParams(top_k=1, max_tokens=40),
        SamplingParams(top_k=1, max_tokens=10),
        SamplingParams(top_k=1, max_tokens=20),
    ]

    outputs = llm.generate(prompts, sampling_params)

    trn2 = current_instance_type() == "trn2"
    expected_outputs = [
        " the head of state and government of the United States",
        " Paris. The Eiffel Tower is located in Paris. The Eiffel Tower is " + ("one of" if trn2 else "a famous"),
        " The world was holding its breath as the world's top scientists and engineers gathered at the secret underground facility to witness the unveiling of the ultimate time machine.",
        " to find happiness and fulfillment in the present moment. It's a simple yet profound concept that can bring joy and peace to our lives.\n\nAs I reflect on my own life, I realize that I've",
        " blue, but what about the colour of the sky",
        " of my grandmother, who was a kind and gentle soul. She had a way of making everyone feel",
    ]

    for expected_output, output, sampling_param in zip(expected_outputs, outputs, sampling_params):
        completion_output = output.outputs[0]
        generated_text = completion_output.text
        print(f"Prompt: {output.prompt!r}, Generated text: {generated_text!r}")
        assert len(completion_output.token_ids) == sampling_param.max_tokens, (
            f"Expected {sampling_param.max_tokens} tokens, got {len(completion_output.token_ids)} generated tokens"
        )
        assert expected_output == generated_text
