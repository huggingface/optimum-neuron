# ruff: noqa: E402 ignore imports not at top-level
import pytest


# Do not collect tests from this file if vllm is not installed
pytest.importorskip("vllm")

from vllm import LLM, RequestOutput, SamplingParams
from vllm.platforms import current_platform

from optimum.neuron.utils import map_torch_dtype
from optimum.neuron.vllm.platform import OptimumNeuronPlatform


def test_vllm_optimum_neuron_platform():
    assert isinstance(current_platform, OptimumNeuronPlatform)


def _test_vllm_generation(llm):
    prompts = ["One of my fondest memory is"]

    # First generation using greedy sampling
    sampling_params = [SamplingParams(top_k=1, max_tokens=10)]

    outputs = llm.generate(prompts, sampling_params)
    assert outputs is not None and len(outputs) == 1
    assert isinstance(outputs[0], RequestOutput)
    first_token_ids = outputs[0].outputs[0].token_ids
    assert len(first_token_ids) == 10

    # Second generation, still using greedy sampling
    outputs = llm.generate(prompts, sampling_params)
    second_token_ids = outputs[0].outputs[0].token_ids
    assert first_token_ids == second_token_ids

    # Third generation, now using top-k sampling
    sampling_params = [SamplingParams(top_k=10, max_tokens=10)]
    outputs = llm.generate(prompts, sampling_params)
    third_token_ids = outputs[0].outputs[0].token_ids
    assert first_token_ids != third_token_ids


def test_vllm_from_neuron_model(base_neuron_decoder_path):
    """Test vLLm generation on a single model exported locally."""
    llm = LLM(model=base_neuron_decoder_path, device="neuron")
    _test_vllm_generation(llm)


def test_vllm_from_hub_model(neuron_decoder_config):
    """Test vLLm generation on all cached test models from the hub."""
    model_id = neuron_decoder_config["model_id"]
    export_kwargs = neuron_decoder_config["export_kwargs"]
    llm = LLM(
        model=model_id,
        max_num_seqs=export_kwargs["batch_size"],
        max_model_len=export_kwargs["sequence_length"],
        tensor_parallel_size=export_kwargs["num_cores"],
        dtype=map_torch_dtype(export_kwargs["auto_cast_type"]),
        device="neuron",
    )
    _test_vllm_generation(llm)


def test_vllm_greedy_expectations(neuron_decoder_config):
    """Test vLLm generation on all test model types."""
    llm = LLM(
        model=neuron_decoder_config["neuron_model_path"],
        max_num_seqs=neuron_decoder_config["export_kwargs"]["batch_size"],
        device="neuron",
    )
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

    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"{generated_text!r},")

    expected_outputs = {
        "llama": [
            " the head of state and government of the United States",
            " Paris. The Eiffel Tower is located in Paris. The Louvre Museum is also located in",
            " The world was holding its breath as the world's top scientists and engineers gathered at the secret underground facility to witness the unveiling of the ultimate time machine.\n",
            " to find happiness and fulfillment in our daily lives. It's a journey, not a destination. We all have our own unique path to follow, and it's up to each of us to find what",
            " blue, but what about the colour of the sky",
            " of my grandmother's kitchen, where I spent countless hours helping her in the kitchen. She was a",
        ],
        "qwen2": [
            " a very important person. He is the leader of",
            " Paris. It is the largest city in Europe and the second largest in the world. It is also",
            " I was sitting in my room, staring at the ceiling, when the door opened and in came a man with a big, bushy beard. “",
            " to be happy. I believe that happiness is to be able to enjoy life. I believe that happiness is to be able to enjoy life in a way that is not limited by circumstances. I believe that",
            " changing. The sky is getting darker and darker.",
            " of a time when I was a child and my mother was in the hospital. I was in the",
        ],
        "granite": [
            " the head of the state, and the vice pres",
            " Paris.\n\nThe capital of France is not a city, it's a city-state",
            '\n\nNurse: "I\'m sorry, Mr. Anderson, but I can\'t do that.\n\n"Clocks": "',
            " to find and share our unique experiences, our love, and our sense of purpose, and to learn from each other, and to grow and to share in the process.\n\nLet's break down",
            " blue. The colour of the sky is blue.",
            " of a time when I was in the 7th grade and my school's basketball team was",
        ],
        "phi": [
            " elected through a process known as the Electoral College",
            " Paris.\n\n- [Query]: What is the capital of Germany?\n\n- [Response",
            " Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind, slipped quickly through the glass",
            " to find happiness and fulfillment in our relationships and personal growth.\n\n**Chatbot**: That's a beautiful perspective. It's important to find joy in the connections we",
            " blue.\n\n\n### Response\n\n",
            " of my first day at school. I was a shy, timid child, and the thought",
        ],
        "qwen3": [
            " the same as the president of the United Nations.",
            " Paris. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of China",
            " The clockmaker's wife was in the kitchen, baking cookies. She baked 12 dozen cookies in the morning and 18 dozen in the",
            " to find a balance between the two things that are important to me. I want to find a way to make sure that I can do this. I am not sure if I can do this, but",
            " due to the absorption of light by the atmosphere.",
            " the time I was in the first grade. I remember the smell of the school, the sound of",
        ],
    }[neuron_decoder_config["name"]]

    for expected_output, output in zip(expected_outputs, outputs):
        generated_text = output.outputs[0].text
        print(f"Prompt: {output.prompt!r}, Generated text: {generated_text!r}")
        assert expected_output == generated_text
