# ruff: noqa: E402 ignore imports not at top-level
import pytest


# Do not collect tests from this file if vllm is not installed
pytest.importorskip("vllm")

from vllm import LLM, RequestOutput, SamplingParams
from vllm.platforms import current_platform

from optimum.neuron.utils import DTYPE_MAPPER
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


def test_vllm_from_neuron_model(base_neuron_llm_path):
    """Test vLLm generation on a single model exported locally."""
    llm = LLM(model=base_neuron_llm_path)
    _test_vllm_generation(llm)


def test_vllm_from_hub_model(neuron_llm_config):
    """Test vLLm generation on all cached test models from the hub."""
    model_id = neuron_llm_config["model_id"]
    export_kwargs = neuron_llm_config["export_kwargs"]
    llm = LLM(
        model=model_id,
        max_num_seqs=export_kwargs["batch_size"],
        max_model_len=export_kwargs["sequence_length"],
        tensor_parallel_size=export_kwargs["tensor_parallel_size"],
        dtype=DTYPE_MAPPER.pt(export_kwargs["auto_cast_type"]),
    )
    _test_vllm_generation(llm)


def test_vllm_greedy_expectations(neuron_llm_config):
    """Test vLLm generation on all test model types."""
    if neuron_llm_config["name"] == "qwen3":
        pytest.xfail("Qwen3 outputs are not deterministic with greedy sampling")
    llm = LLM(
        model=neuron_llm_config["neuron_model_path"],
        max_num_seqs=neuron_llm_config["export_kwargs"]["batch_size"],
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

    expected_outputs = {
        "llama": [
            " the head of state and government of the United States",
            " Paris. The Eiffel Tower is located in Paris. The Louvre Museum is also located in",
            " The world was holding its breath as the world's top scientists and engineers gathered at the secret underground facility to witness the unveiling of the ultimate time machine.\n",
            " to find happiness and fulfillment in the present moment. It's a simple yet profound concept that can bring joy and peace to our lives.\n\nAs I reflect on my own life, I realize that I've",
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
            " the head of state and government, and the command",
            " Paris.\n\nParis is located in the northern part of the country, on the",
            '\n\nThis opening line from George Orwell\'s dystopian novel "1984" sets the tone for a world where reality',
            " to give it meaning.\" - Viktor Frankl\n\nThis quote encapsulates Frankl's perspective on the human condition, particularly in the face of existential anxiety and suffering.",
            " blue.\n\nThe colour of the grass",
            " of a trip to the Grand Canyon. The vastness of the canyon, the",
        ],
        "phi": [
            " elected every four years.\n\n\n###",
            " Paris.\n\n\n### Response:The capital of France is Paris.",
            " Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind, slipped quickly through the glass",
            " to find happiness and fulfillment in our relationships and personal growth.\n\n**Chatbot**: That's a beautiful perspective. It's important to find joy in the connections we",
            " blue.\n\n\n### Response\n\n",
            " of my first day at school. I was a shy, timid child, and the thought",
        ],
        "qwen3": [
            " the same as the president of the United Nations.",
            " Paris. The capital of France is also the capital of the French Republic. The capital of France is",
            " The clockmaker's wife was in the kitchen, baking cookies. She baked 12 dozen cookies in the morning and 10 dozen in the",
            " to find a balance between the two things that are important to me. I want to find a balance between the two things that are important to me. I want to find a balance between the two things",
            " due to the absorption of light by the atmosphere.",
            " the time I was in the first grade. I remember the day I got the first grade, the",
        ],
        "smollm3": [
            " the head of state and government of the United States",
            " Paris. The Eiffel Tower is in Paris. The Eiffel Tower is a famous landmark",
            " It was a special day, for it was the first of April, and people were putting the finishing touches to their April Fools' Day pranks",
            " to be happy. I believe that happiness is the most important thing in life. I believe that happiness is not just a feeling, but a state of being. I believe that happiness is not something that",
            " blue because of Rayleigh scattering. This is the",
            " of my grandmother, who was a wonderful cook, making a delicious chicken and dumplings soup. She",
        ],
    }[neuron_llm_config["name"]]

    for expected_output, output in zip(expected_outputs, outputs):
        generated_text = output.outputs[0].text
        print(f"Prompt: {output.prompt!r}, Generated text: {generated_text!r}")
        assert expected_output == generated_text
