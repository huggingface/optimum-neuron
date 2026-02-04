# ruff: noqa: E402 ignore imports not at top-level
from typing import Any

import pytest
import torch
import torch.nn.functional as F


# Do not collect tests from this file if vllm is not installed
pytest.importorskip("vllm")

from vllm import LLM


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


def compute_similarity(embeddings):
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = embeddings[:2] @ embeddings[2:].T
    return scores


@pytest.mark.parametrize(
    "neuron_llm_config",
    [
        "qwen3-embedding-4x8192",
    ],
    indirect=True,
)
def test_vllm_compute_similarity(neuron_llm_config: dict[str, Any]):
    neuron_model_path = neuron_llm_config["neuron_model_path"]

    # Each query must come with a one-sentence instruction that describes the task
    task = "Given a web search query, retrieve relevant passages that answer the query"

    queries = [
        get_detailed_instruct(task, "What is the capital of China?"),
        get_detailed_instruct(task, "Explain gravity"),
    ]
    # No need to add instruction for retrieval documents
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]
    input_texts = queries + documents

    # Get embeddings on Neuron from vLLM
    llm = LLM(model=neuron_model_path, task="embed")
    outputs = llm.embed(input_texts)
    embeddings_list = [output.outputs.embedding for output in outputs]
    embeddings = torch.tensor(embeddings_list, dtype=torch.bfloat16)
    neuron_scores = compute_similarity(embeddings)

    # We can't evaluate the CPU results as the vLLM neuron backend takes precedence
    # Attempts at evaluating the CPU results with transformers lead to test hangs, probably
    # with conflicts inside the chain of imports
    cpu_scores = torch.tensor([[0.76171875, 0.140625], [0.1357421875, 0.6015625]]).to(neuron_scores.dtype)

    print(f"CPU scores: {cpu_scores}")
    print(f"Neuron scores: {neuron_scores.tolist()}")

    assert torch.allclose(cpu_scores, neuron_scores, atol=1e-2)
