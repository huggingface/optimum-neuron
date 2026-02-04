import pytest
import torch
import torch.nn.functional as F


# Do not collect tests from this file if vllm is not installed
pytest.importorskip("vllm")


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


@pytest.fixture
async def any_embedding_model_vllm_service(vllm_launcher, neuron_llm_config):
    model_name_or_path = neuron_llm_config["neuron_model_path"]
    service_name = neuron_llm_config["name"]
    with vllm_launcher(service_name, model_name_or_path, extra_args=["--task", "embed"]) as vllm_service:
        await vllm_service.health(600)
        yield vllm_service


@pytest.mark.asyncio
@pytest.mark.parametrize("neuron_llm_config", ["qwen3-embedding-4x8192"], indirect=True)
async def test_vllm_service_embedding(any_embedding_model_vllm_service, neuron_llm_config):
    """Test vLLM service embedding generation."""

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

    # Get embeddings from the vLLM service via OpenAI API
    client = any_embedding_model_vllm_service.client

    # Create embeddings for each input text
    response = await client.embeddings.create(
        input=input_texts,
        model=any_embedding_model_vllm_service.client.model_name,
    )

    # Extract embeddings and convert to tensor
    embeddings_list = [embedding.embedding for embedding in response.data]
    embeddings = torch.tensor(embeddings_list, dtype=torch.bfloat16)

    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = embeddings[:2] @ embeddings[2:].T

    print(f"Similarity scores: {scores.tolist()}")

    expected_scores = [[0.76171875, 0.140625], [0.1357421875, 0.6015625]]

    assert torch.allclose(scores, torch.tensor(expected_scores, dtype=torch.bfloat16), atol=1e-2)
