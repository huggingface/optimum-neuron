import pytest
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

from optimum.neuron import NeuronModelForEmbedding


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


def compute_similarity(tokenized_inputs, model):
    tokenized_inputs = tokenized_inputs.to(model.device)
    outputs = model(**tokenized_inputs)
    if hasattr(outputs, "last_hidden_state"):
        outputs = outputs.last_hidden_state
    if outputs.shape[1] == 1:
        embeddings = outputs[:, 0, :]
    else:
        embeddings = last_token_pool(outputs, tokenized_inputs["attention_mask"])

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = embeddings[:2] @ embeddings[2:].T
    return scores


@pytest.mark.parametrize(
    "neuron_llm_config",
    [
        "qwen3-embedding-4x8192",
        "qwen3-embedding-6x8192",
    ],
    indirect=True,
)
def test_decoder_similarity(neuron_llm_config):
    model_id = neuron_llm_config["model_id"]
    neuron_model_path = neuron_llm_config["neuron_model_path"]
    sequence_length = neuron_llm_config["export_kwargs"]["sequence_length"]
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

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right")

    tokenized_inputs = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=sequence_length,
        return_tensors="pt",
    )

    # We just evaluate the CPU results as a reference, but the neuron results
    # are not that close so we don't do an explicit comparison
    model = AutoModel.from_pretrained(model_id)
    cpu_scores = compute_similarity(tokenized_inputs, model)

    neuron_model = NeuronModelForEmbedding.from_pretrained(neuron_model_path)
    neuron_scores = compute_similarity(tokenized_inputs, neuron_model)
    cpu_scores = cpu_scores.to(neuron_scores.dtype)
    print("CPU scores: ", cpu_scores.tolist())
    # [[0.7645566463470459, 0.14142508804798126], [0.13549773395061493, 0.5999549627304077]]
    print("Neuron scores: ", neuron_scores.tolist())
    assert torch.allclose(neuron_scores, cpu_scores, atol=1e-2)
