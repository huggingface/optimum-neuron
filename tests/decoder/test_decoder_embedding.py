import pytest
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

from optimum.neuron.models.inference.auto_models import Qwen3NeuronModelForEmbedding


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
    embeddings = last_token_pool(outputs, tokenized_inputs["attention_mask"])

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = embeddings[:2] @ embeddings[2:].T
    return scores


@pytest.mark.parametrize(
    "batch_size, expected_scores",
    [
        (4, [[0.7265625, 0.203125], [0.2578125, 0.46484375]]),
        (6, [[0.73046875, 0.2021484375], [0.2578125, 0.45703125]]),
    ],
    ids=["without batch padding", "with batch padding"],
)
def test_decoder_similarity(batch_size, expected_scores):
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

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", padding_side="left")

    max_length = 8192

    tokenized_inputs = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # We just evaluate the CPU results as a reference, but the neuron results
    # are not that close so we don't do an explicit comparison
    model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
    cpu_scores = compute_similarity(tokenized_inputs, model)
    print(cpu_scores.tolist())

    neuron_config = Qwen3NeuronModelForEmbedding.get_neuron_config(
        "Qwen/Qwen3-Embedding-0.6B", batch_size=batch_size, sequence_length=1024
    )
    neuron_model = Qwen3NeuronModelForEmbedding.export(
        model_id="Qwen/Qwen3-Embedding-0.6B", neuron_config=neuron_config, load_weights=True
    )
    neuron_scores = compute_similarity(tokenized_inputs, neuron_model)
    print(neuron_scores.tolist())
    assert torch.allclose(neuron_scores, torch.tensor(expected_scores, dtype=torch.bfloat16), atol=1e-2)
