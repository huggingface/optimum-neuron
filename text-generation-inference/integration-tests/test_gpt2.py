import os

import huggingface_hub
import Levenshtein
import pytest


MODEL_ID = "gpt2"
NEURON_MODEL_ID = "aws-neuron/gpt2-neuronx-bs4-seqlen1024"
BATCH_SIZE = 4
SEQUENCE_LENGTH = 1024
NUM_CORES = 2


@pytest.fixture(scope="module", params=["hub-neuron", "hub", "local-neuron"])
def model_name_or_path(request, data_volume):
    if request.param == "hub":
        os.environ["HF_BATCH_SIZE"] = str(BATCH_SIZE)
        os.environ["HF_SEQUENCE_LENGTH"] = str(SEQUENCE_LENGTH)
        os.environ["HF_NUM_CORES"] = str(NUM_CORES)
        yield MODEL_ID
    elif request.param == "hub-neuron":
        yield NEURON_MODEL_ID
    else:
        model_dir = f"gpt2-neuron-{BATCH_SIZE}x{SEQUENCE_LENGTH}x{NUM_CORES}"
        local_path = os.path.join(data_volume, model_dir)
        huggingface_hub.snapshot_download(NEURON_MODEL_ID, local_dir=local_path)
        # Return the path of the model inside the mounted volume
        yield os.path.join("/data", model_dir)


@pytest.fixture(scope="module")
def tgi_service(launcher, model_name_or_path):
    with launcher(model_name_or_path) as tgi_service:
        yield tgi_service


@pytest.fixture(scope="module")
async def tgi_client(tgi_service):
    await tgi_service.health(300)
    return tgi_service.client


@pytest.mark.asyncio
async def test_model_single_request(tgi_client):
    # Greedy bounded without input
    response = await tgi_client.generate(
        "What is Deep Learning?",
        max_new_tokens=17,
        decoder_input_details=True,
    )
    assert response.details.generated_tokens == 17
    assert response.generated_text == "\n\nDeep learning is a new field of research that has been around for a while"

    # Greedy bounded with input
    response = await tgi_client.generate(
        "What is Deep Learning?",
        max_new_tokens=17,
        return_full_text=True,
        decoder_input_details=True,
    )
    assert response.details.generated_tokens == 17
    assert (
        response.generated_text
        == "What is Deep Learning?\n\nDeep learning is a new field of research that has been around for a while"
    )

    # Sampling
    response = await tgi_client.generate(
        "What is Deep Learning?",
        do_sample=True,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        max_new_tokens=1000,
        seed=42,
        decoder_input_details=True,
    )
    assert "A lot of researchers are trying to explain what it is" in response.generated_text


@pytest.mark.asyncio
async def test_model_multiple_requests(tgi_client, generate_load):
    num_requests = 4
    responses = await generate_load(
        tgi_client,
        "What is Deep Learning?",
        max_new_tokens=17,
        n=num_requests,
    )

    assert len(responses) == 4
    expected = "\n\nDeep learning is a new field of research that has been around for a while"
    for r in responses:
        assert r.details.generated_tokens == 17
        # Compute the similarity with the expectation using the levenshtein distance
        # We should not have more than two substitutions or additions
        assert Levenshtein.distance(r.generated_text, expected) < 3
