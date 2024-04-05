import os

import Levenshtein
import pytest

# These tests will often break as it relies on many factors like the optimum version, the neuronx-cc version,
# and on what is synced in the cache for these specific versions...

MODELS = ["openai-community/gpt2", "aws-neuron/gpt2-neuronx-bs4-seqlen1024"]

# Not sure this is relevant to check the expected output that will often change and thus break tests.
# Not sure we will even catch anything weird in the quality of the generated text anyway, at least not this way.
# We will see with usage but we should probably just check the tgi service returns OK to requests
EXPECTED = {
    "openai-community/gpt2": "There's a new book by John C. Snider",
    "aws-neuron/gpt2-neuronx-bs4-seqlen1024": "The purpose of the current post is to introduce the concepts",
}


@pytest.fixture(scope="module", params=MODELS)
def model_and_expected_output(request, data_volume):
    # the tgi_env.py script will take care of setting these
    for var in [
        "MAX_BATCH_SIZE",
        "MAX_INPUT_LENGTH",
        "MAX_TOTAL_TOKEN",
        "HF_BATCH_SIZE",
        "HF_NUM_CORES",
        "HF_SEQUENCE_LENGTH",
        "HF_AUTO_CAST_TYPE",
    ]:
        if var in os.environ:
            del os.environ[var]
    yield request.param, EXPECTED[request.param]


@pytest.fixture(scope="module")
def tgi_service(launcher, model_and_expected_output):
    model, _ = model_and_expected_output
    with launcher(model) as tgi_service:
        yield tgi_service


@pytest.fixture(scope="module")
async def tgi_client(tgi_service):
    await tgi_service.health(300)
    return tgi_service.client


@pytest.mark.asyncio
async def test_model_single_request(tgi_client, model_and_expected_output):
    _, expected = model_and_expected_output
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
    assert expected in response.generated_text


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
