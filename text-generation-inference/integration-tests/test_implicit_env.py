import os

import pytest
from text_generation.errors import ValidationError


# These tests will often break as it relies on many factors like the optimum version, the neuronx-cc version,
# and on what is synced in the cache for these specific versions...

MODELS = ["openai-community/gpt2", "aws-neuron/gpt2-neuronx-bs4-seqlen1024"]


@pytest.fixture(scope="module", params=MODELS)
def get_model_and_set_env(request):
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
    yield request.param


@pytest.fixture(scope="module")
def tgi_service(launcher, get_model_and_set_env):
    with launcher(get_model_and_set_env) as tgi_service:
        yield tgi_service


@pytest.fixture(scope="module")
async def tgi_client(tgi_service):
    await tgi_service.health(300)
    return tgi_service.client


@pytest.mark.asyncio
async def test_model_single_request(tgi_client):

    # Just verify that the generation works, and nothing is raised, with several set of params

    # No params
    await tgi_client.generate(
        "What is Deep Learning?",
    )

    response = await tgi_client.generate(
        "How to cook beans ?",
        max_new_tokens=17,
        decoder_input_details=True,
    )
    assert response.details.generated_tokens == 17

    # check error
    try:
        await tgi_client.generate("What is Deep Learning?", max_new_tokens=170000)
    except ValidationError:
        pass
    else:
        raise AssertionError(
            "The previous text generation request should have failed, "
            "because too many tokens were requested, it succeeded"
        )

    # Sampling
    await tgi_client.generate(
        "What is Deep Learning?",
        do_sample=True,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        max_new_tokens=1000,
        seed=42,
        decoder_input_details=True,
    )
