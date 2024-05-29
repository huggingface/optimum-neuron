import Levenshtein
import pytest


@pytest.fixture
async def tgi_service(launcher, neuron_model_config):
    model_name_or_path = neuron_model_config["neuron_model_path"]
    service_name = neuron_model_config["name"]
    with launcher(service_name, model_name_or_path) as tgi_service:
        await tgi_service.health(600)
        yield tgi_service


@pytest.mark.asyncio
async def test_model_single_request(tgi_service):
    service_name = tgi_service.client.service_name
    prompt = "What is Deep Learning?"
    # Greedy bounded without input
    response = await tgi_service.client.text_generation(
        prompt, max_new_tokens=17, details=True, decoder_input_details=True
    )
    assert response.details.generated_tokens == 17
    greedy_expectations = {
        "gpt2": "\n\nDeep learning is a new field of research that has been around for a while",
        "llama": "\n\nDeep learning is a subset of machine learning that uses artificial neural networks to model",
        "mistral": "\nWhat is Deep Learning?\nDeep Learning is a type of machine learning that",
    }
    assert response.generated_text == greedy_expectations[service_name]

    # Greedy bounded with input
    response = await tgi_service.client.text_generation(
        "What is Deep Learning?", max_new_tokens=17, return_full_text=True, details=True, decoder_input_details=True
    )
    assert response.details.generated_tokens == 17
    assert response.generated_text == prompt + greedy_expectations[service_name]

    # Sampling
    response = await tgi_service.client.text_generation(
        "What is Deep Learning?",
        do_sample=True,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        max_new_tokens=128,
        seed=42,
        details=True,
        decoder_input_details=True,
    )
    sample_expectations = {
        "gpt2": "A lot of researchers have tried to make a broad, intuitive definition of Deep Learning",
        "llama": "Deep Learning is a technique for training artificial neural networks",
        "mistral": "Why is deep learning important?",
    }
    assert sample_expectations[service_name] in response.generated_text


@pytest.mark.asyncio
async def test_model_multiple_requests(tgi_service, generate_load):
    num_requests = 4
    responses = await generate_load(
        tgi_service.client,
        "What is Deep Learning?",
        max_new_tokens=17,
        n=num_requests,
    )

    assert len(responses) == 4
    expectations = {
        "gpt2": "\n\nDeep learning is a new field of research that has been around for a while",
        "llama": "\n\nDeep learning is a subset of machine learning that uses artificial neural networks to model",
        "mistral": "\nWhat is Deep Learning?\nDeep Learning is a type of machine learning that",
    }
    expected = expectations[tgi_service.client.service_name]
    for r in responses:
        assert r.details.generated_tokens == 17
        # Compute the similarity with the expectation using the levenshtein distance
        # We should not have more than two substitutions or additions
        assert Levenshtein.distance(r.generated_text, expected) < 3
