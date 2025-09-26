import pytest

from optimum.neuron.utils import DTYPE_MAPPER


@pytest.fixture
async def multi_model_vllm_service(vllm_launcher, neuron_llm_config):
    model_name_or_path = neuron_llm_config["neuron_model_path"]
    service_name = neuron_llm_config["name"]
    with vllm_launcher(service_name, model_name_or_path) as vllm_service:
        await vllm_service.health(600)
        yield vllm_service


@pytest.fixture
async def vllm_service_from_local_neuron_model(vllm_launcher, base_neuron_llm_config):
    model_name_or_path = base_neuron_llm_config["neuron_model_path"]
    service_name = base_neuron_llm_config["name"]
    with vllm_launcher(service_name, model_name_or_path) as vllm_service:
        await vllm_service.health(600)
        yield vllm_service


@pytest.fixture
async def vllm_service_from_hub_neuron_model(vllm_launcher, base_neuron_llm_config):
    model_name_or_path = base_neuron_llm_config["neuron_model_id"]
    service_name = base_neuron_llm_config["name"]
    with vllm_launcher(service_name, model_name_or_path) as vllm_service:
        await vllm_service.health(600)
        yield vllm_service


@pytest.fixture
async def vllm_service_from_hub_model(vllm_launcher, base_neuron_llm_config):
    model_name_or_path = base_neuron_llm_config["model_id"]
    service_name = base_neuron_llm_config["name"]
    export_kwargs = base_neuron_llm_config["export_kwargs"]
    batch_size = export_kwargs["batch_size"]
    sequence_length = export_kwargs["sequence_length"]
    tensor_parallel_size = export_kwargs["tensor_parallel_size"]
    dtype = DTYPE_MAPPER.pt(export_kwargs["auto_cast_type"])
    with vllm_launcher(
        service_name,
        model_name_or_path,
        batch_size=batch_size,
        sequence_length=sequence_length,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
    ) as vllm_service:
        await vllm_service.health(600)
        yield vllm_service


async def _test_vllm_service_base(vllm_service, prompt, max_output_tokens):
    greedy_tokens, greedy_text = await vllm_service.client.greedy(prompt, max_output_tokens=max_output_tokens)

    assert greedy_tokens == max_output_tokens


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prompt, max_output_tokens",
    [
        ("What is Deep Learning?", 17),
        ("One of my fondest memory is", 32),
    ],
)
async def test_vllm_service_from_local_neuron_model(vllm_service_from_local_neuron_model, prompt, max_output_tokens):
    await _test_vllm_service_base(vllm_service_from_local_neuron_model, prompt, max_output_tokens)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prompt, max_output_tokens",
    [
        ("What is Deep Learning?", 17),
        ("One of my fondest memory is", 32),
    ],
)
async def test_vllm_service_from_hub_neuron_model(vllm_service_from_hub_neuron_model, prompt, max_output_tokens):
    await _test_vllm_service_base(vllm_service_from_hub_neuron_model, prompt, max_output_tokens)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prompt, max_output_tokens",
    [
        ("What is Deep Learning?", 17),
        ("One of my fondest memory is", 32),
    ],
)
async def test_vllm_service_from_hub_model(vllm_service_from_hub_model, prompt, max_output_tokens):
    await _test_vllm_service_base(vllm_service_from_hub_model, prompt, max_output_tokens)


@pytest.mark.asyncio
async def test_vllm_service_greedy_generation(multi_model_vllm_service):
    service_name = multi_model_vllm_service.client.service_name
    prompt = "What is Deep Learning?"
    max_output_tokens = 17
    # Greedy bounded without input
    greedy_tokens, greedy_text = await multi_model_vllm_service.client.greedy(
        prompt, max_output_tokens=max_output_tokens
    )

    assert greedy_tokens == max_output_tokens

    greedy_expectations = {
        "llama": "Deep learning is a subfield of machine learning that involves the use of neural networks with",
        "qwen2": "Deep Learning is a subset of Machine Learning that involves the use of artificial neural networks to",
        "granite": "Deep Learning is a subset of machine learning, which is itself a branch of artificial",
        "qwen3": "<think>\nOkay, the user is asking about what Deep Learning is. Let me start",
        "phi": " Deep learning is a subset of machine learning, which is itself a subset of artificial intelligence",
        "smollm3": "<think>\nOkay, so I need to explain what Deep Learning is. Let me start",
    }
    # Compare expectations in a case-insensitive way as the results may slightly vary when the enviroment changes
    assert greedy_text.lower() == greedy_expectations[service_name].lower()


@pytest.mark.asyncio
async def test_vllm_service_sampling_parameters(vllm_service_from_local_neuron_model):
    prompt = "What is Deep Learning?"
    max_output_tokens = 17
    # Greedy bounded without input
    greedy_tokens, greedy_text = await vllm_service_from_local_neuron_model.client.greedy(
        prompt, max_output_tokens=max_output_tokens
    )

    assert greedy_tokens == max_output_tokens

    # Sampling
    sample_tokens, sample_text = await vllm_service_from_local_neuron_model.client.sample(
        prompt,
        max_output_tokens=max_output_tokens,
        temperature=0.8,
        top_p=0.9,
    )
    assert sample_tokens == max_output_tokens
    # The response must be different
    assert sample_text != greedy_text

    # Greedy with stop sequence (using one of the words returned from the previous test)
    stop_sequence = greedy_text.split(" ")[-5]
    greedy_tokens_with_stop, greedy_text_with_stop = await vllm_service_from_local_neuron_model.client.greedy(
        "What is Deep Learning?",
        max_output_tokens=max_output_tokens,
        stop=[stop_sequence],
    )
    assert greedy_tokens_with_stop < max_output_tokens
    assert greedy_text.startswith(greedy_text_with_stop + stop_sequence)
