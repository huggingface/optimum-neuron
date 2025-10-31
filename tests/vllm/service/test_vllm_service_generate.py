from tempfile import TemporaryDirectory

import pytest
from huggingface_hub import get_token, snapshot_download


# Do not collect tests from this file if vllm is not installed
pytest.importorskip("vllm")


@pytest.fixture
async def multi_model_vllm_service(vllm_launcher, neuron_llm_config):
    model_name_or_path = neuron_llm_config["neuron_model_path"]
    service_name = neuron_llm_config["name"]
    with vllm_launcher(service_name, model_name_or_path) as vllm_service:
        await vllm_service.health(600)
        yield vllm_service


@pytest.mark.asyncio
@pytest.fixture(params=["local_neuron", "hub_neuron", "hub_explicit", "hub_implicit", "local_implicit"])
async def vllm_service_from_model(request, vllm_launcher, base_neuron_llm_config):
    service_name = base_neuron_llm_config["name"]
    if request.param == "hub_explicit":
        model_name_or_path = base_neuron_llm_config["model_id"]
        export_kwargs = base_neuron_llm_config["export_kwargs"]
        batch_size = export_kwargs["batch_size"]
        sequence_length = export_kwargs["sequence_length"]
        tensor_parallel_size = export_kwargs["tensor_parallel_size"]
        with vllm_launcher(
            service_name,
            model_name_or_path,
            batch_size=batch_size,
            sequence_length=sequence_length,
            tensor_parallel_size=tensor_parallel_size,
        ) as vllm_service:
            await vllm_service.health(600)
            yield vllm_service
    elif request.param == "local_implicit":
        served_model_name = base_neuron_llm_config["model_id"]
        with TemporaryDirectory() as local_model_path:
            # Manually download weights
            token = get_token()
            snapshot_download(
                served_model_name,
                local_dir=local_model_path,
                token=token,
                allow_patterns=[
                    "model*.safetensors",
                    "*.json",
                ],
            )
            model_name_or_path = local_model_path
            with vllm_launcher(service_name, model_name_or_path, served_model_name) as vllm_service:
                await vllm_service.health(600)
                yield vllm_service
    else:
        if request.param == "local_neuron":
            model_name_or_path = base_neuron_llm_config["neuron_model_path"]
        elif request.param == "hub_neuron":
            model_name_or_path = base_neuron_llm_config["neuron_model_id"]
        elif request.param == "hub_implicit":
            model_name_or_path = base_neuron_llm_config["model_id"]
        else:
            raise ValueError(f"Unknown request.param: {request.param}")
        with vllm_launcher(service_name, model_name_or_path) as vllm_service:
            await vllm_service.health(600)
            yield vllm_service


@pytest.mark.asyncio
async def test_vllm_service_from_model(vllm_service_from_model):
    prompt = "What is Deep Learning?"
    max_output_tokens = 17
    greedy_tokens, greedy_text = await vllm_service_from_model.client.greedy(
        prompt, max_output_tokens=max_output_tokens
    )
    print(f"Greedy output: {greedy_text}")
    assert greedy_tokens == max_output_tokens


@pytest.mark.asyncio
async def test_vllm_service_greedy_generation(multi_model_vllm_service):
    prompt = "What is Deep Learning?"
    max_output_tokens = 17
    # Greedy bounded without input
    greedy_tokens, _ = await multi_model_vllm_service.client.greedy(prompt, max_output_tokens=max_output_tokens)

    assert greedy_tokens == max_output_tokens
