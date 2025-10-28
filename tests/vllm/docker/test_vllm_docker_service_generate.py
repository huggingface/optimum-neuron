from tempfile import TemporaryDirectory

import pytest
from huggingface_hub import get_token, snapshot_download


# Do not collect tests from this file if docker or vllm are not installed
pytest.importorskip("docker")
pytest.importorskip("vllm")


@pytest.mark.asyncio
@pytest.fixture(params=["local_neuron", "hub_neuron", "hub_explicit", "hub_implicit", "local_implicit"])
async def vllm_docker_service_from_model(request, vllm_docker_launcher, base_neuron_llm_config):
    service_name = base_neuron_llm_config["name"]
    if request.param == "hub_explicit":
        model_name_or_path = base_neuron_llm_config["model_id"]
        export_kwargs = base_neuron_llm_config["export_kwargs"]
        batch_size = export_kwargs["batch_size"]
        sequence_length = export_kwargs["sequence_length"]
        tensor_parallel_size = export_kwargs["tensor_parallel_size"]
        with vllm_docker_launcher(
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
            with vllm_docker_launcher(service_name, model_name_or_path, served_model_name) as vllm_service:
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
        with vllm_docker_launcher(service_name, model_name_or_path) as vllm_service:
            await vllm_service.health(600)
            yield vllm_service


@pytest.mark.asyncio
async def test_vllm_docker_service_from_model(vllm_docker_service_from_model):
    prompt = "What is the colour of the sky ?"
    max_output_tokens = 24
    greedy_tokens, greedy_text = await vllm_docker_service_from_model.client.greedy(
        prompt, max_output_tokens=max_output_tokens
    )
    print(f"Greedy output: {greedy_text}")
    assert greedy_tokens == max_output_tokens


@pytest.mark.asyncio
async def test_vllm_docker_service_sampling_parameters(base_neuron_llm_config, vllm_docker_launcher):
    prompt = "What is Deep Learning?"
    max_output_tokens = 17
    with vllm_docker_launcher(
        base_neuron_llm_config["name"],
        base_neuron_llm_config["neuron_model_path"],
    ) as vllm_docker_service_from_local_neuron_model:
        await vllm_docker_service_from_local_neuron_model.health(600)

        # Greedy bounded without input
        greedy_tokens, greedy_text = await vllm_docker_service_from_local_neuron_model.client.greedy(
            prompt, max_output_tokens=max_output_tokens
        )

        assert greedy_tokens == max_output_tokens

        # Sampling
        sample_tokens, sample_text = await vllm_docker_service_from_local_neuron_model.client.sample(
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
        (
            greedy_tokens_with_stop,
            greedy_text_with_stop,
        ) = await vllm_docker_service_from_local_neuron_model.client.greedy(
            prompt,
            max_output_tokens=max_output_tokens,
            stop=[stop_sequence],
        )
        assert greedy_tokens_with_stop < max_output_tokens
        assert greedy_text.startswith(greedy_text_with_stop + stop_sequence)
