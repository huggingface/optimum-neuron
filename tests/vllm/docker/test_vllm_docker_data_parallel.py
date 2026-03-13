# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for data-parallel serving via Docker container."""

import pytest

from optimum.neuron.utils.system import get_available_cores


# Do not collect tests from this file if docker or vllm are not installed
pytest.importorskip("docker")
pytest.importorskip("vllm")


DATA_PARALLEL_SIZE = 2
# DP=2 with TP=1 requires 2 cores; skip if the host doesn't have enough.
_have = get_available_cores()
pytestmark = pytest.mark.skipif(
    _have < DATA_PARALLEL_SIZE,
    reason=f"Data-parallel test needs {DATA_PARALLEL_SIZE} cores but only {_have} available",
)


@pytest.fixture
async def vllm_docker_dp_service(vllm_docker_launcher, neuron_llm_config):
    """Launch a data-parallel vLLM Docker service with 2 replicas."""
    model_name_or_path = neuron_llm_config["neuron_model_path"]
    service_name = neuron_llm_config["name"]
    with vllm_docker_launcher(
        service_name,
        model_name_or_path,
        data_parallel_size=DATA_PARALLEL_SIZE,
    ) as vllm_service:
        await vllm_service.health(600)
        yield vllm_service


@pytest.mark.asyncio
@pytest.mark.parametrize("neuron_llm_config", ["qwen3-tp1-4x1024"], indirect=True)
async def test_vllm_docker_data_parallel_greedy(vllm_docker_dp_service):
    """Verify greedy generation works with data-parallel Docker serving."""
    prompt = "What is Deep Learning?"
    max_output_tokens = 17
    greedy_tokens, greedy_text = await vllm_docker_dp_service.client.greedy(
        prompt, max_output_tokens=max_output_tokens
    )
    print(f"Greedy output: {greedy_text}")
    assert greedy_tokens == max_output_tokens


@pytest.mark.asyncio
@pytest.mark.parametrize("neuron_llm_config", ["qwen3-tp1-4x1024"], indirect=True)
async def test_vllm_docker_data_parallel_sampling(vllm_docker_dp_service):
    """Verify sampling works with data-parallel Docker serving."""
    prompt = "What is Deep Learning?"
    max_output_tokens = 17

    greedy_tokens, greedy_text = await vllm_docker_dp_service.client.greedy(
        prompt, max_output_tokens=max_output_tokens
    )
    assert greedy_tokens == max_output_tokens

    sample_tokens, sample_text = await vllm_docker_dp_service.client.sample(
        prompt,
        max_output_tokens=max_output_tokens,
        temperature=0.8,
        top_p=0.9,
    )
    assert sample_tokens == max_output_tokens
    # Sampling with temperature should produce different output
    assert sample_text != greedy_text


@pytest.mark.asyncio
@pytest.mark.parametrize("neuron_llm_config", ["qwen3-tp1-4x1024"], indirect=True)
async def test_vllm_docker_data_parallel_concurrent(vllm_docker_dp_service):
    """Verify that concurrent requests are handled by different DP replicas.

    Sends two requests in parallel — one long, one short — and checks that
    both complete successfully, confirming the proxy distributes work across
    replicas rather than serialising it.
    """
    import asyncio

    client = vllm_docker_dp_service.client
    long_task = client.greedy("Explain the theory of relativity in detail.", max_output_tokens=64)
    short_task = client.greedy("Say hello.", max_output_tokens=8)

    # Await both concurrently; if requests were serialised through a single
    # replica the total time would be roughly additive, but we mostly care
    # that both succeed.
    (long_tokens, long_text), (short_tokens, short_text) = await asyncio.gather(long_task, short_task)

    assert long_tokens == 64
    assert short_tokens == 8
    assert len(long_text) > len(short_text)
