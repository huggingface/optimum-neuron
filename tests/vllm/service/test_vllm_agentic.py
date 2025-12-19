import datetime

import pytest
from smolagents import CodeAgent, OpenAIModel


# Do not collect tests from this file if vllm is not installed
pytest.importorskip("vllm")


# Note: we use SmolLM3 as a test model because it is a small model that is easy to test and it supports tool calling.
@pytest.fixture
async def smollm3_model_agentic_vllm(vllm_launcher, base_neuron_llm_config, request):
    model_name_or_path = base_neuron_llm_config["neuron_model_path"]
    service_name = base_neuron_llm_config["name"]
    extra_args = ["--enable-auto-tool-choice", "--tool-call-parser", "hermes"]
    with vllm_launcher(service_name, model_name_or_path, extra_args=extra_args) as vllm_service:
        await vllm_service.health(600)
        yield vllm_service


@pytest.mark.asyncio
@pytest.mark.parametrize("base_neuron_llm_config", ["smollm3"], indirect=True)
async def test_vllm_agentic_test(smollm3_model_agentic_vllm):
    model_name = smollm3_model_agentic_vllm.client.model_name
    base_url = str(smollm3_model_agentic_vllm.client.base_url)

    model = OpenAIModel(
        model_id=model_name,
        api_base=base_url,
        api_key="dummy",
        max_tokens=1900,
        temperature=0.0,  # Setting temperature to 0.0 to make the test deterministic
        seed=42,
    )
    # The agent can run python code, so it can figure out the day of the week without other tools.
    agent = CodeAgent(model=model, tools=[])
    response = agent.run("What's today's day of the week?", max_steps=1)
    day_of_week = datetime.datetime.today().strftime("%A")
    assert day_of_week in response
