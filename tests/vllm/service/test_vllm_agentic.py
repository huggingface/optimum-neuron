import datetime
import json
from typing import Any, Dict, List

import pytest
from transformers.utils import get_json_schema


# Do not collect tests from this file if vllm is not installed
pytest.importorskip("vllm")


@pytest.fixture
async def neuron_model_agentic_vllm(vllm_launcher, neuron_llm_config, request):
    model_name_or_path = neuron_llm_config["neuron_model_path"]
    service_name = neuron_llm_config["name"]
    extra_args = ["--enable-auto-tool-choice", "--tool-call-parser", "hermes"]
    with vllm_launcher(service_name, model_name_or_path, extra_args=extra_args) as vllm_service:
        await vllm_service.health(600)
        yield vllm_service


async def greedy_with_tools(
    client,
    prompt: str,
    max_output_tokens: int,
    top_p: float | None = None,
    stop: List[str] | None = None,
    tools: List[Dict[str, Any]] | None = None,
):
    temperature = 0  # greedy sampling
    response = await client.chat.completions.create(
        model=client.model_name,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        tools=tools,
    )
    generated_tokens = response.usage.completion_tokens
    generated_text = response.choices[0].message.content
    tool_calls = response.choices[0].message.tool_calls
    return generated_tokens, generated_text, tool_calls


# Note: we use Qwen3-0.6B as a test model because it is a small model that is easy to test and it supports tool calling.
@pytest.mark.asyncio
@pytest.mark.parametrize("neuron_llm_config", ["qwen3-1x8192"], indirect=True)
async def test_vllm_agentic_test(neuron_model_agentic_vllm):
    today_day_of_week = datetime.datetime.today().strftime("%A")
    function_called = False

    def get_day_of_week(day: str) -> str:
        """
        Return day of the week for the given day.

        Args:
            day: The day, it should be "today" or "tomorrow".

        Returns:
            str: The day of the week.
        """
        nonlocal function_called
        function_called = True
        if day == "today":
            return today_day_of_week
        elif day == "tomorrow":
            return (datetime.datetime.today() + datetime.timedelta(days=1)).strftime("%A")
        else:
            raise ValueError(f"Invalid day: {day}")

    tools = [get_day_of_week]
    tools_dict = {tool.__name__: tool for tool in tools}
    tokens, _, tool_calls = await greedy_with_tools(
        neuron_model_agentic_vllm.client,
        prompt="What's today's day of the week?",
        max_output_tokens=2000,
        tools=[get_json_schema(tool) for tool in tools],
    )
    tool_call = tool_calls[0]
    tool_call_name = tool_call.function.name
    tool_call_arguments = tool_call.function.arguments
    # Parse the arguments
    arguments = json.loads(tool_call_arguments)
    assert tool_call_name in tools_dict
    result = tools_dict[tool_call_name](**arguments)
    # validate it was called with the expected arguments, returning the expected result
    assert result == today_day_of_week
    assert function_called
    # At this point, we should reformat a new message containing the original prompt and the result of the function
    # call, but for this test we can just stop here.
