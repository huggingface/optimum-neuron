# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Service-level tests verifying that a vLLM server backed by a chunked-prefill
model starts up correctly and produces valid greedy generation output.
"""

from typing import Any

import pytest


# Do not collect tests from this file if vllm is not installed
pytest.importorskip("vllm")


PROMPT = "What is deep learning?"
MAX_OUTPUT_TOKENS = 20


@pytest.mark.parametrize("neuron_llm_config", ["llama-1x8192"], indirect=True)
def test_vllm_service_chunked_prefill_generation(event_loop, vllm_launcher, neuron_llm_config: dict[str, Any]):
    """Chunked-prefill service must start and respond with the expected number of tokens."""

    async def _greedy(service) -> tuple[int, str]:
        return await service.client.greedy(PROMPT, max_output_tokens=MAX_OUTPUT_TOKENS)

    with vllm_launcher("chunked-prefill", neuron_llm_config["neuron_model_path"]) as service:
        event_loop.run_until_complete(service.health(600))
        n_tokens, text = event_loop.run_until_complete(_greedy(service))

    assert n_tokens == MAX_OUTPUT_TOKENS, f"Expected {MAX_OUTPUT_TOKENS} tokens, got {n_tokens}: {text!r}"
    assert len(text) > 0, "Generated text is empty"
