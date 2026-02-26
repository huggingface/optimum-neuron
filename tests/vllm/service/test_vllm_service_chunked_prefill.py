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

Design note: Neuron cores are exclusive — only one vLLM service can hold them at a
time.  The two services (standard and chunked) are therefore started and stopped
*sequentially* in a single ``service_outputs`` fixture: the standard service runs
first and its output is recorded, then it is shut down and the chunked-prefill
service is started.  All tests compare from the pre-computed results.
"""

from typing import Any

import pytest


# Do not collect tests from this file if vllm is not installed
pytest.importorskip("vllm")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PROMPT = "What is deep learning?"
MAX_OUTPUT_TOKENS = 20


@pytest.fixture(scope="module")
def service_outputs(event_loop, vllm_launcher, chunked_prefill_llm_config: dict[str, Any]):
    """Run std and chunk services *sequentially* and return their greedy outputs.

    Neuron cores are exclusive: only one service can hold them at a time.  The
    fixture starts the standard service, records its response, shuts it down,
    then starts the chunked-prefill service and records its response.
    """

    async def _greedy(service) -> tuple[int, str]:
        return await service.client.greedy(PROMPT, max_output_tokens=MAX_OUTPUT_TOKENS)

    std_path = chunked_prefill_llm_config["std_neuron_model_path"]
    chunk_path = chunked_prefill_llm_config["chunk_neuron_model_path"]

    # --- Standard (context-encoding) service ---
    with vllm_launcher("chunked-prefill-std", std_path) as std_service:
        event_loop.run_until_complete(std_service.health(600))
        std_n_tokens, std_text = event_loop.run_until_complete(_greedy(std_service))
    # context manager exit shuts down the process → cores released

    # --- Chunked-prefill service ---
    with vllm_launcher("chunked-prefill-chunk", chunk_path) as chunk_service:
        event_loop.run_until_complete(chunk_service.health(600))
        chunk_n_tokens, chunk_text = event_loop.run_until_complete(_greedy(chunk_service))

    return {
        "std_n_tokens": std_n_tokens,
        "std_text": std_text,
        "chunk_n_tokens": chunk_n_tokens,
        "chunk_text": chunk_text,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_vllm_service_chunked_prefill_generation(service_outputs: dict):
    """Chunked-prefill service must respond with the expected number of tokens."""
    n_tokens = service_outputs["chunk_n_tokens"]
    text = service_outputs["chunk_text"]
    assert n_tokens == MAX_OUTPUT_TOKENS, f"Expected {MAX_OUTPUT_TOKENS} tokens, got {n_tokens}: {text!r}"


def test_vllm_service_std_vs_chunked_prefill(service_outputs: dict):
    """Greedy output must closely match between standard and chunked-prefill services.

    The two prefill paths use different attention implementations (NKI kernel
    for standard context encoding, XLA compute_for_token_gen for chunked),
    which can accumulate small numerical differences over multiple
    autoregressive steps.  We require that the generated texts share a
    common prefix of at least ``MIN_PREFIX_WORDS`` words.
    """
    MIN_PREFIX_WORDS = 10  # out of ~20 generated tokens
    std_text = service_outputs["std_text"]
    chunk_text = service_outputs["chunk_text"]

    std_words = std_text.split()
    chunk_words = chunk_text.split()
    common = 0
    for s, c in zip(std_words, chunk_words):
        if s != c:
            break
        common += 1
    assert common >= MIN_PREFIX_WORDS, (
        f"Chunked-prefill service diverged too early from standard service "
        f"(first {common} words match, need {MIN_PREFIX_WORDS}).\n"
        f"  std  : {std_text!r}\n"
        f"  chunk: {chunk_text!r}"
    )
