# ruff: noqa: E402 ignore imports not at top-level
# coding=utf-8
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Engine-level tests verifying that a chunked-prefill model generates correct
output through the vLLM engine.

Each test exercises a distinct code path relative to the model's chunk_size:

  short  —  chunk_size // 2 tokens  →  padding path in _execute_chunked_prefill
  long   —  chunk_size * 2 tokens   →  KV accumulation across two complete chunks
  batch  —  [short, long] in one call →  verifies consistency with individual runs
"""

from typing import Any

import pytest


# Do not collect tests from this file if vllm is not installed
pytest.importorskip("vllm")


from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MAX_NEW_TOKENS = 20


def _make_prompt(n_tokens: int) -> dict:
    """Build a vLLM input dict with exactly n_tokens prompt token IDs."""
    return {"prompt_token_ids": [1] * n_tokens}


def _generate(llm: LLM, prompts: list[dict], max_new_tokens: int = MAX_NEW_TOKENS) -> list[tuple[int, ...]]:
    """Run greedy generation and return a list of token-ID tuples (one per prompt)."""
    params = SamplingParams(top_k=1, max_tokens=max_new_tokens)
    outputs = llm.generate(prompts, params)
    return [tuple(out.outputs[0].token_ids) for out in outputs]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def chunked_prefill_data(neuron_llm_config: dict[str, Any]):
    """Load a chunked-prefill model and collect generation results.

    All tests in this module share the pre-computed data — the model is loaded
    once, all prompts are run, and the results are stored for comparison.
    """
    chunk_size = 1024
    short_prompt = _make_prompt(chunk_size // 2)
    long_prompt = _make_prompt(chunk_size * 2)
    batch_prompts = [short_prompt, long_prompt]

    batch_size = neuron_llm_config["export_kwargs"]["batch_size"]
    llm = LLM(model=neuron_llm_config["neuron_model_path"], max_num_seqs=batch_size)
    short = _generate(llm, [short_prompt])[0]
    long_ = _generate(llm, [long_prompt])[0]
    batch = _generate(llm, batch_prompts)
    del llm

    return {
        "short": short,
        "long": long_,
        "batch": batch,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("neuron_llm_config", ["llama-1x8192"], indirect=True)
def test_chunked_prefill_engine_short_prompt(neuron_llm_config, chunked_prefill_data: dict):
    """Short prompt (< chunk_size) generates the expected number of tokens."""
    ids = chunked_prefill_data["short"]
    assert len(ids) == MAX_NEW_TOKENS, f"Expected {MAX_NEW_TOKENS} tokens, got {len(ids)}: {ids}"


@pytest.mark.parametrize("neuron_llm_config", ["llama-1x8192"], indirect=True)
def test_chunked_prefill_engine_long_prompt(neuron_llm_config, chunked_prefill_data: dict):
    """Long prompt (2 × chunk_size) generates the expected number of tokens."""
    ids = chunked_prefill_data["long"]
    assert len(ids) == MAX_NEW_TOKENS, f"Expected {MAX_NEW_TOKENS} tokens, got {len(ids)}: {ids}"


@pytest.mark.parametrize("neuron_llm_config", ["llama-1x8192"], indirect=True)
def test_chunked_prefill_engine_batch(neuron_llm_config, chunked_prefill_data: dict):
    """Batched chunked prefill must produce consistent results.

    Running a short and long prompt together in one batch must produce the same
    output as running them individually.  This verifies the vLLM scheduler and
    runner correctly handle mixed-length prompts during chunked prefill.
    """
    batch = chunked_prefill_data["batch"]
    short = chunked_prefill_data["short"]
    long_ = chunked_prefill_data["long"]
    assert batch[0] == short, (
        f"Batch[0] (short) doesn't match individual run.\n  batch: {batch[0]}\n  individual: {short}"
    )
    assert batch[1] == long_, (
        f"Batch[1] (long) doesn't match individual run.\n  batch: {batch[1]}\n  individual: {long_}"
    )
