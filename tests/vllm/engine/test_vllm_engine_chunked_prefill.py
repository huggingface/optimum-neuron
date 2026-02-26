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
"""Engine-level tests verifying that chunked prefill produces the same vLLM outputs
as standard context encoding.

Prompt IDs are constructed from known token IDs (all-1) so that exact lengths are
controlled without a tokenizer round-trip.  Each test case chooses a length relative
to CHUNKED_PREFILL_CHUNK_SIZE (512) to exercise distinct code paths:

  short  —  chunk_size // 2 tokens  →  padding path in _execute_chunked_prefill
  long   —  chunk_size * 2 tokens   →  KV accumulation across two complete chunks
  batch  —  [short, long] in one call →  "exhaust+repeat" path for the shorter
             sequence during the second chunk round

Design note: Neuron hardware only has 2 cores on this instance, and each LLM engine
holds those cores exclusively.  The two models (std and chunk) are therefore loaded
and exercised sequentially in a single ``comparison_data`` fixture: std model runs
first, results are stored, it is deleted (releasing the cores), then the chunk model
runs and its results are stored for comparison.
"""

import itertools
from typing import Any

import pytest


# Do not collect tests from this file if vllm is not installed
pytest.importorskip("vllm")


from vllm import LLM, SamplingParams


# Must match CHUNKED_PREFILL_CHUNK_SIZE in tests/fixtures/llm/export_models.py
CHUNKED_PREFILL_CHUNK_SIZE = 512


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prompt(n_tokens: int) -> dict:
    """Build a vLLM input dict with exactly n_tokens prompt token IDs."""
    return {"prompt_token_ids": [1] * n_tokens}


def _generate(llm: LLM, prompts: list[dict], max_new_tokens: int = 20) -> list[tuple[int, ...]]:
    """Run greedy generation and return a list of token-ID tuples (one per prompt)."""
    params = SamplingParams(top_k=1, max_tokens=max_new_tokens)
    outputs = llm.generate(prompts, params)
    return [tuple(out.outputs[0].token_ids) for out in outputs]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def comparison_data(chunked_prefill_llm_config: dict[str, Any]):
    """Load std and chunk models *sequentially* and collect generation results.

    Neuron cores are exclusive: only one LLM can hold them at a time.  The
    fixture therefore:
      1. creates the std LLM, runs all prompts, stores results, deletes it;
      2. creates the chunk LLM, runs the same prompts, stores results.

    All tests in this module share the pre-computed data — no extra loads.
    """
    short_prompt = _make_prompt(CHUNKED_PREFILL_CHUNK_SIZE // 2)
    long_prompt = _make_prompt(CHUNKED_PREFILL_CHUNK_SIZE * 2)
    batch_prompts = [short_prompt, long_prompt]

    # --- Standard (context-encoding) model ---
    std_llm = LLM(model=chunked_prefill_llm_config["std_neuron_model_path"])
    std_short = _generate(std_llm, [short_prompt])[0]
    std_long = _generate(std_llm, [long_prompt])[0]
    std_batch = _generate(std_llm, batch_prompts)
    del std_llm  # releases EngineCore subprocess → Neuron cores freed

    # --- Chunked-prefill model ---
    chunk_llm = LLM(model=chunked_prefill_llm_config["chunk_neuron_model_path"])
    chunk_short = _generate(chunk_llm, [short_prompt])[0]
    chunk_long = _generate(chunk_llm, [long_prompt])[0]
    chunk_batch = _generate(chunk_llm, batch_prompts)
    del chunk_llm

    return {
        "std_short": std_short,
        "std_long": std_long,
        "std_batch": std_batch,
        "chunk_short": chunk_short,
        "chunk_long": chunk_long,
        "chunk_batch": chunk_batch,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_chunked_prefill_engine_short_prompt(comparison_data: dict):
    """Short prompt (< chunk_size) must exactly match standard context encoding.

    A single chunk goes through a similar code path as standard context encoding,
    so we expect an exact token match.
    """
    std_ids = comparison_data["std_short"]
    chunk_ids = comparison_data["chunk_short"]
    assert std_ids == chunk_ids, (
        f"Chunked prefill (short) diverged from standard context encoding.\n  std  : {std_ids}\n  chunk: {chunk_ids}"
    )


def test_chunked_prefill_engine_long_prompt(comparison_data: dict):
    """Long prompt (2 × chunk_size) must closely match standard context encoding.

    Multi-chunk KV accumulation uses XLA compute_for_token_gen, while the
    standard model uses the NKI attention kernel.  These numerically different
    paths can cause greedy-token divergence after several autoregressive steps.
    We require a prefix match rather than exact equality.
    """
    MIN_PREFIX_MATCH = 10  # out of 20 generated tokens
    std_ids = comparison_data["std_long"]
    chunk_ids = comparison_data["chunk_long"]
    common = sum(1 for _ in itertools.takewhile(lambda sc: sc[0] == sc[1], zip(std_ids, chunk_ids)))
    assert common >= MIN_PREFIX_MATCH, (
        f"Chunked prefill (long) diverged too early from standard "
        f"context encoding (first {common} tokens match, need {MIN_PREFIX_MATCH}).\n"
        f"  std  : {std_ids}\n"
        f"  chunk: {chunk_ids}"
    )


def test_chunked_prefill_engine_batch(comparison_data: dict):
    """Batched chunked prefill must closely match standard context encoding.

    Sends a short and a long prompt in one call.  During the second chunk
    round the short sequence is already exhausted, exercising the skip path
    in _execute_chunked_prefill.

    We compare chunk_batch[i] against std_batch[i] (same batch composition).
    Neuron hardware introduces small numerical differences between the two
    prefill paths (NKI kernel for std, XLA compute_for_token_gen for chunked),
    which can cause greedy-token divergence after many autoregressive steps.
    We therefore require that at least ``MIN_PREFIX_MATCH`` leading tokens
    match exactly — this catches real bugs while tolerating late accumulated
    drift.
    """
    MIN_PREFIX_MATCH = 10  # out of 20 generated tokens

    std_ids_list = comparison_data["std_batch"]
    chunk_ids_list = comparison_data["chunk_batch"]

    for i, (std_ids, chunk_ids) in enumerate(zip(std_ids_list, chunk_ids_list)):
        common = sum(1 for _ in itertools.takewhile(lambda sc: sc[0] == sc[1], zip(std_ids, chunk_ids)))
        assert common >= MIN_PREFIX_MATCH, (
            f"Chunked prefill batch[{i}] diverged too early from standard "
            f"context encoding (first {common} tokens match, need {MIN_PREFIX_MATCH}).\n"
            f"  std  : {std_ids}\n"
            f"  chunk: {chunk_ids}"
        )
