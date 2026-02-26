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
"""Tests verifying that chunked prefill produces equivalent output to standard context encoding."""

import os
from tempfile import TemporaryDirectory

import pytest
import torch
from transformers import AutoTokenizer

from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.models.inference.backend.config import NxDNeuronConfig
from optimum.neuron.utils.instance import current_instance_type
from optimum.neuron.utils.system import cores_per_device
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"

# Keep sequence_length small so both models compile quickly.
# chunk_size must divide sequence_length.
SEQUENCE_LENGTH = 512
CHUNK_SIZE = 64


@pytest.fixture(scope="module")
def chunked_prefill_models():
    """Compile a standard (context-encoding) and a chunked-prefill model and yield both.

    Module-scoped so the two compilations happen only once for all parametrized tests.
    on_device_sampling is disabled for both models: chunked prefill is incompatible with it.
    """
    tp_degree = cores_per_device()
    instance_type = current_instance_type()

    common_kwargs = {
        "checkpoint_id": MODEL_ID,
        "batch_size": 1,
        "sequence_length": SEQUENCE_LENGTH,
        "tp_degree": tp_degree,
        "torch_dtype": "bfloat16",
        "on_device_sampling": False,
        "fused_qkv": True,
        "target": instance_type,
    }

    with TemporaryDirectory() as tmpdir:
        std_path = os.path.join(tmpdir, "standard")
        std_neuron_config = NxDNeuronConfig(**common_kwargs)
        std_model = NeuronModelForCausalLM.export(MODEL_ID, neuron_config=std_neuron_config)
        std_model.save_pretrained(std_path)
        del std_model
        std_model = NeuronModelForCausalLM.from_pretrained(std_path)

        chunk_path = os.path.join(tmpdir, "chunked")
        chunk_neuron_config = NxDNeuronConfig(**common_kwargs, prefill_chunk_size=CHUNK_SIZE)
        chunk_model = NeuronModelForCausalLM.export(MODEL_ID, neuron_config=chunk_neuron_config)
        chunk_model.save_pretrained(chunk_path)
        del chunk_model
        chunk_model = NeuronModelForCausalLM.from_pretrained(chunk_path)

        yield std_model, chunk_model


@pytest.mark.parametrize(
    "prompt_tokens",
    [
        CHUNK_SIZE // 2,  # partial chunk — exercises wrapper padding logic
        SEQUENCE_LENGTH // 2,  # multiple complete chunks — exercises multi-chunk KV accumulation
    ],
    ids=["short_context", "long_context"],
)
@is_inferentia_test
@requires_neuronx
def test_chunked_prefill_generates_same_tokens(chunked_prefill_models, prompt_tokens):
    """Greedy generation from chunked prefill must exactly match standard context encoding.

    Covers two prompt lengths:
    - short (< 1 chunk): exercises the chunk-padding path in the wrapper
    - long (multiple complete chunks): exercises KV accumulation across chunks
    """
    std_model, chunk_model = chunked_prefill_models
    input_ids = torch.ones((1, prompt_tokens), dtype=torch.int64)
    attention_mask = torch.ones((1, prompt_tokens), dtype=torch.int64)

    std_output = std_model.generate(input_ids, attention_mask=attention_mask, do_sample=False, max_new_tokens=20)
    chunk_output = chunk_model.generate(input_ids, attention_mask=attention_mask, do_sample=False, max_new_tokens=20)

    if std_output.tolist() != chunk_output.tolist():
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        std_text = tokenizer.decode(std_output[0, prompt_tokens:], skip_special_tokens=True)
        chunk_text = tokenizer.decode(chunk_output[0, prompt_tokens:], skip_special_tokens=True)
        pytest.fail(
            f"Chunked prefill generated different tokens than standard context encoding "
            f"(prompt_tokens={prompt_tokens}, chunk_size={CHUNK_SIZE}).\n"
            f"  std   : {std_text!r}\n"
            f"  chunk : {chunk_text!r}"
        )


@is_inferentia_test
@requires_neuronx
def test_chunked_prefill_graph_structure(chunked_prefill_models):
    """The chunked model must compile a chunked_prefill_model instead of context_encoding_model."""
    std_model, chunk_model = chunked_prefill_models

    assert hasattr(std_model, "context_encoding_model"), "Standard model missing context_encoding_model"
    assert not hasattr(std_model, "chunked_prefill_model"), "Standard model should not have chunked_prefill_model"

    assert hasattr(chunk_model, "chunked_prefill_model"), "Chunked model missing chunked_prefill_model"
    assert not hasattr(chunk_model, "context_encoding_model"), "Chunked model should not have context_encoding_model"

    assert hasattr(std_model, "token_generation_model"), "Standard model missing token_generation_model"
    assert hasattr(chunk_model, "token_generation_model"), "Chunked model missing token_generation_model"
