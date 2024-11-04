# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import os

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from optimum.neuron import NeuronModelForSeq2SeqLM
from optimum.neuron.utils.testing_utils import is_inferentia_test, is_trainium_test, requires_neuronx
from optimum.neuron.utils.training_utils import patch_generation_mixin_to_general_neuron_generation_mixin


def _test_model_generation_trn(model, tokenizer, batch_size, input_length, **gen_kwargs):
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    model = model.to(device)
    patch_generation_mixin_to_general_neuron_generation_mixin(model)
    input_ids = torch.ones((batch_size, input_length), dtype=torch.int64)
    sample_output = model.generate(input_ids, **gen_kwargs)
    assert sample_output.shape[0] == batch_size


@is_inferentia_test
@requires_neuronx
def test_seq2seq_generation_beam(neuron_seq2seq_beam_path):
    model = NeuronModelForSeq2SeqLM.from_pretrained(neuron_seq2seq_beam_path)
    tokenizer = AutoTokenizer.from_pretrained(neuron_seq2seq_beam_path)
    inputs = tokenizer("translate English to German: Lets eat good food.", return_tensors="pt")

    # 1. max length
    output = model.generate(**inputs, num_return_sequences=2, max_length=5)
    assert len(output[0]) <= 5

    # 2. min length
    output = model.generate(**inputs, num_return_sequences=2, min_length=10)
    assert len(output[0]) >= 10

    # 3. max new tokens
    output = model.generate(**inputs, num_return_sequences=2, max_new_tokens=5)
    assert len(output[0].unique()) <= 5 + 1  # +1 for `decoder_start_token_id`


@is_inferentia_test
@requires_neuronx
def test_seq2seq_generation_beam_with_optional_outputs(neuron_seq2seq_beam_path_with_optional_outputs):
    model = NeuronModelForSeq2SeqLM.from_pretrained(neuron_seq2seq_beam_path_with_optional_outputs)
    tokenizer = AutoTokenizer.from_pretrained(neuron_seq2seq_beam_path_with_optional_outputs)
    inputs = tokenizer("translate English to German: Lets eat good food.", return_tensors="pt")

    output = model.generate(
        **inputs,
        num_return_sequences=1,
        max_length=20,
        output_scores=True,
        output_attentions=True,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    assert "scores" in output
    assert "decoder_attentions" in output
    assert "cross_attentions" in output
    assert "decoder_hidden_states" in output


@is_inferentia_test
@requires_neuronx
def test_seq2seq_generation_greedy(neuron_seq2seq_greedy_path):
    model = NeuronModelForSeq2SeqLM.from_pretrained(neuron_seq2seq_greedy_path)
    tokenizer = AutoTokenizer.from_pretrained(neuron_seq2seq_greedy_path)
    inputs = tokenizer("translate English to German: Lets eat good food.", return_tensors="pt")

    # 1. max length
    output = model.generate(**inputs, num_return_sequences=1, max_length=5)
    assert len(output[0]) <= 5

    # 2. min length
    output = model.generate(**inputs, num_return_sequences=1, min_length=10)
    assert len(output[0]) >= 10

    # 3. max new tokens
    output = model.generate(**inputs, num_return_sequences=1, max_new_tokens=5)
    assert len(output[0]) <= 5 + 1  # +1 for `decoder_start_token_id`


@is_inferentia_test
@requires_neuronx
def test_seq2seq_generation_greedy_with_optional_outputs(neuron_seq2seq_greedy_path_with_optional_outputs):
    model = NeuronModelForSeq2SeqLM.from_pretrained(neuron_seq2seq_greedy_path_with_optional_outputs)
    tokenizer = AutoTokenizer.from_pretrained(neuron_seq2seq_greedy_path_with_optional_outputs)
    inputs = tokenizer("translate English to German: Lets eat good food.", return_tensors="pt")

    output = model.generate(
        **inputs,
        num_return_sequences=1,
        max_length=20,
        output_attentions=True,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    assert "decoder_attentions" in output
    assert "cross_attentions" in output
    assert "decoder_hidden_states" in output


@is_inferentia_test
@requires_neuronx
def test_seq2seq_generation_tp2(neuron_seq2seq_tp2_path):
    model = NeuronModelForSeq2SeqLM.from_pretrained(neuron_seq2seq_tp2_path)
    tokenizer = AutoTokenizer.from_pretrained(neuron_seq2seq_tp2_path)
    inputs = tokenizer("translate English to German: Lets eat good food.", return_tensors="pt")

    output = model.generate(
        **inputs,
        num_return_sequences=1,
        max_length=20,
        output_attentions=True,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    assert "decoder_attentions" in output
    assert "cross_attentions" in output
    assert "decoder_hidden_states" in output


@pytest.mark.skip("Makes pytest fail, to fix.")
@pytest.mark.parametrize(
    "gen_kwargs",
    [
        {"do_sample": True},
        {"do_sample": True, "temperature": 0.7},
        {"do_sample": False},
        {"do_sample": False, "repetition_penalty": 1.2},
        {"num_beams": 4},
        {"num_beams": 4, "num_beam_groups": 2, "diversity_penalty": 0.1},
    ],
    ids=["sample", "sample-with-temp", "greedy", "greedy_no-repeat", "beam", "group-beam"],
)
@is_trainium_test
@requires_neuronx
def test_general_decoder_generation(export_trn_decoder_id, gen_kwargs):
    os.environ["NEURON_CC_FLAGS"] = "-O1 --model-type=transformer"
    model = AutoModelForCausalLM.from_pretrained(export_trn_decoder_id)
    tokenizer = AutoTokenizer.from_pretrained(export_trn_decoder_id)
    _test_model_generation_trn(model, tokenizer, 1, 10, **gen_kwargs)


@pytest.mark.skip("Makes pytest fail, to fix.")
@pytest.mark.parametrize(
    "gen_kwargs",
    [
        {"do_sample": True},
        {"do_sample": True, "temperature": 0.7},
        {"do_sample": False},
        {"do_sample": False, "repetition_penalty": 1.2},
        {"num_beams": 4},
        {"num_beams": 4, "do_sample": True, "temperature": 0.7},
        {"num_beams": 4, "num_beam_groups": 2, "diversity_penalty": 0.1},
    ],
    ids=["sample", "sample-with-temp", "greedy", "greedy_no-repeat", "beam", "beam-sample", "group-beam"],
)
@is_trainium_test
@requires_neuronx
def test_general_seq2seq_generation(export_seq2seq_id, export_seq2seq_model_class, gen_kwargs):
    os.environ["NEURON_CC_FLAGS"] = "-O1 --model-type=transformer"
    model = export_seq2seq_model_class.from_pretrained(export_seq2seq_id)
    tokenizer = AutoTokenizer.from_pretrained(export_seq2seq_id)
    _test_model_generation_trn(model, tokenizer, 1, 10, **gen_kwargs)


# Compulsory for multiprocessing tests, since we want children processes to be spawned only in the main program.
# eg. tensor parallel tracing, `neuronx_distributed.parallel_model_trace` will spawn multiple processes to trace
# and compile the model.
if __name__ == "__main__":
    pytest.main([__file__])
