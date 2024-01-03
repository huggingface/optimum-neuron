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

import pytest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from optimum.neuron import NeuronModelForCausalLM, NeuronModelForSeq2SeqLM
from optimum.neuron.utils.testing_utils import is_inferentia_test, is_trainium_test, requires_neuronx
from optimum.neuron.utils.training_utils import patch_generation_mixin_to_general_neuron_generation_mixin


def _test_model_generation(model, tokenizer, batch_size, input_length, **gen_kwargs):
    input_ids = torch.ones((batch_size, input_length), dtype=torch.int64)
    with torch.inference_mode():
        sample_output = model.generate(input_ids, **gen_kwargs)
        assert sample_output.shape[0] == batch_size


@pytest.mark.parametrize(
    "gen_kwargs",
    [
        {"do_sample": True},
        {"do_sample": True, "temperature": 0.7},
        {"do_sample": False},
        {"do_sample": False, "repetition_penalty": 1.2},
    ],
    ids=["sample", "sample-with-temp", "greedy", "greedy_no-repeat"],
)
@is_inferentia_test
@requires_neuronx
def test_decoder_generation(neuron_decoder_path, gen_kwargs):
    model = NeuronModelForCausalLM.from_pretrained(neuron_decoder_path)
    tokenizer = AutoTokenizer.from_pretrained(neuron_decoder_path)
    _test_model_generation(model, tokenizer, model.batch_size, 10, **gen_kwargs)


@is_inferentia_test
@requires_neuronx
def test_model_generation_input_dimensions(neuron_decoder_path):
    model = NeuronModelForCausalLM.from_pretrained(neuron_decoder_path)
    tokenizer = AutoTokenizer.from_pretrained(neuron_decoder_path)
    # Using valid input dimensions
    _test_model_generation(model, tokenizer, model.batch_size, model.max_length // 2)
    # Using an incompatible batch_size
    with pytest.raises(ValueError, match="The specified batch_size"):
        _test_model_generation(model, tokenizer, model.batch_size + 1, model.max_length)
    # Using an incompatible input length
    with pytest.raises(ValueError, match="The input sequence length"):
        _test_model_generation(model, tokenizer, model.batch_size, input_length=model.max_length * 2)


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
def test_general_decoder_generation(export_decoder_id, gen_kwargs):
    model = AutoModelForCausalLM.from_pretrained(export_decoder_id)
    patch_generation_mixin_to_general_neuron_generation_mixin(model)
    tokenizer = AutoTokenizer.from_pretrained(export_decoder_id)
    _test_model_generation(model, tokenizer, 1, 10, **gen_kwargs)


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
def test_general_seq2seq_generation(export_seq2seq_id, export_seq2seq_model_class, gen_kwargs):
    model = export_seq2seq_model_class.from_pretrained(export_seq2seq_id)
    patch_generation_mixin_to_general_neuron_generation_mixin(model)
    tokenizer = AutoTokenizer.from_pretrained(export_seq2seq_id)
    _test_model_generation(model, tokenizer, 1, 10, **gen_kwargs)