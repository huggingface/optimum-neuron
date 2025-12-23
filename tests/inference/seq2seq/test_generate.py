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

from transformers import AutoTokenizer

from optimum.neuron import NeuronModelForSeq2SeqLM
from optimum.neuron.utils.testing_utils import requires_neuronx


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
