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
from transformers import AutoTokenizer

from optimum.neuron import NeuronModelForSeq2SeqLM
from optimum.neuron.utils.testing_utils import requires_neuronx


@pytest.mark.skip(reason="Skipping the test since `parallel_model_trace` is deprecated(to fix).")
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


# Compulsory for multiprocessing tests, since we want children processes to be spawned only in the main program.
# eg. tensor parallel tracing, `neuronx_distributed.parallel_model_trace` will spawn multiple processes to trace
# and compile the model.
if __name__ == "__main__":
    pytest.main([__file__])
