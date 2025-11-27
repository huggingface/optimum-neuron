# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

from optimum.neuron import NeuronModelForSeq2SeqLM
from optimum.neuron.utils.testing_utils import requires_neuronx


@pytest.mark.parametrize(
    "batch_size, sequence_length, num_beams",
    [
        [1, 64, 1],
        [1, 64, 4],
    ],
)
@requires_neuronx
def test_seq2seq_export(export_seq2seq_id, batch_size, sequence_length, num_beams):
    model = NeuronModelForSeq2SeqLM.from_pretrained(
        export_seq2seq_id,
        export=True,
        batch_size=batch_size,
        sequence_length=sequence_length,
        num_beams=num_beams,
    )
    return model


@requires_neuronx
def test_seq2seq_model_from_path(neuron_seq2seq_greedy_path):
    model = NeuronModelForSeq2SeqLM.from_pretrained(neuron_seq2seq_greedy_path)
    return model
