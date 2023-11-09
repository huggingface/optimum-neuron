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
from generation_utils import check_neuron_model

from optimum.neuron import NeuronModelForCausalLM, NeuronModelForSeq2SeqLM
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


class DecoderTests:
    @pytest.mark.parametrize(
        "batch_size, sequence_length, num_cores, auto_cast_type",
        [
            [1, 100, 2, "fp32"],
            [1, 100, 2, "fp16"],
            [2, 100, 2, "fp16"],
        ],
    )
    @is_inferentia_test
    @requires_neuronx
    def test_decoder_export(export_decoder_id, batch_size, sequence_length, num_cores, auto_cast_type):
        model = NeuronModelForCausalLM.from_pretrained(
            export_decoder_id,
            export=True,
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_cores=num_cores,
            auto_cast_type=auto_cast_type,
        )
        check_neuron_model(model, batch_size, sequence_length, num_cores, auto_cast_type)

    @is_inferentia_test
    @requires_neuronx
    def test_model_from_path(neuron_decoder_path):
        model = NeuronModelForCausalLM.from_pretrained(neuron_decoder_path)
        check_neuron_model(model)


class Seq2SeqTests:
    @pytest.mark.parametrize(
        "batch_size, sequence_length, num_beams",
        [
            [1, 32, 1],
            [1, 32, 4],
        ],
    )
    @is_inferentia_test
    @requires_neuronx
    def test_seq2seq_export(export_seq2seq_id, batch_size, sequence_length, num_beams):
        model = NeuronModelForSeq2SeqLM.from_pretrained(
            export_seq2seq_id,
            export=True,
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_beams=num_beams,
        )

    @is_inferentia_test
    @requires_neuronx
    def test_model_from_path(neuron_seq2seq_path):
        model = NeuronModelForSeq2SeqLM.from_pretrained(neuron_seq2seq_path)
