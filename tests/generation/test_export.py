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

from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


@pytest.mark.parametrize(
    "batch_size, sequence_length, num_cores, auto_cast_type",
    [
        [1, 128, 2, "fp32"],
        [1, 128, 2, "fp32"],
        [2, 512, 2, "bf16"],
    ],
)
@is_inferentia_test
@requires_neuronx
def test_model_export(export_model_id, batch_size, sequence_length, num_cores, auto_cast_type):
    model = NeuronModelForCausalLM.from_pretrained(
        export_model_id,
        export=True,
        batch_size=batch_size,
        sequence_length=sequence_length,
        num_cores=num_cores,
        auto_cast_type=auto_cast_type,
    )
    check_neuron_model(model, batch_size, sequence_length, num_cores, auto_cast_type)


@is_inferentia_test
@requires_neuronx
def test_model_from_path(neuron_model_path):
    model = NeuronModelForCausalLM.from_pretrained(neuron_model_path)
    check_neuron_model(model)
