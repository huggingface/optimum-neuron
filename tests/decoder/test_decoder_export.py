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

from tempfile import TemporaryDirectory

import pytest
from transformers import AutoModelForCausalLM

from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


DECODER_MODEL_ARCHITECTURES = ["llama", "granite", "qwen2", "phi3"]
DECODER_MODEL_NAMES = {
    "llama": "llamafactory/tiny-random-Llama-3",
    "qwen2": "yujiepan/qwen2.5-128k-tiny-random",
    "granite": "hf-internal-testing/tiny-random-GraniteForCausalLM",
    "phi3": "yujiepan/phi-4-tiny-random",
}


@pytest.fixture(
    scope="session", params=[DECODER_MODEL_NAMES[model_arch] for model_arch in DECODER_MODEL_ARCHITECTURES]
)
def export_decoder_id(request):
    return request.param


def check_neuron_model(neuron_model, batch_size=None, sequence_length=None, num_cores=None, auto_cast_type=None):
    neuron_config = neuron_model.neuron_config
    if batch_size:
        assert neuron_config.batch_size == batch_size
    if sequence_length:
        assert neuron_config.sequence_length == sequence_length
    if num_cores:
        assert neuron_config.tp_degree == num_cores
    if auto_cast_type:
        assert neuron_config.auto_cast_type == auto_cast_type


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
@pytest.mark.parametrize("local", [True, False], ids=["local", "from_hub"])
def test_decoder_export_save_reload(local, export_decoder_id, batch_size, sequence_length, num_cores, auto_cast_type):
    export_kwargs = {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "num_cores": num_cores,
        "auto_cast_type": auto_cast_type,
    }
    with TemporaryDirectory() as model_path:
        if local:
            with TemporaryDirectory() as tmpdir:
                model = AutoModelForCausalLM.from_pretrained(export_decoder_id)
                model.save_pretrained(tmpdir)
                model = NeuronModelForCausalLM.from_pretrained(tmpdir, export=True, **export_kwargs)
                model.save_pretrained(model_path)
        else:
            model = NeuronModelForCausalLM.from_pretrained(export_decoder_id, export=True, **export_kwargs)
            model.save_pretrained(model_path)
        check_neuron_model(model, **export_kwargs)
        del model
        model = NeuronModelForCausalLM.from_pretrained(model_path)
        check_neuron_model(model, **export_kwargs)
