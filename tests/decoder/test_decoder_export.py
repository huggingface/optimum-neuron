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
import torch
from transformers import AutoModelForCausalLM

from optimum.neuron import NeuronModelForCausalLM


DECODER_MODELS = {
    "llama": "llamafactory/tiny-random-Llama-3",
    "llama4_text": "tiny-random/llama-4",
    "qwen2": "yujiepan/qwen2.5-128k-tiny-random",
    "qwen3-moe": "optimum-internal-testing/tiny-random-qwen3_moe",
    "granite": "hf-internal-testing/tiny-random-GraniteForCausalLM",
    "phi3": "yujiepan/phi-4-tiny-random",
    "mixtral": "dacorvo/Mixtral-tiny",
    "smollm3": "HuggingFaceTB/SmolLM3-3B",
}


@pytest.fixture(
    scope="session",
    params=[DECODER_MODELS[model_arch] for model_arch in DECODER_MODELS],
    ids=list(DECODER_MODELS.keys()),
)
def any_decoder_model(request):
    return request.param


def check_neuron_config(neuron_config, **kwargs):
    for key, value in kwargs.items():
        aliases = {
            "num_cores": "tp_degree",
            "tensor_parallel_size": "tp_degree",
        }
        if key in aliases:
            key = aliases[key]
        if value is not None:
            assert getattr(neuron_config, key) == value, (
                f"Expected {key} to be {value}, but got {getattr(neuron_config, key)}"
            )


def check_neuron_model(neuron_model):
    batch_size = neuron_model.neuron_config.batch_size
    input_shape = (batch_size, min(10, neuron_model.neuron_config.sequence_length))
    input_ids = torch.ones(input_shape, dtype=torch.int64)
    position_ids = torch.arange(input_shape[1]).unsqueeze(0).repeat(batch_size, 1)
    seq_ids = torch.arange(batch_size)
    sampling_params = torch.ones((batch_size, 3))
    outputs = neuron_model(
        input_ids=input_ids,
        position_ids=position_ids,
        seq_ids=seq_ids,
        sampling_params=sampling_params,
    )
    assert outputs is not None, "Model outputs should not be None"


def _test_decoder_export_save_reload(
    model_id: str,
    is_local: bool,
    load_weights: bool,
):
    export_kwargs = {"batch_size": 1, "sequence_length": 1024, "tensor_parallel_size": 2}
    neuron_config = NeuronModelForCausalLM.get_neuron_config(model_name_or_path=model_id, **export_kwargs)
    with TemporaryDirectory() as model_path:
        if is_local:
            with TemporaryDirectory() as tmpdir:
                model = AutoModelForCausalLM.from_pretrained(model_id)
                model.save_pretrained(tmpdir)
                model = NeuronModelForCausalLM.export(
                    model_id=tmpdir, neuron_config=neuron_config, load_weights=load_weights
                )
                model.save_pretrained(model_path)
        else:
            model = NeuronModelForCausalLM.export(
                model_id=model_id, neuron_config=neuron_config, load_weights=load_weights
            )
            model.save_pretrained(model_path)
        check_neuron_config(model.neuron_config, **export_kwargs)
        if load_weights:
            check_neuron_model(model)
        model = NeuronModelForCausalLM.from_pretrained(model_path)
        check_neuron_model(model)


def test_decoder_export_hub_models(
    any_decoder_model: str,
):
    _test_decoder_export_save_reload(model_id=any_decoder_model, is_local=False, load_weights=False)


# TODO: remove this test, it is just a test for CI
@pytest.mark.parametrize("is_local", [True, False], ids=["local", "from_hub"])
@pytest.mark.parametrize("load_weights", [True, False], ids=["with-weights", "without-weights"])
def test_decoder_export_save_reload(
    is_local: bool,
    load_weights: bool,
):
    _test_decoder_export_save_reload(model_id=DECODER_MODELS["qwen2"], is_local=is_local, load_weights=load_weights)
