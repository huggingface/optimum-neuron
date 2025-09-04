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
from transformers import AutoConfig, AutoModelForCausalLM

from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.models.auto_model import get_neuron_model_class
from optimum.neuron.utils import DTYPE_MAPPER
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


DECODER_MODEL_ARCHITECTURES = ["llama", "granite", "qwen2", "qwen3-moe", "phi3", "mixtral"]
DECODER_MODEL_NAMES = {
    "llama": "llamafactory/tiny-random-Llama-3",
    "qwen2": "yujiepan/qwen2.5-128k-tiny-random",
    "qwen3-moe": "optimum-internal-testing/tiny-random-qwen3_moe",
    "granite": "hf-internal-testing/tiny-random-GraniteForCausalLM",
    "phi3": "yujiepan/phi-4-tiny-random",
    "mixtral": "dacorvo/Mixtral-tiny",
    "smollm3": "HuggingFaceTB/SmolLM3-3B",
}


@pytest.fixture(
    scope="session", params=[DECODER_MODEL_NAMES[model_arch] for model_arch in DECODER_MODEL_ARCHITECTURES]
)
def export_decoder_id(request):
    return request.param


def check_neuron_config(neuron_config, **kwargs):
    for key, value in kwargs.items():
        aliases = {
            "num_cores": "tp_degree",
            "tensor_parallel_size": "tp_degree",
            "auto_cast_type": "torch_dtype",
        }
        if key in aliases:
            key = aliases[key]
        if value is not None:
            if key == "torch_dtype" and isinstance(value, str):
                value = DTYPE_MAPPER.pt(value)
            assert getattr(neuron_config, key) == value, (
                f"Expected {key} to be {value}, but got {getattr(neuron_config, key)}"
            )


def check_neuron_model(neuron_model):
    batch_size = neuron_model.neuron_config.batch_size
    input_shape = (batch_size, min(10, neuron_model.neuron_config.sequence_length))
    input_ids = torch.ones(input_shape, dtype=torch.int64)
    attention_mask = torch.ones(input_shape, dtype=torch.int64)
    on_device_sampling = getattr(neuron_model.neuron_config, "on_device_sampling", False)
    sampling_params = torch.ones((batch_size, 3)) if on_device_sampling else None
    model_inputs = neuron_model.prepare_inputs_for_prefill(
        input_ids=input_ids, attention_mask=attention_mask, sampling_params=sampling_params
    )
    outputs = neuron_model(**model_inputs)
    assert outputs is not None, "Model outputs should not be None"


@pytest.mark.parametrize(
    "batch_size, sequence_length, num_cores, auto_cast_type",
    [
        [1, 1024, 2, "bf16"],
        [2, 1024, 2, "fp16"],
    ],
)
@is_inferentia_test
@requires_neuronx
@pytest.mark.parametrize("is_local", [True, False], ids=["local", "from_hub"])
def test_decoder_export_from_pretrained_save_reload(
    is_local: bool,
    export_decoder_id: str,
    batch_size: int,
    sequence_length: int,
    num_cores: int,
    auto_cast_type: str,
):
    model_id = export_decoder_id
    export_kwargs = {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "num_cores": num_cores,
        "auto_cast_type": auto_cast_type,
    }
    with TemporaryDirectory() as model_path:
        if is_local:
            with TemporaryDirectory() as tmpdir:
                model = AutoModelForCausalLM.from_pretrained(model_id)
                model.save_pretrained(tmpdir)
                model = NeuronModelForCausalLM.from_pretrained(tmpdir, export=True, **export_kwargs)
                model.save_pretrained(model_path)
        else:
            model = NeuronModelForCausalLM.from_pretrained(model_id, export=True, **export_kwargs)
            model.save_pretrained(model_path)
        check_neuron_config(model.neuron_config, **export_kwargs)
        check_neuron_model(model)
        del model
        model = NeuronModelForCausalLM.from_pretrained(model_path)
        check_neuron_model(model)


@pytest.mark.parametrize("is_local", [True, False], ids=["local", "from_hub"])
def test_decoder_export_only_save_reload(
    is_local: bool,
    export_decoder_id: str,
):
    model_id = export_decoder_id
    model_config = AutoConfig.from_pretrained(export_decoder_id)
    neuron_cls = get_neuron_model_class(model_config.model_type, task="text-generation", mode="inference")
    export_kwargs = {"batch_size": 1, "sequence_length": 1024, "tensor_parallel_size": 2, "auto_cast_type": "bf16"}
    neuron_config = neuron_cls.get_neuron_config(
        model_name_or_path=export_decoder_id, config=model_config, **export_kwargs
    )
    with TemporaryDirectory() as model_path:
        if is_local:
            with TemporaryDirectory() as tmpdir:
                model = AutoModelForCausalLM.from_pretrained(model_id)
                model.save_pretrained(tmpdir)
                model = neuron_cls.export(
                    model_id=tmpdir, config=model_config, neuron_config=neuron_config, load_weights=False
                )
                model.save_pretrained(model_path)
        else:
            model = neuron_cls.export(
                model_id=model_id, config=model_config, neuron_config=neuron_config, load_weights=False
            )
            model.save_pretrained(model_path)
        check_neuron_config(model.neuron_config, **export_kwargs)
        model = NeuronModelForCausalLM.from_pretrained(model_path)
        check_neuron_model(model)
