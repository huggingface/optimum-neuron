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
from optimum.neuron.models.inference.nxd.llama.modeling_llama import LlamaNxDModelForCausalLM
from optimum.neuron.utils import map_torch_dtype
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


DECODER_MODEL_ARCHITECTURES = ["llama", "granite", "qwen2", "phi3", "mixtral"]
DECODER_MODEL_NAMES = {
    "llama": "llamafactory/tiny-random-Llama-3",
    "qwen2": "yujiepan/qwen2.5-128k-tiny-random",
    "granite": "hf-internal-testing/tiny-random-GraniteForCausalLM",
    "phi3": "yujiepan/phi-4-tiny-random",
    "mixtral": "dacorvo/Mixtral-tiny",
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
        if hasattr(neuron_config, "auto_cast_type"):
            assert neuron_config.auto_cast_type == auto_cast_type
        elif hasattr(neuron_config, "torch_dtype"):
            assert neuron_config.torch_dtype == map_torch_dtype(auto_cast_type)


def _test_decoder_export_save_reload(
    model_cls,
    is_local: bool,
    model_id: str,
    batch_size: int,
    sequence_length: int,
    num_cores: int,
    auto_cast_type: str,
):
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
                model = model_cls.from_pretrained(tmpdir, export=True, **export_kwargs)
                model.save_pretrained(model_path)
        else:
            model = model_cls.from_pretrained(model_id, export=True, **export_kwargs)
            model.save_pretrained(model_path)
        check_neuron_model(model, **export_kwargs)
        del model
        model = model_cls.from_pretrained(model_path)
        check_neuron_model(model, **export_kwargs)


@pytest.mark.parametrize(
    "batch_size, sequence_length, num_cores, auto_cast_type",
    [
        [1, 100, 2, "bf16"],
        [1, 100, 2, "fp16"],
        [2, 100, 2, "fp16"],
    ],
)
@is_inferentia_test
@requires_neuronx
@pytest.mark.parametrize("is_local", [True, False], ids=["local", "from_hub"])
def test_decoder_export_save_reload(
    is_local, export_decoder_id, batch_size, sequence_length, num_cores, auto_cast_type
):
    _test_decoder_export_save_reload(
        NeuronModelForCausalLM,
        is_local=is_local,
        model_id=export_decoder_id,
        batch_size=batch_size,
        sequence_length=sequence_length,
        num_cores=num_cores,
        auto_cast_type=auto_cast_type,
    )


@pytest.mark.parametrize(
    "batch_size, sequence_length, num_cores, auto_cast_type",
    [
        [1, 100, 2, "bf16"],
        [1, 100, 2, "fp16"],
        [2, 100, 2, "fp16"],
    ],
)
@is_inferentia_test
@requires_neuronx
@pytest.mark.parametrize("is_local", [True, False], ids=["local", "from_hub"])
def test_nxd_llama_export_save_reload(is_local, batch_size, sequence_length, num_cores, auto_cast_type):
    _test_decoder_export_save_reload(
        LlamaNxDModelForCausalLM,
        is_local=is_local,
        model_id=DECODER_MODEL_NAMES["llama"],
        batch_size=batch_size,
        sequence_length=sequence_length,
        num_cores=num_cores,
        auto_cast_type=auto_cast_type,
    )
