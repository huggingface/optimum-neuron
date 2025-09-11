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
from optimum.utils.testing_utils import USER
from transformers import T5ForConditionalGeneration

from optimum.neuron import NeuronModelForSeq2SeqLM
from optimum.neuron.utils.testing_utils import requires_neuronx


SEQ2SEQ_MODEL_NAMES = {
    "t5": "hf-internal-testing/tiny-random-t5",
}
SEQ2SEQ_MODEL_CLASSES = {
    "t5": T5ForConditionalGeneration,
}


@pytest.fixture(scope="module", params=[SEQ2SEQ_MODEL_NAMES[model_arch] for model_arch in SEQ2SEQ_MODEL_NAMES])
def export_seq2seq_id(request):
    return request.param


@pytest.fixture(scope="module", params=[SEQ2SEQ_MODEL_CLASSES[model_arch] for model_arch in SEQ2SEQ_MODEL_NAMES])
def export_seq2seq_model_class(request):
    return request.param


@pytest.fixture(scope="module")
@requires_neuronx
def neuron_seq2seq_beam_path(export_seq2seq_id):
    model = NeuronModelForSeq2SeqLM.from_pretrained(
        export_seq2seq_id, export=True, batch_size=1, sequence_length=64, num_beams=4
    )
    model_dir = TemporaryDirectory()
    model_path = model_dir.name
    model.save_pretrained(model_path)
    del model
    # Yield instead of returning to keep a reference to the temporary directory.
    # It will go out of scope and be released only once all tests needing the fixture
    # have been completed.
    yield model_path


@pytest.fixture(scope="module")
@requires_neuronx
def neuron_seq2seq_beam_path_with_optional_outputs(export_seq2seq_id):
    model = NeuronModelForSeq2SeqLM.from_pretrained(
        export_seq2seq_id,
        export=True,
        batch_size=1,
        sequence_length=64,
        num_beams=4,
        output_attentions=True,
        output_hidden_states=True,
    )
    model_dir = TemporaryDirectory()
    model_path = model_dir.name
    model.save_pretrained(model_path)
    del model
    # Yield instead of returning to keep a reference to the temporary directory.
    # It will go out of scope and be released only once all tests needing the fixture
    # have been completed.
    yield model_path


@pytest.fixture(scope="module")
@requires_neuronx
def neuron_seq2seq_greedy_path(export_seq2seq_id):
    model = NeuronModelForSeq2SeqLM.from_pretrained(
        export_seq2seq_id, export=True, batch_size=1, sequence_length=64, num_beams=1
    )
    model_dir = TemporaryDirectory()
    model_path = model_dir.name
    model.save_pretrained(model_path)
    del model
    # Yield instead of returning to keep a reference to the temporary directory.
    # It will go out of scope and be released only once all tests needing the fixture
    # have been completed.
    yield model_path


@pytest.fixture(scope="module")
@requires_neuronx
def neuron_seq2seq_greedy_path_with_optional_outputs(export_seq2seq_id):
    model = NeuronModelForSeq2SeqLM.from_pretrained(
        export_seq2seq_id,
        export=True,
        batch_size=1,
        sequence_length=64,
        num_beams=1,
        output_attentions=True,
        output_hidden_states=True,
    )
    model_dir = TemporaryDirectory()
    model_path = model_dir.name
    model.save_pretrained(model_path)
    del model
    # Yield instead of returning to keep a reference to the temporary directory.
    # It will go out of scope and be released only once all tests needing the fixture
    # have been completed.
    yield model_path


@pytest.fixture(scope="module")
@requires_neuronx
def neuron_seq2seq_tp2_path():
    model = NeuronModelForSeq2SeqLM.from_pretrained(
        "michaelbenayoun/t5-tiny-random",
        export=True,
        tensor_parallel_size=2,
        dynamic_batch_size=False,
        batch_size=1,
        sequence_length=64,
        num_beams=4,
    )
    model_dir = TemporaryDirectory()
    model_path = model_dir.name
    model.save_pretrained(model_path)
    del model
    # Yield instead of returning to keep a reference to the temporary directory.
    # It will go out of scope and be released only once all tests needing the fixture
    # have been completed.
    yield model_path


@pytest.fixture(scope="module")
def neuron_push_seq2seq_id(export_seq2seq_id):
    model_name = export_seq2seq_id.split("/")[-1]
    repo_id = f"{USER}/{model_name}-neuronx"
    return repo_id


def check_neuron_model(neuron_model, batch_size=None, sequence_length=None, num_cores=None, auto_cast_type=None):
    neuron_config = getattr(neuron_model.config, "neuron", None)
    assert neuron_config
    if batch_size:
        assert neuron_config["batch_size"] == batch_size
    if sequence_length:
        assert neuron_config["sequence_length"] == sequence_length
    if num_cores:
        assert neuron_config["num_cores"] == num_cores
    if auto_cast_type:
        assert neuron_config["auto_cast_type"] == auto_cast_type
