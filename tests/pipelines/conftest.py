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
from tempfile import TemporaryDirectory

import pytest
from transformers import AutoTokenizer

from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.utils.testing_utils import requires_neuronx


STD_TEXT_TASKS = [
    "feature-extraction",
    "fill-mask",
    "question-answering",
    "text-classification",
    "token-classification",
]


@pytest.fixture(scope="module", params=STD_TEXT_TASKS)
def std_text_task(request):
    return request.param


@pytest.fixture(scope="module")
@requires_neuronx
def inf_decoder_path(inf_decoder_model):
    model = NeuronModelForCausalLM.from_pretrained(
        inf_decoder_model, export=True, batch_size=1, sequence_length=128, num_cores=2
    )
    model_dir = TemporaryDirectory()
    model_path = model_dir.name
    model.save_pretrained(model_path)
    del model
    tokenizer = AutoTokenizer.from_pretrained(inf_decoder_model)
    tokenizer.save_pretrained(model_path)
    del tokenizer
    # Yield instead of returning to keep a reference to the temporary directory.
    # It will go out of scope and be released only once all tests needing the fixture
    # have been completed.
    yield model_path
