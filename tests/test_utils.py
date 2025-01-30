# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Tests for utility functions and classes."""

from transformers import BertConfig, BertForSequenceClassification, PreTrainedModel, Wav2Vec2Config, Wav2Vec2Model

from optimum.neuron.accelerate.accelerator import MODEL_PATCHING_SPECS
from optimum.neuron.utils import ModelPatcher
from optimum.neuron.utils.testing_utils import is_trainium_test
from optimum.neuron.utils.training_utils import is_model_officially_supported


@is_trainium_test
def test_is_model_officially_supported():
    class DummyModelClass(PreTrainedModel):
        pass

    unsupported_model = DummyModelClass(BertConfig())
    assert is_model_officially_supported(unsupported_model) is False

    class Child(BertForSequenceClassification):
        pass

    child_model = Child(BertConfig())
    assert is_model_officially_supported(child_model) is False

    bert_model = BertForSequenceClassification(BertConfig())
    assert is_model_officially_supported(bert_model) is True


def test_patch_model():
    bert_model = BertForSequenceClassification(BertConfig())
    patching_specs = []
    for spec in MODEL_PATCHING_SPECS:
        patching_specs.append((bert_model,) + spec)

    with ModelPatcher(patching_specs, ignore_missing_attributes=True):
        assert getattr(bert_model.config, "layerdrop", None) == 0
        # Checking that the context manager exists.
        with bert_model.no_sync():
            pass

    wav2vec2_model = Wav2Vec2Model(Wav2Vec2Config())
    assert wav2vec2_model.config.layerdrop > 0, (
        "Default Wav2vec2Config layerdrop value is already 0 so the test will not check anything."
    )
    patching_specs = []
    for spec in MODEL_PATCHING_SPECS:
        patching_specs.append((wav2vec2_model,) + spec)
    with ModelPatcher(patching_specs, ignore_missing_attributes=True):
        assert wav2vec2_model.config.layerdrop == 0, "layerdrop was not patched properly."

        # Checking that the context manager exists.
        with wav2vec2_model.no_sync():
            pass
