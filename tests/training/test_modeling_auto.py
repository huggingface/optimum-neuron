# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
import torch
from transformers import AutoConfig

from optimum.neuron.models.training import NeuronModel, NeuronModelForCausalLM, TrainingNeuronConfig
from optimum.neuron.models.training.training_utils import is_custom_modeling_model
from optimum.neuron.utils.testing_utils import is_trainium_test

from .distributed_utils import distributed_test


@pytest.mark.parametrize("from_pretrained", [False, True], ids=["from_config", "from_pretrained"])
@distributed_test(world_size=1)
@is_trainium_test
def test_auto_model_with_supported_architecture(from_pretrained):
    trn_config = TrainingNeuronConfig()
    kwargs = {"torch_dtype": torch.bfloat16}
    for model_name_or_path in [
        "michaelbenayoun/llama-2-tiny-4kv-heads-4layers-random",
        "michaelbenayoun/granite-tiny-4kv-heads-4layers-random",
        "michaelbenayoun/qwen3-tiny-4kv-heads-4layers-random",
    ]:
        print(f"Testing model: {model_name_or_path}")
        if from_pretrained:
            model = NeuronModel.from_pretrained(model_name_or_path, trn_config=trn_config, **kwargs)
        else:
            config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)
            model = NeuronModel.from_config(config, trn_config=trn_config)

        assert is_custom_modeling_model(model), "Model should be a custom Neuron model for training."
        if from_pretrained:
            assert next(model.parameters()).dtype is torch.bfloat16, "Model parameters should be in bfloat16 dtype."


@distributed_test(world_size=1)
@is_trainium_test
def test_auto_model_with_unsupported_architecture():
    with pytest.raises(
        ValueError, match="Model type bert is not supported for task model in neuron in training mode.(.)*"
    ):
        NeuronModel.from_pretrained("bert-base-uncased", TrainingNeuronConfig())


@pytest.mark.parametrize("from_pretrained", [False, True], ids=["from_config", "from_pretrained"])
@distributed_test(world_size=1)
@is_trainium_test
def test_auto_model_for_causal_lm_with_supported_architecture(from_pretrained):
    trn_config = TrainingNeuronConfig()
    kwargs = {"torch_dtype": torch.bfloat16}
    for model_name_or_path in [
        "michaelbenayoun/llama-2-tiny-4kv-heads-4layers-random",
        "michaelbenayoun/granite-tiny-4kv-heads-4layers-random",
        "michaelbenayoun/qwen3-tiny-4kv-heads-4layers-random",
    ]:
        print(f"Testing model: {model_name_or_path}")
        if from_pretrained:
            model = NeuronModelForCausalLM.from_pretrained(model_name_or_path, trn_config=trn_config, **kwargs)
        else:
            config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)
            model = NeuronModelForCausalLM.from_config(config, trn_config=trn_config)
        assert is_custom_modeling_model(model), "Model should be a custom Neuron model for training."
        if from_pretrained:
            assert next(model.parameters()).dtype is torch.bfloat16, "Model parameters should be in bfloat16 dtype."


@distributed_test(world_size=1)
@is_trainium_test
def test_auto_model_for_causal_lm_with_unsupported_architecture():
    with pytest.raises(
        ValueError, match="Model type gpt2 is not supported for task text-generation in neuron in training mode.(.)*"
    ):
        NeuronModelForCausalLM.from_pretrained("gpt2", TrainingNeuronConfig())
