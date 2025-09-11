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

import os

import datasets
import pytest
import torch
from neuronx_distributed.parallel_layers.parallel_state import (
    get_pipeline_model_parallel_size,
    get_tensor_model_parallel_size,
)
from transformers import AutoTokenizer

from optimum.neuron.accelerate import NeuronAccelerator
from optimum.neuron.accelerate.utils.dataclasses import MixedPrecisionConfig, MixedPrecisionMode
from optimum.neuron.models.training import LlamaForCausalLM as NeuronLlamaForCausalLM
from optimum.neuron.models.training.config import TrainingNeuronConfig
from optimum.neuron.trainers import NeuronTrainer, NeuronTrainingArguments
from optimum.neuron.utils.testing_utils import is_trainium_test

from ..distributed_utils import distributed_test


TINY_MODEL_NAME = "michaelbenayoun/llama-2-tiny-4kv-heads-4layers-random"

@pytest.fixture(scope="module")
def inputs():
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(
        "Paris is the most beautiful city in the world.", return_tensors="pt", padding="max_length", max_length=1024
    )
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs


@pytest.fixture(scope="module")
def train_dataset(inputs):
    dataset = datasets.Dataset.from_dict(inputs)
    dataset = dataset.select([0] * 10000)  # 10k samples
    return dataset

@is_trainium_test
@distributed_test(world_size=1, tp_size=1, pp_size=1)
def test_mixed_precision_config_invalid_fp32_grad_acc():
    with pytest.raises(
        ValueError, match="optimizer_use_fp32_grad_acc requires optimizer_use_master_weights to be True"
    ):
        MixedPrecisionConfig(
            mode="FULL_BF16",
            optimizer_use_master_weights=False,
            optimizer_use_fp32_grad_acc=True,
        )


@is_trainium_test
@distributed_test(world_size=1, tp_size=1, pp_size=1)
def test_mixed_precision_config_invalid_save_master_weights():
    with pytest.raises(
        ValueError, match="optimizer_save_master_weights_in_ckpt requires optimizer_use_master_weights to be True"
    ):
        MixedPrecisionConfig(
            mode="FULL_BF16",
            optimizer_use_master_weights=False,
            optimizer_save_master_weights_in_ckpt=True,
        )


@is_trainium_test
@distributed_test(world_size=1, tp_size=1, pp_size=1)
def test_mixed_precision_config_stochastic_rounding_env_var():
    # Test stochastic rounding enabled
    MixedPrecisionConfig(mode="FULL_BF16", stochastic_rounding=True)
    assert os.environ.get("NEURON_RT_STOCHASTIC_ROUNDING_EN") == "1"

    # Test stochastic rounding disabled
    MixedPrecisionConfig(mode="FULL_BF16", stochastic_rounding=False)
    assert os.environ.get("NEURON_RT_STOCHASTIC_ROUNDING_EN") == "0"

    # Test that NO mode disables stochastic rounding regardless of setting
    config_no_mode = MixedPrecisionConfig(mode="NO", stochastic_rounding=True)
    assert config_no_mode.stochastic_rounding is False
    assert os.environ.get("NEURON_RT_STOCHASTIC_ROUNDING_EN") == "0"


@distributed_test(world_size=8, tp_size=2, pp_size=1)
@is_trainium_test
def test_accelerator_model_preparation_fp32():
    tp_size = get_tensor_model_parallel_size()
    pp_size = get_pipeline_model_parallel_size()

    trn_config = TrainingNeuronConfig(tensor_parallel_size=tp_size, pipeline_parallel_size=pp_size)
    mixed_precision_config = MixedPrecisionConfig("NO")

    accelerator = NeuronAccelerator(trn_config=trn_config, mixed_precision_config=mixed_precision_config)

    model = NeuronLlamaForCausalLM.from_pretrained(
        TINY_MODEL_NAME,
        trn_config=trn_config,
    )

    # Model should be in float32 before preparation
    first_param = next(model.parameters())
    assert first_param.dtype == torch.float32

    prepared_model = accelerator.prepare(model)

    # Model should still be in float32 after preparation
    first_param_after = next(prepared_model.parameters())
    assert first_param_after.dtype == torch.float32


@distributed_test(world_size=8, tp_size=2, pp_size=1)
@is_trainium_test
def test_accelerator_model_preparation_full_bf16():
    tp_size = get_tensor_model_parallel_size()
    pp_size = get_pipeline_model_parallel_size()

    trn_config = TrainingNeuronConfig(tensor_parallel_size=tp_size, pipeline_parallel_size=pp_size)
    mixed_precision_config = MixedPrecisionConfig("FULL_BF16")

    accelerator = NeuronAccelerator(trn_config=trn_config, mixed_precision_config=mixed_precision_config)

    model = NeuronLlamaForCausalLM.from_pretrained(
        TINY_MODEL_NAME,
        trn_config=trn_config,
    )

    # Model should be in float32 before preparation
    first_param = next(model.parameters())
    assert first_param.dtype == torch.float32

    prepared_model = accelerator.prepare(model)

    # Model should be in bfloat16 after preparation with FULL_BF16
    first_param_after = next(prepared_model.parameters())
    assert first_param_after.dtype == torch.bfloat16


@distributed_test(world_size=8, tp_size=2, pp_size=1)
@is_trainium_test
def test_accelerator_model_preparation_autocast():
    tp_size = get_tensor_model_parallel_size()
    pp_size = get_pipeline_model_parallel_size()

    trn_config = TrainingNeuronConfig(tensor_parallel_size=tp_size, pipeline_parallel_size=pp_size)
    mixed_precision_config = MixedPrecisionConfig("AUTOCAST_BF16")

    accelerator = NeuronAccelerator(trn_config=trn_config, mixed_precision_config=mixed_precision_config)

    model = NeuronLlamaForCausalLM.from_pretrained(
        TINY_MODEL_NAME,
        trn_config=trn_config,
    )

    prepared_model = accelerator.prepare(model)

    # Model should remain in float32 for autocast mode
    first_param = next(prepared_model.parameters())
    assert first_param.dtype == torch.float32


@distributed_test(world_size=8, tp_size=2, pp_size=1)
@is_trainium_test
def test_trainer_autocast(train_dataset, inputs, tmpdir, set_cache_for_ci):
    tp_size = get_tensor_model_parallel_size()
    pp_size = get_pipeline_model_parallel_size()

    training_args = NeuronTrainingArguments(
        output_dir=tmpdir,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        bf16=True,
        use_autocast=True,
    )

    model = NeuronLlamaForCausalLM.from_pretrained(TINY_MODEL_NAME, trn_config=training_args.trn_config)

    trainer = NeuronTrainer(model, training_args, train_dataset=train_dataset)

    # Check that mixed precision works with model parallelism
    assert trainer.mixed_precision_config.mode is MixedPrecisionMode.AUTOCAST_BF16

    # Prepare the model
    prepared_model = trainer.accelerator.prepare(trainer.model)
    inputs_on_device = {k: v.to(training_args.device) for k, v in inputs.items()}

    # Test autocast context manager
    with trainer.autocast_smart_context_manager():
        outputs = prepared_model(**inputs_on_device)

    # Model should remain in float32 for autocast mode
    first_param = next(prepared_model.parameters())
    assert first_param.dtype == torch.float32

    # In AUTOCAST_BF16 mode, outputs should be in bfloat16
    assert outputs.logits.dtype == torch.bfloat16


@distributed_test(world_size=32, tp_size=2, pp_size=4)
@is_trainium_test
def test_trainer_full_bf16(train_dataset, tmpdir, set_cache_for_ci):
    tp_size = get_tensor_model_parallel_size()
    pp_size = get_pipeline_model_parallel_size()

    training_args = NeuronTrainingArguments(
        output_dir=tmpdir,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        bf16=True,
        use_autocast=False,  # Full BF16
        stochastic_rounding_enabled=True,
    )

    model = NeuronLlamaForCausalLM.from_pretrained(TINY_MODEL_NAME, trn_config=training_args.trn_config)

    trainer = NeuronTrainer(model, training_args, train_dataset=train_dataset)

    # Check that mixed precision works with model parallelism
    assert trainer.mixed_precision_config.mode is MixedPrecisionMode.FULL_BF16

    # Model should be prepared correctly
    prepared_model = trainer.accelerator.prepare(trainer.model)

    first_param = next(prepared_model.local_parameters())
    assert first_param.dtype == torch.bfloat16
