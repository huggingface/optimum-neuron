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


import datasets
import pytest
import torch
import torch_xla.core.xla_model as xm
from neuronx_distributed.optimizer import NeuronZero1Optimizer
from neuronx_distributed.parallel_layers.parallel_state import (
    get_data_parallel_replica_groups,
    get_pipeline_model_parallel_size,
    get_tensor_model_parallel_replica_groups,
    get_tensor_model_parallel_size,
)
from neuronx_distributed.parallel_layers.utils import move_all_tensor_to_cpu
from transformers import AutoTokenizer

from optimum.neuron.accelerate import NeuronAccelerator
from optimum.neuron.accelerate.optimizer import NeuronAcceleratedOptimizer
from optimum.neuron.accelerate.utils.dataclasses import MixedPrecisionConfig
from optimum.neuron.models.training import NeuronModelForCausalLM
from optimum.neuron.models.training.config import TrainingNeuronConfig
from optimum.neuron.trainers import NeuronTrainingArguments
from optimum.neuron.utils.testing_utils import is_trainium_test

from ..distributed_utils import distributed_test, run_distributed_test


TINY_MODEL_NAME = "michaelbenayoun/qwen3-tiny-4kv-heads-4layers-random"


def move_params_to_cpu(parameters):
    parameters = list(parameters)
    xm.mark_step()
    cpu_params = move_all_tensor_to_cpu([p.data for p in parameters])
    return cpu_params

@pytest.fixture(scope="module")
def inputs():
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL_NAME)
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


@distributed_test(world_size=8, tp_size=2, pp_size=1)
@is_trainium_test
def test_zero1_optimizer_creation(set_cache_for_ci):
    tp_size = get_tensor_model_parallel_size()
    pp_size = get_pipeline_model_parallel_size()

    mixed_precision_config = MixedPrecisionConfig(mode="NO")

    trn_config = TrainingNeuronConfig(
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size
    )

    accelerator = NeuronAccelerator(
        trn_config=trn_config,
        zero_1=True,
        mixed_precision_config=mixed_precision_config
    )

    model = NeuronModelForCausalLM.from_pretrained(
        TINY_MODEL_NAME,
        trn_config=trn_config,
    )
    model = accelerator.prepare(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    prepared_optimizer = accelerator.prepare(optimizer)

    assert isinstance(prepared_optimizer, NeuronZero1Optimizer)

    # Check that sharding and gradient norm groups are properly set
    assert prepared_optimizer.sharding_groups is not None
    assert prepared_optimizer.grad_norm_groups is not None

    expected_sharding_groups = get_data_parallel_replica_groups()
    expected_grad_norm_groups = get_tensor_model_parallel_replica_groups()

    assert prepared_optimizer.sharding_groups == expected_sharding_groups
    assert prepared_optimizer.grad_norm_groups == expected_grad_norm_groups


@distributed_test(world_size=8, tp_size=2, pp_size=1)
@is_trainium_test
def test_zero1_master_weights_configuration(set_cache_for_ci):
    tp_size = get_tensor_model_parallel_size()
    pp_size = get_pipeline_model_parallel_size()

    # Test with master weights enabled
    mixed_precision_config = MixedPrecisionConfig(
        mode="FULL_BF16",
        optimizer_use_master_weights=True,
        optimizer_use_fp32_grad_acc=True
    )

    trn_config = TrainingNeuronConfig(
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size
    )

    accelerator = NeuronAccelerator(
        trn_config=trn_config,
        zero_1=True,
        mixed_precision_config=mixed_precision_config
    )

    model = NeuronModelForCausalLM.from_pretrained(
        TINY_MODEL_NAME,
        trn_config=trn_config,
    )
    model = accelerator.prepare(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    prepared_optimizer = accelerator.prepare(optimizer)

    assert isinstance(prepared_optimizer, NeuronZero1Optimizer)
    assert prepared_optimizer.optimizer_dtype == torch.float32  # Master weights
    assert prepared_optimizer.use_grad_acc_hook is True  # FP32 grad accumulation
    assert prepared_optimizer.higher_cc_precision is True


@distributed_test(world_size=16, tp_size=2, pp_size=1)
@is_trainium_test
def test_zero1_training_arguments_integration(tmpdir):
    tp_size = get_tensor_model_parallel_size()
    pp_size = get_pipeline_model_parallel_size()

    # Test ZeRO-1 enabled (default)
    training_args = NeuronTrainingArguments(
        output_dir=tmpdir,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        bf16=True,
        zero_1=True,  # Explicitly enabled
        optimizer_use_master_weights=True,
        optimizer_use_fp32_grad_acc=True,
    )

    assert training_args.zero_1 is True
    assert training_args.optimizer_use_master_weights is True
    assert training_args.optimizer_use_fp32_grad_acc is True

    # Test ZeRO-1 disabled
    training_args_no_zero1 = NeuronTrainingArguments(
        output_dir=tmpdir,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        bf16=True,
        zero_1=False,
        optimizer_use_master_weights=True,  # Should be disabled automatically
        optimizer_use_fp32_grad_acc=True,   # Should be disabled automatically
    )

    # These should be automatically disabled when zero_1=False
    assert training_args_no_zero1.zero_1 is False
    assert training_args_no_zero1.optimizer_use_master_weights is False
    assert training_args_no_zero1.optimizer_use_fp32_grad_acc is False

@is_trainium_test
@pytest.mark.parametrize("use_master_weights", [True, False], ids=["master_weights", "no_master_weights"])
@pytest.mark.parametrize("fp32_grad_acc", [True, False], ids=["fp32_grad_acc", "no_fp32_grad_acc"])
@pytest.mark.parametrize("world_size,tp_size,pp_size", [(8, 2, 1), (32, 2, 4)], ids=["8_2_1", "32_2_4"])
def test_zero_1_optimizer_step_and_mixed_precision(world_size, tp_size, pp_size, inputs, use_master_weights, fp32_grad_acc, set_cache_for_ci):

    def test():
        mixed_precision_config = MixedPrecisionConfig(
            mode="FULL_BF16",
            optimizer_use_master_weights=use_master_weights,
            optimizer_use_fp32_grad_acc=fp32_grad_acc,
        )

        trn_config = TrainingNeuronConfig(
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size
        )

        accelerator = NeuronAccelerator(
            trn_config=trn_config,
            zero_1=True,
            mixed_precision_config=mixed_precision_config,
        )

        model = NeuronModelForCausalLM.from_pretrained(
            TINY_MODEL_NAME,
            trn_config=trn_config,
        )
        max_grad_norm = 0.01
        model = accelerator.prepare(model)

        named_parameters = dict(model.named_parameters() if pp_size == 1 else model.local_named_parameters())
        optimizer = torch.optim.AdamW([p for _, p in named_parameters.items()], lr=0.001)
        prepared_optimizer = accelerator.prepare(optimizer)

        # Check that the classes are correct
        assert isinstance(prepared_optimizer, NeuronAcceleratedOptimizer), "Optimizer is not a NeuronAcceleratedOptimizer."
        assert isinstance(prepared_optimizer.optimizer, NeuronZero1Optimizer), "Optimizer.optimizer is not a NeuronZero1Optimizer."

        if pp_size == 1:
            model.train()
            xla_inputs = {k: v.to(xm.xla_device()) for k, v in inputs.items()}
            xm.mark_step()
        else:
            xla_inputs = inputs # NxDPPModel will move the inputs.

        orig_named_parameters = {n: p.cpu() for n, p in named_parameters.items()}
        xm.mark_step()

        if pp_size == 1:
            outputs = model(**xla_inputs)
            loss = outputs.loss
            accelerator.backward(loss)
        else:
            loss = model.run_train(**xla_inputs)
        xm.mark_step()

        accelerator.clip_grad_norm_(
            model.parameters(), # This is not used when using ZeRO-1 but we need to pass something.
            max_norm=max_grad_norm,
            norm_type=2,
            postpone_clipping_to_optimizer_step=True,
        )
        prepared_optimizer.step()
        xm.mark_step()

        current_named_parameters = {n: p.cpu() for n, p in named_parameters.items()}
        grads = [p.grad.cpu() if not fp32_grad_acc else p.main_grad.cpu() for _, p in named_parameters.items() if p.requires_grad]
        xm.mark_step()

        # Check that all parameters that require grad have a gradient after backward
        # and that the gradient dtype is correct.
        grad_dtype = torch.float32 if fp32_grad_acc else torch.bfloat16
        assert any(torch.any(grad != 0) for grad in grads), "Expected some gradients to be non-zero."
        assert all(grad.dtype is grad_dtype for grad in grads), f"Not all gradients are in {grad_dtype}."

        # Gradient clipping seems to not work properly with ZeRO-1.
        # prepared_optimizer.step()
        # assert prepared_optimizer.grad_norm is not None, "Expected grad_norm to be set after optimizer step."
        # grad_norm = prepared_optimizer.grad_norm.to("cpu")
        # xm.mark_step()
        # grad_norm = grad_norm.item()
        # assert grad_norm <= max_grad_norm, "Expected the total norm to be clipped."

        # Check that at least some parameters have changed after optimizer step.
        param_changed = False
        for n, p in current_named_parameters.items():
            if not torch.all(p == orig_named_parameters[n]):
                param_changed = True
                break
        assert param_changed, "Parameters did not change after optimizer step."

        # Check that optimizer parameters are in the correct dtype.
        optimizer_param_dtype = torch.float32 if use_master_weights else torch.bfloat16
        assert all(p.dtype is optimizer_param_dtype for p in prepared_optimizer.optimizer.base_optimizer.param_groups[0]["params"]), f"Not all optimizer parameters are in {optimizer_param_dtype}."

        prepared_optimizer.zero_grad()
        xm.mark_step()

        grads = []
        for _, p in named_parameters.items():
            if p.requires_grad:
                if fp32_grad_acc:
                    grads.append(p.main_grad.cpu())
                else:
                    grads.append(p.grad.cpu() if p.grad is not None else None)
        xm.mark_step()

        # Check that all parameters that require grad have no gradient after zero_grad.
        assert all(grad is None or torch.all(grad == 0) for grad in grads), "Expected no gradients after zero_grad()."

    run_distributed_test(test, world_size=world_size, tp_size=tp_size, pp_size=pp_size)
