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
from neuronx_distributed.parallel_layers.parallel_state import (
    get_pipeline_model_parallel_size,
    get_tensor_model_parallel_size,
)
from transformers import AutoTokenizer

from optimum.neuron import NeuronTrainer, NeuronTrainingArguments
from optimum.neuron.models.training import NeuronModelForCausalLM

from ..distributed_utils import distributed_test


TINY_MODEL_NAME = "michaelbenayoun/qwen3-tiny-4kv-heads-4layers-random"


@pytest.fixture(scope="module")
def train_dataset():
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL_NAME)
    inputs = tokenizer(
        "Paris is the most beautiful city in the world.", return_tensors="pt", padding="max_length", max_length=1024
    )
    inputs["labels"] = inputs["input_ids"].clone()
    dataset = datasets.Dataset.from_dict(inputs)
    dataset = dataset.select([0] * 10000)  # 10k samples
    return dataset


@distributed_test(
    world_size=8,
    tp_size=2,
    pp_size=1,
)
def test_num_examples_and_num_tokens(train_dataset, tmpdir):
    tp_size = get_tensor_model_parallel_size()
    pp_size = get_pipeline_model_parallel_size()
    training_args = NeuronTrainingArguments(
        output_dir=tmpdir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=3,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
    )
    model = NeuronModelForCausalLM.from_pretrained(TINY_MODEL_NAME, trn_config=training_args.trn_config)

    trainer = NeuronTrainer(model, training_args, train_dataset=train_dataset)
    train_dataloader = trainer.get_train_dataloader()

    num_examples = trainer.num_examples(train_dataloader)
    assert num_examples == 10000, f"Expected 10000 examples, got {num_examples}"

    num_tokens = trainer.num_tokens(train_dataloader)
    assert num_tokens == 10000 * 1024, f"Expected {10000 * 1024} tokens, got {num_tokens}"


@distributed_test(
    world_size=8,
    tp_size=2,
    pp_size=1,
)
def test_autocast_smart_context_manager_no_autocast(train_dataset, tmpdir):
    tp_size = get_tensor_model_parallel_size()
    pp_size = get_pipeline_model_parallel_size()

    training_args = NeuronTrainingArguments(
        output_dir=tmpdir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=3,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
    )
    model = NeuronModelForCausalLM.from_pretrained(TINY_MODEL_NAME, trn_config=training_args.trn_config)
    trainer = NeuronTrainer(model, training_args, train_dataset=train_dataset)

    # This will move the model to the right device.
    model = trainer.accelerator.prepare(model)

    inputs = next(iter(trainer.get_train_dataloader()))
    inputs = {k: v.to(training_args.device) for k, v in inputs.items()}

    with trainer.autocast_smart_context_manager():
        outputs = model(**inputs)

    assert outputs.logits.dtype is torch.float32, f"Expected logits to be float32, got {outputs.logits.dtype}"


@distributed_test(
    world_size=8,
    tp_size=2,
    pp_size=1,
)
def test_autocast_smart_context_manager_enabled(train_dataset, tmpdir):
    tp_size = get_tensor_model_parallel_size()
    pp_size = get_pipeline_model_parallel_size()

    training_args = NeuronTrainingArguments(
        output_dir=tmpdir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=3,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        bf16=True,
        use_autocast=True,
    )

    model = NeuronModelForCausalLM.from_pretrained(TINY_MODEL_NAME, trn_config=training_args.trn_config)
    trainer = NeuronTrainer(model, training_args, train_dataset=train_dataset)

    # This will move the model to the right device.
    model = trainer.accelerator.prepare(model)

    inputs = next(iter(trainer.get_train_dataloader()))
    inputs = {k: v.to(training_args.device) for k, v in inputs.items()}

    with trainer.autocast_smart_context_manager():
        outputs = model(**inputs)

    assert outputs.logits.dtype is torch.bfloat16, f"Expected logits to be bfloat16, got {outputs.logits.dtype}"


@distributed_test(
    world_size=8,
    tp_size=2,
    pp_size=1,
)
def test_set_initial_training_values(train_dataset, tmpdir):
    tp_size = get_tensor_model_parallel_size()
    pp_size = get_pipeline_model_parallel_size()

    # Case 1: no max_steps, epoch based training
    training_args = NeuronTrainingArguments(
        output_dir=tmpdir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=3,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
    )

    model = NeuronModelForCausalLM.from_pretrained(TINY_MODEL_NAME, trn_config=training_args.trn_config)
    trainer = NeuronTrainer(model, training_args, train_dataset=train_dataset)

    total_batch_size = training_args.train_batch_size * training_args.gradient_accumulation_steps

    train_dataloader = trainer.get_train_dataloader()
    (
        num_train_epochs,
        num_update_steps_per_epoch,
        num_examples,
        num_train_samples,
        epoch_based,
        len_dataloader,
        max_steps,
    ) = trainer.set_initial_training_values(training_args, train_dataloader, total_batch_size)

    expected_num_update_steps_per_epoch = (10000 + total_batch_size - 1) // total_batch_size

    assert num_train_epochs == 3, f"Expected 3 epochs, got {num_train_epochs}"
    assert num_update_steps_per_epoch == expected_num_update_steps_per_epoch, (
        f"Expected {expected_num_update_steps_per_epoch} update steps per epoch, got {num_update_steps_per_epoch}"
    )

    assert num_examples == 10000, f"Expected 10000 examples, got {num_examples}"
    assert num_train_samples == 10000 * 3, f"Expected 30000 train samples, got {num_train_samples}"
    assert epoch_based is True, f"Expected epoch_based to be True, got {epoch_based}"
    assert len_dataloader == 10000 // training_args.train_batch_size, (
        f"Expected dataloader length to be {10000 // training_args.train_batch_size}, got {len_dataloader}"
    )
    assert max_steps == 3 * expected_num_update_steps_per_epoch, (
        f"Expected max_steps to be {3 * expected_num_update_steps_per_epoch}, got {max_steps}"
    )

    # Case 2: max_steps
    steps = 500
    training_args = NeuronTrainingArguments(
        output_dir=tmpdir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=3,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        max_steps=steps,
    )

    model = NeuronModelForCausalLM.from_pretrained(TINY_MODEL_NAME, trn_config=training_args.trn_config)
    trainer = NeuronTrainer(model, training_args, train_dataset=train_dataset)

    total_batch_size = training_args.train_batch_size * training_args.gradient_accumulation_steps

    train_dataloader = trainer.get_train_dataloader()
    (
        num_train_epochs,
        num_update_steps_per_epoch,
        num_examples,
        num_train_samples,
        epoch_based,
        len_dataloader,
        max_steps,
    ) = trainer.set_initial_training_values(training_args, train_dataloader, total_batch_size)

    expected_num_train_epochs = steps // expected_num_update_steps_per_epoch + int(
        steps % expected_num_update_steps_per_epoch > 0
    )
    assert num_train_epochs == expected_num_train_epochs, (
        f"Expected {expected_num_train_epochs} epochs, got {num_train_epochs}"
    )
    assert num_update_steps_per_epoch == expected_num_update_steps_per_epoch, (
        f"Expected {expected_num_update_steps_per_epoch} update steps per epoch, got {num_update_steps_per_epoch}"
    )

    assert num_examples == 10000, f"Expected 10000 examples, got {num_examples}"
    assert num_train_samples == total_batch_size * steps, (
        f"Expected {total_batch_size * steps}  train samples, got {num_train_samples}"
    )
    assert epoch_based is False, f"Expected epoch_based to be False, got {epoch_based}"
    assert len_dataloader == 10000 // training_args.train_batch_size, (
        f"Expected dataloader length to be {10000 // training_args.train_batch_size}, got {len_dataloader}"
    )
    assert max_steps == steps, f"Expected max_steps to be {steps}, got {max_steps}"


@distributed_test(
    world_size=8,
    tp_size=2,
    pp_size=1,
)
def test_basic_training_loop(train_dataset, tmpdir):
    tp_size = get_tensor_model_parallel_size()
    pp_size = get_pipeline_model_parallel_size()

    training_args = NeuronTrainingArguments(
        output_dir=tmpdir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        max_steps=2,  # Just 2 training steps for fast execution
        logging_steps=1,
        save_steps=10,  # Don't save during this short test
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        learning_rate=1e-2,
    )

    model = NeuronModelForCausalLM.from_pretrained(TINY_MODEL_NAME, trn_config=training_args.trn_config)
    trainer = NeuronTrainer(model, training_args, train_dataset=train_dataset)

    # Get initial model parameter to check it changes
    initial_param = None
    for param in trainer.model.parameters():
        if param.requires_grad:
            initial_param = param.data.clone()

    assert initial_param is not None, "No trainable parameters found in model"

    # Verify initial state
    assert trainer.state.global_step == 0, f"Expected global_step=0 initially, got {trainer.state.global_step}"
    assert trainer.state.epoch is None, f"Expected epoch=None initially, got {trainer.state.epoch}"

    # Run training
    trainer.train()

    # Verify training completed and state updated correctly
    assert trainer.state.global_step == 2, f"Expected global_step=2 after training, got {trainer.state.global_step}"
    assert trainer.state.epoch > 0, f"Expected epoch>0 after training, got {trainer.state.epoch}"

    # Verify model parameters were updated (gradients applied)
    final_param = None
    for param in trainer.model.parameters():
        if param.requires_grad:
            final_param = param.data

    if final_param is not None:
        final_param = final_param.cpu()

    assert final_param is not None, "No trainable parameters found in model after training"
    assert not torch.equal(initial_param, final_param), "Model parameters were not updated during training"

    # Verify training logs exist
    assert len(trainer.state.log_history) > 0, "No training logs found"

    # Check that loss was logged
    loss_logged = any("loss" in log for log in trainer.state.log_history)
    assert loss_logged, "Loss was not logged during training"
