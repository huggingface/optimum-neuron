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

import json
import os
from pathlib import Path

import datasets
import pytest
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from neuronx_distributed.parallel_layers.parallel_state import (
    get_data_parallel_rank,
    get_data_parallel_size,
    get_pipeline_model_parallel_size,
    get_tensor_model_parallel_size,
)
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, BertForSequenceClassification, BertTokenizer

from optimum.neuron import NeuronTrainer, NeuronTrainingArguments
from optimum.neuron.models.training import NeuronModelForCausalLM, TrainingNeuronConfig
from optimum.neuron.models.training.training_utils import get_model_param_count
from optimum.neuron.peft import NeuronPeftModel
from optimum.neuron.utils.testing_utils import is_trainium_test

from .distributed_utils import distributed_test, run_distributed_test


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
@pytest.mark.parametrize(
    "bf16,use_autocast,expected_dtype",
    [
        (False, False, torch.float32),
        (True, True, torch.bfloat16),
    ],
    ids=["no_autocast", "bf16_autocast"],
)
def test_autocast_smart_context_manager(train_dataset, tmpdir, bf16, use_autocast, expected_dtype):
    tp_size = get_tensor_model_parallel_size()
    pp_size = get_pipeline_model_parallel_size()

    training_args_kwargs = {
        "output_dir": tmpdir,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "num_train_epochs": 3,
        "tensor_parallel_size": tp_size,
        "pipeline_parallel_size": pp_size,
        "bf16": bf16,
        "use_autocast": use_autocast,
    }

    training_args = NeuronTrainingArguments(**training_args_kwargs)
    model = NeuronModelForCausalLM.from_pretrained(TINY_MODEL_NAME, trn_config=training_args.trn_config)
    trainer = NeuronTrainer(model, training_args, train_dataset=train_dataset)

    # This will move the model to the right device.
    model = trainer.accelerator.prepare(model)

    inputs = next(iter(trainer.get_train_dataloader()))
    inputs = {k: v.to(training_args.device) for k, v in inputs.items()}

    with trainer.autocast_smart_context_manager():
        outputs = model(**inputs)

    assert outputs.logits.dtype is expected_dtype, (
        f"Expected logits to be {expected_dtype}, got {outputs.logits.dtype}"
    )


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


@is_trainium_test
@pytest.mark.parametrize("world_size, tp_size, pp_size", [[8, 2, 1], [8, 2, 4]], ids=["8_2_1", "8_2_4"])
def test_basic_training_loop(train_dataset, tmpdir, world_size, tp_size, pp_size, set_cache_for_ci):
    def test():
        tp_size = get_tensor_model_parallel_size()
        pp_size = get_pipeline_model_parallel_size()

        training_args = NeuronTrainingArguments(
            output_dir=tmpdir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            max_steps=20,
            logging_steps=1,
            save_steps=10,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            learning_rate=1e-2,
        )

        model = NeuronModelForCausalLM.from_pretrained(TINY_MODEL_NAME, trn_config=training_args.trn_config)

        # Keep a copy of the original model parameters for comparison after training.
        orig_named_parameters = dict(model.named_parameters())

        trainer = NeuronTrainer(model, training_args, train_dataset=train_dataset)

        # Verify initial state
        assert trainer.state.global_step == 0, f"Expected global_step=0 initially, got {trainer.state.global_step}"
        assert trainer.state.epoch is None, f"Expected epoch=None initially, got {trainer.state.epoch}"

        # Run training
        trainer.train()

        # Verify training completed and state updated correctly
        assert trainer.state.global_step == 20, (
            f"Expected global_step=20 after training, got {trainer.state.global_step}"
        )
        assert trainer.state.epoch > 0, f"Expected epoch>0 after training, got {trainer.state.epoch}"

        # Verify model parameters were updated (gradients applied)
        if pp_size == 1:
            trained_named_parameters = dict(trainer.model.named_parameters())
        else:
            trained_named_parameters = dict(trainer.model.local_named_parameters())

        # We get the "last" parameter, e.g. the less deep in the model, to check if it was updated.
        name = initial_param = final_param = None
        for n, param in trained_named_parameters.items():
            if param.requires_grad:
                name = n
                initial_param = orig_named_parameters[name]
                final_param = param

        assert final_param is not None, "No trainable parameters found in model after training"
        assert not torch.equal(initial_param, final_param), (
            f"Model parameters were not updated during training, tested on {name}"
        )

        # Verify training logs exist
        assert len(trainer.state.log_history) > 0, "No training logs found"

        # Check that loss was logged
        loss_logged = any("loss" in log for log in trainer.state.log_history)
        assert loss_logged, "Loss was not logged during training"

        # Validate checkpoints were saved correctly
        xm.rendezvous("wait_for_checkpoints")

        # Expected checkpoints at steps 10 and 20
        expected_checkpoints = [10, 20]

        for step in expected_checkpoints:
            checkpoint_dir = os.path.join(tmpdir, f"checkpoint-{step}")
            assert os.path.exists(checkpoint_dir), f"Checkpoint directory checkpoint-{step} was not created"

            # Validate trainer state file
            trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.json")
            assert os.path.exists(trainer_state_path), f"trainer_state.json not found in checkpoint-{step}"

            # Load and validate trainer state
            with open(trainer_state_path, "r") as f:
                state_data = json.load(f)

            assert state_data["global_step"] == step, (
                f"Expected global_step={step} in checkpoint-{step}, got {state_data['global_step']}"
            )

            # Validate training arguments file
            training_args_path = os.path.join(checkpoint_dir, "training_args.bin")
            assert os.path.exists(training_args_path), f"training_args.bin not found in checkpoint-{step}"

            # Validate scheduler file
            scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
            assert os.path.exists(scheduler_path), f"scheduler.pt not found in checkpoint-{step}"

            # Validate shards directory for distributed checkpoints
            shards_dir = os.path.join(checkpoint_dir, "shards")
            assert os.path.exists(shards_dir), f"shards directory not found in checkpoint-{step}"

            # Check for metadata files in shards directory
            metadata_files = [
                f for f in os.listdir(shards_dir) if f.startswith("mp_metadata_pp_rank_") and f.endswith(".json")
            ]
            assert len(metadata_files) > 0, f"No mp_metadata files found in checkpoint-{step}/shards/"

    run_distributed_test(test, world_size=world_size, tp_size=tp_size, pp_size=pp_size)


@is_trainium_test
@pytest.mark.parametrize("world_size, tp_size, pp_size", [[8, 2, 1]], ids=["8_2_1"])
def test_peft_training(train_dataset, tmpdir, world_size, tp_size, pp_size, set_cache_for_ci):
    def test():
        tp_size = get_tensor_model_parallel_size()
        pp_size = get_pipeline_model_parallel_size()

        training_args = NeuronTrainingArguments(
            output_dir=tmpdir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            max_steps=10,
            logging_steps=1,
            save_steps=5,  # Save at step 5 and 10
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            learning_rate=1e-3,
        )

        # Load base model
        base_model = NeuronModelForCausalLM.from_pretrained(TINY_MODEL_NAME, trn_config=training_args.trn_config)

        # Create LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,  # Low rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],  # Target attention modules
        )

        # Create PEFT model
        model = NeuronPeftModel(base_model, lora_config)
        orig_params = dict(model.named_parameters())

        # Store initial LoRA parameters for comparison
        initial_lora_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad and "lora" in name:
                initial_lora_params[name] = param

        assert len(initial_lora_params) > 0, "No LoRA parameters found in model"

        # Verify that only LoRA parameters are trainable
        trainable_params = get_model_param_count(model, trainable_only=True)
        total_params = get_model_param_count(model, trainable_only=False)

        # LoRA should significantly reduce trainable parameters
        assert trainable_params < total_params * 0.1, (
            f"LoRA should reduce trainable params significantly. Trainable: {trainable_params}, Total: {total_params}"
        )

        trainer = NeuronTrainer(model, training_args, train_dataset=train_dataset)

        # Verify initial state
        assert trainer.state.global_step == 0

        # Run training
        trainer.train()

        # Verify training completed
        assert trainer.state.global_step == 10

        # Verify LoRA parameters were updated
        final_lora_params = {name: param.to("cpu") for name, param in model.named_parameters() if "lora" in name}
        xm.mark_step()
        lora_params_changed = False
        for name in initial_lora_params:
            if not torch.equal(initial_lora_params[name], final_lora_params[name]):
                lora_params_changed = True
                break
        assert lora_params_changed, "LoRA parameters were not updated during training"

        # Verify base model parameters were NOT updated (frozen)
        base_params = {name: param.to("cpu") for name, param in model.named_parameters() if "lora" not in name}
        xm.mark_step()
        base_params_changed = False
        for name, param in base_params.items():
            if not torch.equal(orig_params[name], param):
                base_params_changed = True
                break
        assert not base_params_changed, "Base model parameters were updated, but should be frozen"

        # Verify training logs exist
        assert len(trainer.state.log_history) > 0
        loss_logged = any("loss" in log for log in trainer.state.log_history)
        assert loss_logged, "Loss was not logged during PEFT training"

        # Test PEFT model saving
        xm.rendezvous("wait_for_peft_checkpoints")

        # Validate PEFT checkpoints
        expected_checkpoints = [5, 10]

        for step in expected_checkpoints:
            checkpoint_dir = os.path.join(tmpdir, f"checkpoint-{step}")
            assert os.path.exists(checkpoint_dir), f"PEFT checkpoint directory checkpoint-{step} was not created"

            # Validate adapter-specific files exist
            shards_dir = os.path.join(checkpoint_dir, "adapter_default", "adapter_shards")
            assert os.path.exists(shards_dir), f"shards directory not found in PEFT checkpoint-{step}"

            # Verify trainer state
            trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.json")
            assert os.path.exists(trainer_state_path)

            with open(trainer_state_path, "r") as f:
                state_data = json.load(f)

            assert state_data["global_step"] == step, (
                f"Expected global_step={step} in PEFT checkpoint-{step}, got {state_data['global_step']}"
            )

    run_distributed_test(test, world_size=world_size, tp_size=tp_size, pp_size=pp_size)


@is_trainium_test
@distributed_test(world_size=8, tp_size=2, pp_size=1)
def test_neuron_trainer_error_handling(train_dataset, tmpdir, set_cache_for_ci):
    tp_size = get_tensor_model_parallel_size()
    pp_size = get_pipeline_model_parallel_size()

    base_training_args = NeuronTrainingArguments(
        output_dir=tmpdir,
        per_device_train_batch_size=1,
        max_steps=1,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
    )

    model = NeuronModelForCausalLM.from_pretrained(TINY_MODEL_NAME, trn_config=base_training_args.trn_config)

    # Test 1: eval_dataset not supported
    with pytest.raises(RuntimeError, match="Evaluation is not supported in NeuronTrainer"):
        NeuronTrainer(
            model=model,
            args=base_training_args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,  # Should raise error
        )

    # Test 2: resume_from_checkpoint not supported
    trainer = NeuronTrainer(model, base_training_args, train_dataset=train_dataset)
    with pytest.raises(ValueError, match="`resume_from_checkpoint` is not supported"):
        trainer.train(resume_from_checkpoint=True)

    # Test 3: Liger kernel not supported
    liger_args = NeuronTrainingArguments(
        output_dir=tmpdir,
        per_device_train_batch_size=1,
        max_steps=1,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        use_liger_kernel=True,
    )
    with pytest.raises(RuntimeError, match="Liger kernel is not supported"):
        NeuronTrainer(model, liger_args, train_dataset=train_dataset)

    # Test 4: No model provided
    with pytest.raises(ValueError, match="A model must be provided to the NeuronTrainer"):
        NeuronTrainer(None, base_training_args, train_dataset=train_dataset)

    # Test 5: Conflicting optimizer arguments
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer_cls_and_kwargs = (torch.optim.SGD, {"lr": 1e-2})

    with pytest.raises(RuntimeError, match="Passing both `optimizers` and `optimizer_cls_and_kwargs`"):
        NeuronTrainer(
            model,
            base_training_args,
            train_dataset=train_dataset,
            optimizers=(optimizer, None),
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
        )

    # Test 6: Dataset without length and no max_steps
    class DatasetWithoutLength:
        def __iter__(self):
            return iter([])

    dataset_no_length = DatasetWithoutLength()

    no_max_steps_args = NeuronTrainingArguments(
        output_dir=tmpdir,
        per_device_train_batch_size=1,
        num_train_epochs=1,  # No max_steps set
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
    )

    with pytest.raises(ValueError, match="max_steps has to be specified"):
        NeuronTrainer(model, no_max_steps_args, train_dataset=dataset_no_length)


@is_trainium_test
@distributed_test(world_size=8, tp_size=2, pp_size=1)
def test_iterable_dataset_training(tmpdir, set_cache_for_ci):
    dp_size = get_data_parallel_size()
    tp_size = get_tensor_model_parallel_size()
    pp_size = get_pipeline_model_parallel_size()

    dp_rank = get_data_parallel_rank()

    class SimpleIterableDataset(torch.utils.data.IterableDataset):
        def __init__(self, num_samples=50):
            self.num_samples = num_samples
            self.examples = {
                rank: {
                    "input_ids": rank * torch.ones((1024,), dtype=torch.int),
                    "attention_mask": torch.ones((1024,), dtype=torch.int),
                    "labels": rank * torch.ones((1024,), dtype=torch.int),
                }
                for rank in range(dp_size)
            }

        def __iter__(self):
            for _ in range(self.num_samples):
                yield self.examples[dp_rank]

    iterable_dataset = SimpleIterableDataset()

    training_args = NeuronTrainingArguments(
        output_dir=tmpdir,
        per_device_train_batch_size=1,
        max_steps=10,  # Required for IterableDataset
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        logging_steps=1,
    )
    model = NeuronModelForCausalLM.from_pretrained(TINY_MODEL_NAME, trn_config=training_args.trn_config)

    trainer = NeuronTrainer(model, training_args, train_dataset=iterable_dataset)

    # Verify initial state
    assert trainer.state.global_step == 0, f"Expected initial global_step=0, got {trainer.state.global_step}"

    # Run training
    trainer.train()

    # Verify training completed with correct step count
    assert trainer.state.global_step == 10, f"Expected final global_step=10, got {trainer.state.global_step}"
    assert len(trainer.state.log_history) > 0, (
        f"Expected training logs to be recorded, got {len(trainer.state.log_history)} log entries"
    )


@is_trainium_test
@distributed_test(world_size=2, tp_size=1, pp_size=1)
def test_no_parallelism_bert_training(tmpdir, set_cache_for_ci):
    tp_size = get_tensor_model_parallel_size()
    pp_size = get_pipeline_model_parallel_size()

    model_name = "hf-internal-testing/tiny-random-BertForSequenceClassification"

    # Create sequence classification dataset
    tokenizer = BertTokenizer.from_pretrained(model_name)
    texts = [
        "This is a positive example.",
        "This is a negative example.",
        "Another positive text.",
        "Another negative text.",
    ] * 10  # 40 examples total

    labels = [1, 0, 1, 0] * 10  # Binary classification labels

    # Tokenize inputs
    encoded = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=128)

    # Create dataset
    dataset_dict = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    dataset = datasets.Dataset.from_dict(dataset_dict)

    # Training arguments without model parallelism
    training_args = NeuronTrainingArguments(
        output_dir=tmpdir,
        per_device_train_batch_size=2,
        # This way we also "test" gradient accumulation without `num_items_per_batch`.
        # We do not test here that normalization is correct (it should), but that it runs fine.
        gradient_accumulation_steps=2,
        max_steps=5,
        logging_steps=1,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        learning_rate=5e-4,
    )

    # Use regular transformers BERT model
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Store initial parameters for comparison
    initial_params = dict(model.named_parameters())

    trainer = NeuronTrainer(model, training_args, train_dataset=dataset)

    # Verify initial state
    assert trainer.state.global_step == 0, f"Expected initial global_step=0, got {trainer.state.global_step}"

    # Run training
    trainer.train()

    # Verify training completed
    assert trainer.state.global_step == 5, f"Expected final global_step=5, got {trainer.state.global_step}"
    assert len(trainer.state.log_history) > 0, (
        f"Expected training logs to be recorded, got {len(trainer.state.log_history)} log entries"
    )

    # Verify loss was logged
    loss_logged = any("loss" in log for log in trainer.state.log_history)
    assert loss_logged, f"Loss was not logged during BERT training. Log history: {trainer.state.log_history}"

    # Verify parameters were updated
    params_changed = False
    final_params = {name: param.to("cpu") for name, param in model.named_parameters()}
    xm.mark_step()
    for name, param in final_params.items():
        if not torch.equal(initial_params[name], param):
            params_changed = True
            break
    assert params_changed, f"Model parameters were not updated during training. Checked {len(final_params)} parameters"


@distributed_test(8, 2, 1)
@is_trainium_test
@pytest.mark.parametrize("use_async_save", [True, False], ids=["async_save", "sync_save"])
def test_save_during_training(use_async_save, train_dataset, tmpdir, set_cache_for_ci):
    tp_size = get_tensor_model_parallel_size()
    pp_size = get_pipeline_model_parallel_size()

    tmpdir = Path(tmpdir)

    # Configure training with async save
    trn_config = TrainingNeuronConfig(
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        async_save=use_async_save,
        use_xser=False,
    )

    # Load tokenizer & model
    model_name = TINY_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = NeuronModelForCausalLM.from_pretrained(model_name, trn_config)

    # Configure training arguments for frequent saving
    training_args = NeuronTrainingArguments(
        output_dir=str(tmpdir / "save_test"),
        max_steps=20,
        save_steps=1,  # Save at every step - this will test async behavior
        save_strategy="steps",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        logging_steps=5,
        bf16=True,
        report_to="none",  # Disable wandb/tensorboard
    )

    # Create trainer
    trainer = NeuronTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    # Run training - this should complete without hanging on save operations
    trainer.train()

    # Verify that checkpoints were created
    output_dir = Path(training_args.output_dir)
    if xr.global_ordinal() == 0:  # Only check on the main process
        # Check that checkpoint directories exist for each step
        expected_checkpoints = [f"checkpoint-{i}" for i in range(1, 21)]  # steps 1-20

        for checkpoint_name in expected_checkpoints:
            checkpoint_dir = output_dir / checkpoint_name
            assert checkpoint_dir.exists(), f"Checkpoint {checkpoint_name} was not saved"

            # Verify that the checkpoint contains model files
            # At minimum, should contain some model files (exact files depend on save format)
            checkpoint_files = list(checkpoint_dir.glob("*"))
            assert len(checkpoint_files) > 0, f"Checkpoint {checkpoint_name} is empty"

    xm.rendezvous("Async save test completed.")
