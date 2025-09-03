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

import datasets
import pytest
import torch
import torch_xla.core.xla_model as xm
from neuronx_distributed.parallel_layers.parallel_state import (
    get_pipeline_model_parallel_size,
    get_tensor_model_parallel_size,
)
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer
from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES
from transformers.trainer_pt_utils import AcceleratorConfig

from optimum.neuron import NeuronTrainer, NeuronTrainingArguments
from optimum.neuron.models.training import NeuronModelForCausalLM
from optimum.neuron.peft import NeuronPeftModel
from optimum.neuron.utils.testing_utils import is_trainium_test
from optimum.neuron.utils.training_utils import get_model_param_count

from ..distributed_utils import distributed_test, run_distributed_test


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
    with pytest.raises(ValueError, match="A model must be provided to the Trainer"):
        NeuronTrainer(None, base_training_args, train_dataset=train_dataset)

    # Test 5: Invalid model class (MODEL_MAPPING_NAMES)
    # Create a mock model with a name that should be in MODEL_MAPPING_NAMES
    class MockInvalidModel:
        def __init__(self):
            # Pick the first available model name from MODEL_MAPPING_NAMES
            self.__class__.__name__ = list(MODEL_MAPPING_NAMES.keys())[0]

    mock_model = MockInvalidModel()
    with pytest.raises(ValueError, match="cannot be used as is for training"):
        NeuronTrainer(mock_model, base_training_args, train_dataset=train_dataset)

    # Test 6: Conflicting optimizer arguments
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

    # Test 7: Device mismatch between model and optimizer
    # Move model to XLA device first
    model.to(xm.xla_device())

    # Create optimizer on CPU (different device)
    cpu_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Move optimizer params back to CPU to create mismatch
    for param_group in cpu_optimizer.param_groups:
        for param in param_group["params"]:
            if param.device.type == "xla":
                param.data = param.data.cpu()

    with pytest.raises(ValueError, match="model and the optimizer parameters are not on the same device"):
        NeuronTrainer(model, base_training_args, train_dataset=train_dataset, optimizers=(cpu_optimizer, None))

    # Test 8: Dataset without length and no max_steps
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

    # Test 9: Unsupported AcceleratorConfig options (should generate warnings, not errors)

    unsupported_config = AcceleratorConfig(
        split_batches=True,  # Should warn
        dispatch_batches=True,  # Should warn
        even_batches=True,  # Should warn
        use_seedable_sampler=True,  # Should warn
    )

    warn_args = NeuronTrainingArguments(
        output_dir=tmpdir,
        per_device_train_batch_size=1,
        max_steps=1,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        accelerator_config=unsupported_config,
    )

    # This should create warnings but not fail
    with pytest.warns():
        trainer_with_warnings = NeuronTrainer(model, warn_args, train_dataset=train_dataset)
        # Verify the warnings were handled and config was modified
        assert trainer_with_warnings is not None


@is_trainium_test
@distributed_test(world_size=8, tp_size=2, pp_size=1)
def test_iterable_dataset_training(tmpdir, set_cache_for_ci):
    tp_size = get_tensor_model_parallel_size()
    pp_size = get_pipeline_model_parallel_size()

    class SimpleIterableDataset(torch.utils.data.IterableDataset):
        def __init__(self, num_samples=50):
            self.num_samples = num_samples

        def __iter__(self):
            for i in range(self.num_samples):
                yield {
                    "input_ids": torch.randint(1, 1000, (1024,), dtype=torch.long),
                    "attention_mask": torch.ones(1024, dtype=torch.long),
                    "labels": torch.randint(1, 1000, (1024,), dtype=torch.long),
                }

    iterable_dataset = SimpleIterableDataset()

    training_args = NeuronTrainingArguments(
        output_dir=tmpdir,
        per_device_train_batch_size=1,
        max_steps=5,  # Required for IterableDataset
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        logging_steps=1,
    )
    model = NeuronModelForCausalLM.from_pretrained(TINY_MODEL_NAME, trn_config=training_args.trn_config)

    trainer = NeuronTrainer(model, training_args, train_dataset=iterable_dataset)

    # Verify initial state
    assert trainer.state.global_step == 0

    # Run training
    trainer.train()

    # Verify training completed with correct step count
    assert trainer.state.global_step == 5
    assert len(trainer.state.log_history) > 0
