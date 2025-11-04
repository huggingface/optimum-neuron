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

import pytest
import torch
import torch_xla.core.xla_model as xm
from datasets import load_dataset
from neuronx_distributed.parallel_layers.parallel_state import (
    get_pipeline_model_parallel_size,
    get_tensor_model_parallel_size,
)
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer

from optimum.neuron import NeuronSFTConfig, NeuronSFTTrainer, NeuronTrainingArguments
from optimum.neuron.models.training import LlamaForCausalLM as NeuronLlamaForCausalLM
from optimum.neuron.models.training.training_utils import get_model_param_count
from optimum.neuron.utils.testing_utils import is_trainium_test

from .distributed_utils import distributed_test, run_distributed_test
from .utils import MODEL_NAME


@is_trainium_test
@pytest.mark.parametrize(
    "world_size, tp_size, pp_size",
    [[2, 1, 1], [2, 2, 1]],
    ids=["dp=2", "tp=2"],
)
@pytest.mark.parametrize(
    "packing",
    [True, False],
    ids=["packing", "no_packing"],
)
def test_neuron_sft_trainer_basic_training_loop(world_size, tp_size, pp_size, packing, tmpdir, set_cache_for_ci):
    def test():
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

        def format_dolly(sample):
            instruction = f"### Instruction\n{sample['instruction']}"
            context = f"### Context\n{sample['context']}" if len(sample["context"]) > 0 else None
            response = f"### Answer\n{sample['response']}"
            prompt = "\n\n".join([i for i in [instruction, context, response] if i is not None])
            return prompt

        args = NeuronTrainingArguments(
            output_dir=str(tmpdir),
            do_train=True,
            max_steps=10,
            per_device_train_batch_size=1,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            bf16=True,
            logging_steps=1,
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # to prevent warnings

        model = NeuronLlamaForCausalLM.from_pretrained(MODEL_NAME, args.trn_config)

        args = args.to_dict()
        sft_config = NeuronSFTConfig(
            # Using a small sequence-length since we are not validating the outputs.
            max_length=128,
            packing=packing,
            dataset_num_proc=1,
            **args,
        )

        # Create Trainer instance
        trainer = NeuronSFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            formatting_func=format_dolly,
            args=sft_config,
        )

        # Store initial parameters for comparison
        initial_params = dict(model.named_parameters())

        # Verify initial state
        assert trainer.state.global_step == 0, f"Expected initial global_step=0, got {trainer.state.global_step}"

        # Verify that all inputs are padded to max_length
        sample_batch = next(iter(trainer.get_train_dataloader()))
        assert sample_batch["input_ids"].shape[1] == sft_config.max_length, (
            f"Expected input_ids to have length {sft_config.max_length}, got {sample_batch['input_ids'].shape[1]}"
        )
        assert sample_batch["labels"].shape[1] == sft_config.max_length, (
            f"Expected labels to have length {sft_config.max_length}, got {sample_batch['labels'].shape[1]}"
        )
        assert sample_batch["attention_mask"].shape[1] == sft_config.max_length, (
            f"Expected attention_mask to have length {sft_config.max_length}, got {sample_batch['attention_mask'].shape[1]}"
        )

        # Run training
        trainer.train()

        # Verify training completed
        assert trainer.state.global_step == 10, f"Expected final global_step=10, got {trainer.state.global_step}"

        # Verify training logs exist
        assert len(trainer.state.log_history) > 0, (
            f"Expected training logs to be recorded, got {len(trainer.state.log_history)} log entries"
        )

        # Verify loss was logged
        loss_logged = any("loss" in log for log in trainer.state.log_history)
        assert loss_logged, f"Loss was not logged during SFT training. Log history: {trainer.state.log_history}"

        final_params = {name: param.to("cpu") for name, param in model.named_parameters()}
        xm.mark_step()

        params_changed = False
        for name, final_param in final_params.items():
            if name in initial_params and not torch.equal(initial_params[name], final_param):
                params_changed = True
                break
        assert params_changed, (
            f"Model parameters were not updated during SFT training. Checked {len(final_params)} parameters"
        )

    run_distributed_test(test, world_size=world_size, tp_size=tp_size, pp_size=pp_size)


@is_trainium_test
@distributed_test(world_size=8, tp_size=2, pp_size=1)
def test_neuron_sft_trainer_peft_training(tmpdir, set_cache_for_ci):
    tp_size = get_tensor_model_parallel_size()
    pp_size = get_pipeline_model_parallel_size()

    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

    def format_dolly(sample):
        instruction = f"### Instruction\n{sample['instruction']}"
        context = f"### Context\n{sample['context']}" if len(sample["context"]) > 0 else None
        response = f"### Answer\n{sample['response']}"
        prompt = "\n\n".join([i for i in [instruction, context, response] if i is not None])
        return [prompt]  # No packing for simplicity

    args = NeuronTrainingArguments(
        output_dir=str(tmpdir),
        do_train=True,
        max_steps=5,  # Shorter for PEFT test
        per_device_train_batch_size=1,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        bf16=True,
        logging_steps=1,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load base model
    base_model = NeuronLlamaForCausalLM.from_pretrained(MODEL_NAME, args.trn_config)

    # Create LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )

    args = args.to_dict()
    sft_config = NeuronSFTConfig(
        max_length=128,
        packing=False,  # No packing for PEFT test simplicity
        dataset_num_proc=1,
        **args,
    )

    # Create SFT Trainer instance with PEFT model
    trainer = NeuronSFTTrainer(
        model=base_model,
        processing_class=tokenizer,
        train_dataset=dataset,
        formatting_func=format_dolly,
        args=sft_config,
        peft_config=lora_config,
    )

    # Verify that only LoRA parameters are trainable
    trainable_params = get_model_param_count(trainer.model, trainable_only=True)
    total_params = get_model_param_count(trainer.model, trainable_only=False)
    assert trainable_params < total_params * 0.1, (
        f"LoRA should reduce trainable params significantly. Trainable: {trainable_params}, Total: {total_params}"
    )

    # Store initial parameters for comparison
    initial_params = {name: param.to("cpu") for name, param in trainer.model.named_parameters() if param.requires_grad}
    xm.mark_step()

    # Verify initial state
    assert trainer.state.global_step == 0, f"Expected initial global_step=0, got {trainer.state.global_step}"

    # Run training
    trainer.train()

    # Verify training completed
    assert trainer.state.global_step == 5, f"Expected final global_step=5, got {trainer.state.global_step}"

    # Verify training logs exist
    assert len(trainer.state.log_history) > 0, (
        f"Expected training logs to be recorded, got {len(trainer.state.log_history)} log entries"
    )

    # Verify loss was logged
    loss_logged = any("loss" in log for log in trainer.state.log_history)
    assert loss_logged, f"Loss was not logged during SFT PEFT training. Log history: {trainer.state.log_history}"

    final_params = {name: param.to("cpu") for name, param in trainer.model.named_parameters() if param.requires_grad}
    xm.mark_step()

    params_changed = False
    for name, final_param in final_params.items():
        if name in initial_params and not torch.equal(initial_params[name], final_param):
            params_changed = True
            break
    assert params_changed, (
        f"Model parameters were not updated during SFT training. Checked {len(final_params)} parameters"
    )
