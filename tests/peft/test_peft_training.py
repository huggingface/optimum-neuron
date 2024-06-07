# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Tests related to PEFT integration."""

import json
from pathlib import Path

import pytest
import torch
from peft import AutoPeftModelForCausalLM, LoraConfig, PeftModel
from peft import get_peft_model as orig_get_peft_model
from safetensors.torch import load_file
from transformers import LlamaForCausalLM

from optimum.neuron import NeuronTrainer, NeuronTrainingArguments, get_peft_model
from optimum.neuron.distributed.checkpointing import consolidate_model_parallel_checkpoints_to_unified_checkpoint
from optimum.neuron.utils.peft_utils import NeuronPeftModel
from optimum.neuron.utils.testing_utils import is_trainium_test

from .. import DistributedTest
from ..utils import (
    create_accelerator,
    create_dummy_causal_lm_dataset,
    create_static_seed_patcher,
    default_data_collator_for_causal_lm,
    get_tokenizer_and_tiny_llama_model,
)


def get_peft_config():
    target_modules = ["embed_tokens", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    return LoraConfig(
        r=4, lora_alpha=16, target_modules=target_modules, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
    )


def test_get_peft_model():
    peft_config = get_peft_config()
    _, model = get_tokenizer_and_tiny_llama_model()
    orig_peft_model = orig_get_peft_model(model, peft_config)

    assert isinstance(orig_peft_model, PeftModel)
    assert not isinstance(orig_peft_model, NeuronPeftModel)

    _, model = get_tokenizer_and_tiny_llama_model()
    peft_model = get_peft_model(model, peft_config)

    assert isinstance(peft_model, NeuronPeftModel)


@is_trainium_test
class TestPeft(DistributedTest):
    @pytest.fixture(
        scope="class",
        params=[[2, 1, 1], [2, 2, 1]],
        ids=["dp=2", "tp=2"],
    )
    def parallel_sizes(self, request):
        return request.param

    @pytest.mark.world_size(2)
    def test_peft_model_is_converted_to_neuron_peft_model(self):
        model = AutoPeftModelForCausalLM.from_pretrained("peft-internal-testing/tiny-random-BertModel-lora")
        assert isinstance(model, PeftModel)
        accelerator = create_accelerator(1, 1)
        model = accelerator.prepare(model)
        assert isinstance(model, NeuronPeftModel)

    def test_save_pretrained(self, parallel_sizes, tmpdir):
        _, tp_size, pp_size = parallel_sizes

        output_dir = Path(tmpdir)

        peft_config = get_peft_config()

        # PEFT model saved using `PeftModel`.
        seed_patcher = create_static_seed_patcher(LlamaForCausalLM, 42)
        with seed_patcher:
            _, model = get_tokenizer_and_tiny_llama_model()
            orig_model_path = output_dir / "orig_peft"
            orig_peft_model = orig_get_peft_model(model, peft_config)

        orig_peft_model.save_pretrained(orig_model_path.as_posix())

        # PEFT model saved using `NeuronPeftModel`.
        seed_patcher = create_static_seed_patcher(LlamaForCausalLM, 42)
        with seed_patcher:
            _, model = get_tokenizer_and_tiny_llama_model()
            model_path = output_dir / "peft"
            peft_model = get_peft_model(model, peft_config)

        accelerator = create_accelerator(tp_size, pp_size)
        peft_model = accelerator.prepare_model(peft_model)
        peft_model.save_pretrained(model_path.as_posix())

        with open(orig_model_path / "adapter_config.json") as fp:
            orig_adapter_config_content = json.dumps(json.load(fp), sort_keys=True)

        with open(model_path / "adapter_config.json") as fp:
            adapter_config_content = json.dumps(json.load(fp), sort_keys=True)

        assert orig_adapter_config_content == adapter_config_content, "adapter_config.json files do not match"

        orig_state_dict = load_file(orig_model_path / "adapter_model.safetensors")

        if tp_size > 1 or pp_size > 1:
            # In this case, only shards have been saved:
            #   1. We check that they do exist
            #   2. We consolidate them
            #   3. We check that the values match just as the case where tp_size = pp_size = 1.
            assert (model_path / "adapter_shards").is_dir()
            consolidate_model_parallel_checkpoints_to_unified_checkpoint(
                model_path, model_path / "consolidated_checkpoint"
            )
            state_dict = load_file(model_path / "consolidated_checkpoint" / "adapter_model.safetensors")
        else:
            state_dict = load_file(model_path / "adapter_model.safetensors")

        assert orig_state_dict.keys() == state_dict.keys()
        for name, tensor in orig_state_dict.items():
            print(f"Checking that the parameter {name} matches")
            torch.testing.assert_close(tensor, state_dict[name])

    def test_peft_training(self, parallel_sizes, tmpdir):
        _, tp_size, pp_size = parallel_sizes

        per_device_train_batch_size = 1
        output_dir = Path(tmpdir)
        args = NeuronTrainingArguments(
            output_dir=output_dir.as_posix(),
            do_train=True,
            do_eval=False,
            bf16=True,
            per_device_train_batch_size=per_device_train_batch_size,
            save_strategy="epoch",
            logging_steps=10,
            num_train_epochs=2,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
        )

        tokenizer, model = get_tokenizer_and_tiny_llama_model()

        num_train_samples = num_eval_samples = 50
        datasets = create_dummy_causal_lm_dataset(
            model.config.vocab_size, num_train_samples, num_eval_samples, max_number_of_unique_examples=3
        )

        trainer = NeuronTrainer(
            model,
            args,
            tokenizer=tokenizer,
            train_dataset=datasets["train"],
            eval_dataset=datasets["eval"],
            data_collator=default_data_collator_for_causal_lm,
        )

        trainer.train()
