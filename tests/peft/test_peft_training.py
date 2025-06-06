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

import copy
import json
from pathlib import Path

import pytest
import torch
from peft import AutoPeftModelForCausalLM, LoraConfig, PeftModel
from peft import get_peft_model as orig_get_peft_model
from safetensors.torch import load_file

import optimum
from optimum.neuron import NeuronTrainer, NeuronTrainingArguments, get_peft_model
from optimum.neuron.distributed.checkpointing import consolidate_model_parallel_checkpoints_to_unified_checkpoint
from optimum.neuron.distributed.utils import lazy_load_for_parallelism
from optimum.neuron.utils.import_utils import is_neuronx_distributed_available, is_torch_xla_available
from optimum.neuron.utils.peft_utils import NeuronPeftModel
from optimum.neuron.utils.testing_utils import is_trainium_test

from .. import DistributedTest
from ..utils import (
    StaticSeedPatcher,
    create_accelerator,
    create_dummy_causal_lm_dataset,
    default_data_collator_for_causal_lm,
    get_model_inputs,
    get_tokenizer_and_tiny_llama_model,
)


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

if is_neuronx_distributed_available():
    from neuronx_distributed.parallel_layers.parallel_state import (
        get_tensor_model_parallel_group,
        get_tensor_model_parallel_size,
    )
    from neuronx_distributed.utils.model_utils import move_model_to_device


def get_peft_config(lora_on_embeddings: bool = False, lora_on_lm_head: bool = False, lora_droupout: float = 0.1):
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if lora_on_embeddings:
        target_modules.append("embed_tokens")
    if lora_on_lm_head:
        target_modules.append("lm_head")
    return LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=lora_droupout,
        bias="none",
        task_type="CAUSAL_LM",
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

    def test_training(self, parallel_sizes, tmpdir):
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

    def test_tied_weights(self, parallel_sizes):
        _, tp_size, pp_size = parallel_sizes

        _, model = get_tokenizer_and_tiny_llama_model()
        model.model.embed_tokens.weight = model.lm_head.weight
        assert model.model.embed_tokens.weight is model.lm_head.weight

        # Case 1: LoRA on embeddings / No LoRA on the LM head
        peft_config = get_peft_config(lora_on_embeddings=True)
        peft_model = get_peft_model(copy.deepcopy(model), peft_config)
        assert peft_model.base_model.model.model.embed_tokens.weight is peft_model.base_model.model.lm_head.weight

        # Case 2: No LoRA on embeddings / LoRA on the LM head
        peft_config = get_peft_config(lora_on_lm_head=True)
        peft_model = get_peft_model(copy.deepcopy(model), peft_config)
        assert peft_model.base_model.model.model.embed_tokens.weight is peft_model.base_model.model.lm_head.weight

        # Case 3: LoRA on embeddings / LoRA on the LM head
        peft_config = get_peft_config(lora_on_embeddings=True, lora_on_lm_head=True)
        peft_model = get_peft_model(copy.deepcopy(model), peft_config)
        assert peft_model.base_model.model.model.embed_tokens.weight is peft_model.base_model.model.lm_head.weight

    def test_outputs_match(self, parallel_sizes, monkeypatch):
        # This is very important otherwise the parallel cross entropy loss will modify the logits inplace.
        monkeypatch.setattr(optimum.neuron.distributed.utils, "_PARALLEL_CROSS_ENTROPY_SHOULD_PRESERVE_INPUT", True)
        world_size, tp_size, pp_size = parallel_sizes
        dp_size = world_size // (tp_size * pp_size)

        peft_config = get_peft_config(lora_on_embeddings=True, lora_on_lm_head=True)

        accelerator = create_accelerator(
            tp_size,
            pp_size,
            parallelize_embeddings=True,
            sequence_parallel_enabled=True,
        )

        seed_patcher = StaticSeedPatcher(42)
        with seed_patcher:
            _, orig_model = get_tokenizer_and_tiny_llama_model()
        with seed_patcher:
            orig_model = orig_get_peft_model(orig_model, peft_config)

        # It is ok to use this accelerator because `patch_model_for_neuron` does not depend on the TP or PP size.
        orig_model = accelerator.patch_model_for_neuron(orig_model)

        inputs = get_model_inputs(
            orig_model,
            orig_model.config.name_or_path,
            batch_size=dp_size,
            pad_to_multiple_of=16,
        )
        xla_inputs = {k: v.to(xm.xla_device()) for k, v in inputs.items()}

        move_model_to_device(orig_model, xm.xla_device())
        xm.mark_step()

        orig_model.eval()
        orig_outputs = orig_model(**xla_inputs)
        xm.mark_step()

        with lazy_load_for_parallelism(tensor_parallel_size=tp_size, pipeline_parallel_size=pp_size):
            with seed_patcher:
                _, model = get_tokenizer_and_tiny_llama_model()

        model = get_peft_model(model, peft_config)

        with seed_patcher:
            model = accelerator.prepare(model)

        model.eval()
        outputs = model(**xla_inputs)
        xm.mark_step()

        outputs_to_test = ["loss", "logits"]
        orig_outputs = {k: v.to("cpu") for k, v in orig_outputs.items() if k in outputs_to_test}
        outputs = {k: v.to("cpu") for k, v in outputs.items() if k in outputs_to_test}
        xm.mark_step()

        for name in outputs_to_test:
            print(f"Checking that the output {name} matches")
            orig_output = orig_outputs[name]
            output = outputs[name]

            # Taken from `tests/distributed/test_model_parallelization::TestModelParallelization._check_output
            tp_size = get_tensor_model_parallel_size()
            tp_group = get_tensor_model_parallel_group()
            if orig_output.shape != output.shape:
                gather_dim = min(
                    idx for idx in range(orig_output.dim()) if orig_output.shape[idx] != output.shape[idx]
                )

                # We could also slice `orig_output` to get the output for the current rank only but we prefer to
                # gather everything as it would happen in a real setting.
                # size = orig_output.size(gather_dim) // tp_size
                # tp_rank = get_tensor_model_parallel_rank()
                # slices = [slice(None, None) if i != gather_dim else slice(tp_rank * size, (tp_rank + 1) * size)  for i in range(orig_output.dim())]
                # orig_output = orig_output[slices]
                output = output.to(xm.xla_device())
                gathered = [torch.empty_like(output) for _ in range(tp_size)]
                torch.distributed.all_gather(gathered, output, group=tp_group)
                gathered_output = torch.cat(gathered, dim=gather_dim)
                xm.mark_step()
                output = gathered_output.to("cpu")

            torch.testing.assert_close(orig_output, output)

    def test_only_adapters_require_grad(self, parallel_sizes):
        _, tp_size, pp_size = parallel_sizes

        peft_config = get_peft_config(lora_on_embeddings=True, lora_on_lm_head=True)

        _, orig_model = get_tokenizer_and_tiny_llama_model()
        orig_model = orig_get_peft_model(orig_model, peft_config)
        orig_requires_grad = {n: p.requires_grad for n, p in orig_model.named_parameters()}

        _, model = get_tokenizer_and_tiny_llama_model()
        model = get_peft_model(model, peft_config)
        accelerator = create_accelerator(
            tp_size,
            pp_size,
            parallelize_embeddings=True,
            sequence_parallel_enabled=True,
        )
        model = accelerator.prepare_model(model)
        requires_grad = {n: p.requires_grad for n, p in model.named_parameters()}

        assert orig_requires_grad == requires_grad
