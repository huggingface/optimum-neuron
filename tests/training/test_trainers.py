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

from pathlib import Path

import pytest
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaForCausalLM

from optimum.neuron import NeuronSFTConfig, NeuronSFTTrainer, NeuronTrainer, NeuronTrainingArguments
from optimum.neuron.models.training import LlamaForCausalLM as NeuronLlamaForCausalLM
from optimum.neuron.models.training import TrainingNeuronConfig
from optimum.neuron.models.training.modeling_utils import MODEL_PARALLEL_SHARDS_DIR_NAME
from optimum.neuron.utils.testing_utils import is_trainium_test
from optimum.neuron.utils.training_utils import get_model_param_count

from ..distributed_utils import distributed_test
from .utils import (
    MODEL_NAME,
    create_dummy_causal_lm_dataset,
    default_data_collator_for_causal_lm,
)


from neuronx_distributed.parallel_layers.parallel_state import (
    get_data_parallel_rank,
    get_pipeline_model_parallel_rank,
    get_tensor_model_parallel_rank,
)


@is_trainium_test
@distributed_test()
@pytest.mark.parametrize(
    "world_size, tp_size, pp_size",
    [[2, 1, 1], [2, 2, 1]],
    ids=["dp=2", "tp=2"],
)
def test_get_model_param_count(world_size, tp_size, pp_size, set_cache_for_ci):
    orig_model = LlamaForCausalLM.from_pretrained(MODEL_NAME)
    target_num_parameters = sum(p.numel() for p in orig_model.parameters())

    trn_config = TrainingNeuronConfig(tensor_parallel_size=tp_size, pipeline_parallel_size=pp_size)
    model = NeuronLlamaForCausalLM.from_pretrained(MODEL_NAME, trn_config)
    num_parameters = get_model_param_count(model)

    assert num_parameters == target_num_parameters


@is_trainium_test
@distributed_test()
@pytest.mark.parametrize(
    "world_size, tp_size, pp_size",
    [[2, 1, 1], [2, 2, 1]],
    ids=["dp=2", "tp=2"],
)
@pytest.mark.skip("Skipping this test until Trainers refactor is done.")
def test_save_checkpoint(world_size, tp_size, pp_size, tmpdir, set_cache_for_ci):
    output_dir = Path(tmpdir)
    dp_rank = get_data_parallel_rank()
    tp_rank = get_tensor_model_parallel_rank()
    pp_rank = get_pipeline_model_parallel_rank()

    args = NeuronTrainingArguments(
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        do_train=True,
        do_eval=False,
        per_device_train_batch_size=1,
        save_steps=5,
        max_steps=20,
        output_dir=output_dir.as_posix(),
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = NeuronLlamaForCausalLM.from_pretrained(MODEL_NAME, args.trn_config)

    datasets = create_dummy_causal_lm_dataset(model.config.vocab_size, 120, 1, sequence_length=128)

    trainer = NeuronTrainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=datasets["train"],
        data_collator=default_data_collator_for_causal_lm,
    )

    trainer.train()

    checkpoint_directories = [f"checkpoint-{k}" for k in range(5, 20, 5)]
    for checkpoint_dir_name in checkpoint_directories:
        # We check that each checkpoint dir exists and contains:
        #   - The model config
        #   - The tokenizer and its config
        #   - The model weights, one file if tp_size = pp_size = 1 or the shards otherwise.
        #   - The optimizer state if tp_size = pp_size = 1 (otherwise the state is in the sharded checkpoints)
        #   - The scheduler state
        #   - The trainer state
        #   - The RNG state
        #   - The training args
        checkpoint_dir = output_dir / checkpoint_dir_name
        assert checkpoint_dir.is_dir()
        assert (checkpoint_dir / "config.json").is_file()
        assert (checkpoint_dir / "tokenizer.json").is_file()
        assert (checkpoint_dir / "tokenizer.model").is_file()
        assert (checkpoint_dir / "tokenizer_config.json").is_file()
        assert (checkpoint_dir / "special_tokens_map.json").is_file()

        shards_dir = checkpoint_dir / MODEL_PARALLEL_SHARDS_DIR_NAME
        assert shards_dir.is_dir()
        assert (shards_dir / "model").is_dir()
        sharded_stem = f"dp_rank_{dp_rank:02d}_tp_rank_{tp_rank:02d}_pp_rank_{pp_rank:02d}"
        assert (shards_dir / "model" / f"{sharded_stem}.pt").is_file()
        assert (shards_dir / "model" / f"{sharded_stem}.pt.tensors").is_dir()

        assert (checkpoint_dir / "scheduler.pt").is_file()
        assert (checkpoint_dir / "trainer_state.json").is_file()
        for worker_id in range(world_size):
            assert (checkpoint_dir / f"rng_state_{worker_id}.pth").is_file()

        assert (checkpoint_dir / "training_args.bin").is_file()


@is_trainium_test
@distributed_test()
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
def test_without_packing(world_size, tp_size, pp_size, packing, tmpdir, set_cache_for_ci):
    output_dir = Path(tmpdir)
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

    def format_dolly(sample):
        instruction = f"### Instruction\n{sample['instruction']}"
        context = f"### Context\n{sample['context']}" if len(sample["context"]) > 0 else None
        response = f"### Answer\n{sample['response']}"
        # join all the parts together
        prompt = "\n\n".join([i for i in [instruction, context, response] if i is not None])
        if packing:
            return prompt
        return [prompt]

    args = NeuronTrainingArguments(
        output_dir=output_dir,
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
        max_seq_length=128,
        packing=packing,
        dataset_num_proc=1,
        **args,
    )

    # Create Trainer instance
    trainer = NeuronSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=format_dolly,
        args=sft_config,
    )

    trainer.train()
