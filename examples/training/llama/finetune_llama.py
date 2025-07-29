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

from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, HfArgumentParser

from optimum.neuron import NeuronSFTConfig, NeuronSFTTrainer, NeuronTrainingArguments
from optimum.neuron.models.training import NeuronModelForCausalLM


# =============================================================================
# Data Loading and Preprocessing Function
# =============================================================================
# NOTE: this section can be adapted to load any dataset you want.
dataset_id = "databricks/databricks-dolly-15k"
dolly_dataset = load_dataset(dataset_id, split="train")
# To remove, it's just to test training.
dolly_dataset = dolly_dataset.select([0] * 100000)


def format_dolly(example):
    """Format Dolly dataset examples using Llama 3.1 chat template format."""
    user_content = example["instruction"]
    if len(example["context"]) > 0:
        user_content += f"\n\nContext: {example['context']}"

    # Format using Llama 3.1 chat template structure
    formatted_text = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "Cutting Knowledge Date: December 2023\n"
        "Today Date: 29 Jul 2025\n\n"
        "You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}<|eot_id|>\n"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{example['response']}<|eot_id|>"
    )
    return formatted_text


# =============================================================================
# Model Loading and Training Loop Function
# =============================================================================
def train(model_id, tokenizer, dataset, training_args):
    # NOTE: Models with a custom modeling implementation (like Llama, Qwen3, Granite, etc) need a `TrainingNeuronConfig`
    # to be passed. This configuration defines modeling customization and distributed training parameters.
    # It is automatically created when using `NeuronTrainingArguments`.
    trn_config = training_args.trn_config
    dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    model = NeuronModelForCausalLM.from_pretrained(
        model_id,
        trn_config,
        torch_dtype=dtype,
        # Use FlashAttention2 for better performance and to be able to use larger sequence lengths.
        use_flash_attention_2=True,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["embed_tokens", "q_proj", "gate_proj", "v_proj", "o_proj", "k_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Converting the NeuronTrainingArguments to a dictionary to feed them to the NeuronSFTConfig.
    args = training_args.to_dict()

    sft_config = NeuronSFTConfig(
        max_seq_length=2048,
        packing=True,
        **args,
    )

    # The NeuronSFTTrainer will use `format_dolly` to format the dataset and `lora_config` to apply LoRA on the
    # model.
    trainer = NeuronSFTTrainer(
        args=sft_config,
        model=model,
        peft_config=lora_config,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=format_dolly,
    )
    trainer.train()


# =============================================================================
# Defining the script-specific arguments
# =============================================================================
@dataclass
class ScriptArguments:
    model_id: str = field(
        metadata={"help": "The model that you want to train from the Hugging Face hub."},
    )


# =============================================================================
# Main Function
# =============================================================================
if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, NeuronTrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)
    tokenizer.pad_token = "<|finetune_right_pad_id|>"

    train(
        model_id=script_args.model_id,
        tokenizer=tokenizer,
        dataset=dolly_dataset,
        training_args=training_args,
    )
