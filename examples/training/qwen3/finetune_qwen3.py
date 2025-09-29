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
# NOTE: this section can be adapted to laod any dataset you want.
dataset_id = "tengomucho/simple_recipes"
recipes = load_dataset(dataset_id, split="train")


def preprocess_dataset_with_eos(eos_token):
    def preprocess_function(examples):
        recipes = examples["recipes"]
        names = examples["names"]

        chats = []
        for recipe, name in zip(recipes, names):
            # Append the EOS token to the response
            recipe += eos_token

            chat = [
                {"role": "user", "content": f"How can I make {name}?"},
                {"role": "assistant", "content": recipe},
            ]

            chats.append(chat)
        return {"messages": chats}

    dataset = recipes.map(preprocess_function, batched=True, remove_columns=recipes.column_names)
    return dataset


# =============================================================================
# Model Loading and Training Loop Function
# =============================================================================
def train(model_id, tokenizer, dataset, training_args):
    # NOTE: Models with a custom modeling implementation (like Qwen3, Llama, Granite, etc) need a `TrainingNeuronConfig`
    # to be passed. This configuration defines modeling customization and distributed training parameters.
    # It is automatically created when using `NeuronTrainingArguments`.
    trn_config = training_args.trn_config
    dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    model = NeuronModelForCausalLM.from_pretrained(
        model_id,
        trn_config,
        torch_dtype=dtype,
        # Use FlashAttention2 for better performance and to be able to use larger sequence lengths.
        attn_implementation="flash_attention_2",
    )

    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["embed_tokens", "q_proj", "v_proj", "o_proj", "k_proj", "up_proj", "down_proj", "gate_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Converting the NeuronTrainingArguments to a dictionary to feed them to the NeuronSFTConfig.
    args = training_args.to_dict()

    sft_config = NeuronSFTConfig(
        max_seq_length=4096,
        packing=True,
        **args,
    )

    def formatting_function(examples):
        return tokenizer.apply_chat_template(examples["messages"], tokenize=False, add_generation_prompt=False)

    # The NeuronSFTTrainer will use `formatting_function` to format the dataset and `lora_config` to apply LoRA on the
    # model.
    trainer = NeuronSFTTrainer(
        args=sft_config,
        model=model,
        peft_config=lora_config,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=formatting_function,
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
    dataset = preprocess_dataset_with_eos(tokenizer.eos_token)

    train(
        model_id=script_args.model_id,
        tokenizer=tokenizer,
        dataset=dataset,
        training_args=training_args,
    )
