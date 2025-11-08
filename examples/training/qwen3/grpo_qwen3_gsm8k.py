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

import re
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, HfArgumentParser

from optimum.neuron import NeuronGRPOConfig, NeuronGRPOTrainer, NeuronTrainingArguments
from optimum.neuron.models.training import NeuronModelForCausalLM


# =============================================================================
# GSM8K Answer Extraction
# =============================================================================
def extract_answer(completion: str) -> Optional[str]:
    """Extract final answer from GSM8K completion after #### marker."""
    match = re.search(r"####\s*([^\n]+)", completion)
    if match:
        answer = match.group(1).strip()
        # Extract numeric value
        num_match = re.search(r"[\d,]+(?:\.\d+)?", answer)
        if num_match:
            return num_match.group(0).replace(",", "")
    return None


def compute_reward(generated: list[str], ground_truth: list[str]) -> list[float]:
    """Compute reward based on whether extracted answers match ground truth."""
    rewards = []
    for gen, gt in zip(generated, ground_truth):
        gen_answer = extract_answer(gen)
        gt_answer = extract_answer(gt)
        
        if gen_answer and gt_answer and gen_answer == gt_answer:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    
    return rewards


# =============================================================================
# Data Loading and Preprocessing Function
# =============================================================================
dataset_id = "openai/gsm8k"


def load_gsm8k_dataset(split: str = "train", max_samples: Optional[int] = None):
    """Load and preprocess GSM8K dataset."""
    dataset = load_dataset(dataset_id, "main", split=split)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    def preprocess_function(examples):
        questions = examples["question"]
        answers = examples["answer"]
        
        # Create prompt strings from questions
        prompts = []
        for question in questions:
            prompts.append(question)
        
        return {"prompt": prompts, "answer": answers}
    
    dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
    return dataset


# =============================================================================
# Reward Function for GRPO
# =============================================================================
# Store answers by prompt for lookup - simple approach
_answer_cache = {}

def create_reward_function(dataset):
    """Create reward function that looks up answers from dataset."""
    # Build a mapping of prompt to answer
    global _answer_cache
    _answer_cache = {item["prompt"]: item["answer"] for item in dataset}
    
    def reward_function(prompts, completions):
        rewards = []
        for prompt, completion in zip(prompts, completions):
            # Lookup answer for this prompt
            if prompt in _answer_cache:
                gt_answer = _answer_cache[prompt]
                gen_answer = extract_answer(completion)
                
                if gen_answer and gt_answer:
                    gt_num = extract_answer(gt_answer)
                    if gt_num and gen_answer == gt_num:
                        rewards.append(1.0)
                    else:
                        rewards.append(0.0)
                else:
                    rewards.append(0.0)
            else:
                rewards.append(0.0)
        
        return rewards
    
    return reward_function


# =============================================================================
# Model Loading and Training Loop Function
# =============================================================================
def train(model_id, tokenizer, dataset, training_args, model_size):
    # Get model config based on size
    trn_config = training_args.trn_config
    dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    
    model = NeuronModelForCausalLM.from_pretrained(
        model_id,
        trn_config,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
    )

    # LoRA config
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["embed_tokens", "q_proj", "v_proj", "o_proj", "k_proj", "up_proj", "down_proj", "gate_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Create GRPO config
    args = training_args.to_dict()
    
    # GRPO-specific config parameters
    grpo_config = NeuronGRPOConfig(
        max_prompt_length=1024,
        max_completion_length=128,
        num_generations=4,
        steps_per_generation=8,
        **args,
    )

    # Create reward function
    reward_func = create_reward_function(dataset)

    # Create GRPO trainer
    trainer = NeuronGRPOTrainer(
        args=grpo_config,
        model=model,
        peft_config=lora_config,
        tokenizer=tokenizer,
        train_dataset=dataset,
        reward_funcs=[reward_func],
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
    model_size: Optional[str] = field(
        default=None,
        metadata={"help": "Model size for configuration: 'small' or 'large'"},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of samples to use from dataset"},
    )


# =============================================================================
# Main Function
# =============================================================================
if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, NeuronTrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)
    
    # Determine model size if not specified
    if script_args.model_size is None:
        if "0.6" in script_args.model_id or "0.6B" in script_args.model_id:
            model_size = "small"
        elif "8" in script_args.model_id or "8B" in script_args.model_id:
            model_size = "large"
        else:
            model_size = "small"
    else:
        model_size = script_args.model_size
    
    # Load dataset
    max_samples = script_args.max_samples
    dataset = load_gsm8k_dataset(split="train", max_samples=max_samples)
    
    print(f"Loaded {len(dataset)} samples from GSM8K dataset")
    print(f"Model size configuration: {model_size}")

    train(
        model_id=script_args.model_id,
        tokenizer=tokenizer,
        dataset=dataset,
        training_args=training_args,
        model_size=model_size,
    )

