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

"""
Example script for fine-tuning a Qwen3 model using GRPO (Group Relative Policy Optimization) on Neuron devices.

This script demonstrates how to use NeuronGRPOTrainer to train a model with reinforcement learning
using reward functions. GRPO is particularly effective for reasoning tasks and instruction following.

For more information about GRPO, see: https://huggingface.co/papers/2402.03300
"""

from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, HfArgumentParser

from optimum.neuron import NeuronGRPOConfig, NeuronGRPOTrainer, NeuronTrainingArguments
from optimum.neuron.models.training import NeuronModelForCausalLM


# =============================================================================
# Reward Functions
# =============================================================================
# GRPO requires reward functions to score the generated completions.
# These can be:
# 1. Model-based: Use a reward model to score completions
# 2. Rule-based: Custom Python functions that compute rewards
#
# For this example, we use simple rule-based rewards for demonstration.


def length_reward(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    """
    Simple reward function that rewards longer responses (up to a point).

    This is a toy example. In practice, you'd want more sophisticated rewards
    based on task-specific criteria (e.g., accuracy, coherence, helpfulness).

    Args:
        prompts: List of input prompts
        completions: List of generated completions
        **kwargs: Additional arguments (e.g., trainer_state)

    Returns:
        List of reward scores (one per completion)
    """
    rewards = []
    for completion in completions:
        # Reward based on length, but cap at 100 tokens to avoid overly long responses
        length = len(completion.split())
        reward = min(length / 50.0, 2.0)  # Scale: 0-2
        rewards.append(reward)
    return rewards


def unique_words_reward(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    """
    Reward function that encourages diversity by rewarding unique words.

    Args:
        prompts: List of input prompts
        completions: List of generated completions
        **kwargs: Additional arguments

    Returns:
        List of reward scores (one per completion)
    """
    rewards = []
    for completion in completions:
        words = completion.lower().split()
        unique_words = len(set(words))
        total_words = len(words)
        # Reward diversity: ratio of unique words
        reward = unique_words / max(total_words, 1)
        rewards.append(reward)
    return rewards


# =============================================================================
# Data Loading and Preprocessing Function
# =============================================================================
# GRPO requires datasets with a "prompt" column. The trainer will generate
# multiple completions for each prompt and score them using reward functions.


def load_grpo_dataset():
    """
    Load and prepare a dataset for GRPO training.

    For this example, we use the "trl-internal-testing/zen" dataset which is
    a simple test dataset. In practice, you'd use a dataset appropriate for
    your task (e.g., math problems, coding tasks, instruction following).

    Returns:
        Dataset with "prompt" column
    """
    # Load a simple test dataset
    # This dataset has prompts in the "prompt" column
    dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

    # Take a small subset for this example
    dataset = dataset.select([0] * 100000)

    return dataset


# =============================================================================
# Model Loading and Training Loop Function
# =============================================================================
def train(model_id, tokenizer, dataset, training_args):
    """
    Train the model using GRPO.

    Args:
        model_id: HuggingFace model ID or path
        tokenizer: Tokenizer for the model
        dataset: Training dataset with "prompt" column
        training_args: NeuronTrainingArguments
    """
    # NOTE: Models with custom modeling implementation need a TrainingNeuronConfig
    # This is automatically created when using NeuronTrainingArguments
    trn_config = training_args.trn_config
    dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    model = NeuronModelForCausalLM.from_pretrained(
        model_id,
        trn_config,
        torch_dtype=dtype,
        # Use FlashAttention2 for better performance
        # attn_implementation="flash_attention_2",
        attn_implementation="eager",
    )

    # LoRA configuration for efficient fine-tuning
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["embed_tokens", "q_proj", "v_proj", "o_proj", "k_proj", "up_proj", "down_proj", "gate_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Convert NeuronTrainingArguments to dict for NeuronGRPOConfig
    args = training_args.to_dict()

    # GRPO-specific configuration
    grpo_config = NeuronGRPOConfig(
        # Generation parameters
        max_prompt_length=512,  # Maximum prompt length
        max_completion_length=268,  # Maximum completion length
        num_generations=4,  # Number of completions to generate per prompt (G in paper)
        temperature=0.8,  # Sampling temperature
        # GRPO algorithm parameters
        num_iterations=1,  # Number of iterations per batch (μ in paper)
        epsilon=0.2,  # Clipping parameter
        beta=0.01,  # KL divergence coefficient
        scale_rewards="group",  # Reward scaling strategy
        # vLLM parameters
        use_vllm=True,  # Use vLLM for generation (required for Neuron)
        vllm_mode="server",  # Use vLLM server mode
        vllm_server_host="0.0.0.0",
        vllm_server_port=8000,
        # Standard training arguments from NeuronTrainingArguments
        **args,
    )

    # Define reward functions
    # You can use multiple reward functions - they will be summed
    reward_funcs = [
        length_reward,
        unique_words_reward,
    ]

    # Create the GRPO trainer
    trainer = NeuronGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # Train the model
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

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_grpo_dataset()

    # Start training
    train(
        model_id=script_args.model_id,
        tokenizer=tokenizer,
        dataset=dataset,
        training_args=training_args,
    )
