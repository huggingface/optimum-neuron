# coding=utf-8
from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, HfArgumentParser

from optimum.neuron import NeuronTrainingArguments
from optimum.neuron.models.training import NeuronModelForCausalLM
from optimum.neuron.trainers.grpo_config import NeuronGRPOConfig
from optimum.neuron.trainers.grpo_trainer import NeuronGRPOTrainer


# Minimal dataset: prompts only
dataset_id = "tengomucho/simple_recipes"
recipes = load_dataset(dataset_id, split="train")


def preprocess_prompts():
    def to_prompt(examples):
        names = examples["names"]
        prompts = [f"How can I make {name}?" for name in names]
        return {"prompt": prompts}

    ds = recipes.map(to_prompt, batched=True, remove_columns=recipes.column_names)
    
    # Take only first 10 samples for quick testing
    ds = ds.select(range(min(10, len(ds))))
    return ds


def reward_contains_word(word: str):
    def fn(prompts, completions, **kwargs):
        scores = []
        for text in completions:
            text_l = text.lower() if isinstance(text, str) else ""
            scores.append(1.0 if word in text_l else 0.0)
        return scores
    return fn


def train(model_id, tokenizer, dataset, training_args):
    trn_config = training_args.trn_config
    dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    model = NeuronModelForCausalLM.from_pretrained(
        model_id,
        trn_config,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
    )

    args = training_args.to_dict()
    grpo_config = NeuronGRPOConfig(
        # minimal GRPO essentials
        max_prompt_length=256,
        max_completion_length=64,
        num_generations=4,
        **args,
    )

    # minimal reward function
    reward_fn = reward_contains_word("salt")

    trainer = NeuronGRPOTrainer(
        args=grpo_config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
    )
    trainer.train()


@dataclass
class ScriptArguments:
    model_id: str = field(metadata={"help": "Model id from Hugging Face hub."})


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, NeuronTrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)
    dataset = preprocess_prompts()

    train(
        model_id=script_args.model_id,
        tokenizer=tokenizer,
        dataset=dataset,
        training_args=training_args,
    )



