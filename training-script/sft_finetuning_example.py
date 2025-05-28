from dataclasses import dataclass, field
import os
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoTokenizer,
    set_seed,
)

from optimum.neuron import NeuronHfArgumentParser as HfArgumentParser
from optimum.neuron import NeuronSFTConfig, NeuronSFTTrainer, NeuronTrainingArguments

from optimum.neuron.models.training import Qwen3ForCausalLM
# from optimum.neuron.models.training import LlamaForCausalLM


def print0(*args, **kwargs):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


def get_dataset(tokenizer):
    dataset_id = "formido/recipes-20k"

    recipes = load_dataset(dataset_id, split="train")
    recipes = recipes.train_test_split(test_size=0.2)
    recipes = recipes.flatten()

    def preprocess_function(examples):
        recipes = examples["output"]
        names = examples["input"]

        chats = []
        for recipe, name in zip(recipes, names):
            # Sanitize the recipe string
            orig_recipe = recipe
            recipe = recipe.replace("\\'", "'")
            recipe = recipe.strip("\\']")
            recipe = recipe.strip("['")
            recipe = recipe.replace("\', \'", "\n- ")
            recipe = recipe.replace(" , ", ", ")
            # Append the EOS token to the response
            recipe += tokenizer.eos_token

            chat = [
                {"role": "user", "content": f"Give me the instructions to prepare the dish {name}"},
                {"role": "assistant", "content": recipe},
            ]
            tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=False
            )

            chats.append(chat)
        return {"messages": chats}

    dataset = recipes.map(
        preprocess_function, batched=True, remove_columns=recipes["train"].column_names
    )
    return dataset



def training_function(script_args, training_args):
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = get_dataset(tokenizer)

    dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    model = Qwen3ForCausalLM.from_pretrained(script_args.model_id, training_args.trn_config, torch_dtype=dtype, use_flash_attention_2=script_args.use_flash_attention_2)

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        # target_modules=["q_proj", "v_proj", "o_proj", "k_proj", "up_proj", "down_proj"],
        target_modules=["embed_tokens", "q_proj", "v_proj", "o_proj", "k_proj", "up_proj", "down_proj", "gate_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    args = training_args.to_dict()
    packing = True
    sft_config = NeuronSFTConfig(
        max_seq_length=2048,
        packing=packing,
        **args,
    )

    def formatting_function(examples):
        return tokenizer.apply_chat_template(examples["messages"], tokenize=False, add_generation_prompt=False)


    trainer = NeuronSFTTrainer(
        args=sft_config,
        model=model,
        peft_config=config,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        formatting_func=formatting_function,
    )

    # Start training
    train_result = trainer.train()

    trainer.save_model()  # Saves the tokenizer too for easy upload
    metrics = train_result.metrics
    print0("Model trained")
    print0(metrics)
    print0("Evaluating model")
    eval_result = trainer.evaluate()
    print0("Model evaluated")
    print0(eval_result)


@dataclass
class ScriptArguments:
    model_id: str = field(
        default="meta-llama/Meta-Llama-3-8B",
        metadata={"help": "The model that you want to train from the Hugging Face hub."},
    )
    use_flash_attention_2: bool = field(
        default=True,
        metadata={"help": "Whether to use Flash Attention 2."},
    )


def main():
    parser = HfArgumentParser([ScriptArguments, NeuronTrainingArguments])
    script_args, training_args = parser.parse_args_into_dataclasses()

    # set seed
    set_seed(training_args.seed)

    # run training function
    training_function(script_args, training_args)


if __name__ == "__main__":
    main()
