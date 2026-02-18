from dataclasses import dataclass, field

import torch
import torch_xla.core.xla_model as xm
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoTokenizer,
    set_seed,
)

from optimum.neuron import NeuronHfArgumentParser as HfArgumentParser
from optimum.neuron import NeuronSFTConfig, NeuronSFTTrainer, NeuronTrainingArguments
from optimum.neuron.models.training import Qwen3ForCausalLM


def get_dataset(tokenizer):
    dataset_id = "tengomucho/simple_recipes"

    recipes = load_dataset(dataset_id, split="train")
    recipes = recipes.flatten()

    def preprocess_function(examples):
        recipes = examples["recipes"]
        names = examples["names"]

        chats = []
        for recipe, name in zip(recipes, names):
            # Append the EOS token to the response
            recipe += tokenizer.eos_token

            chat = [
                {
                    "role": "user",
                    "content": f"How can I make {name}?",
                },
                {"role": "assistant", "content": recipe},
            ]
            tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)

            chats.append(chat)
        return {"messages": chats}

    dataset = recipes.map(preprocess_function, batched=True, remove_columns=recipes.column_names)
    return dataset


def training_function(script_args, training_args):
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = get_dataset(tokenizer)

    dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    model = Qwen3ForCausalLM.from_pretrained(
        script_args.model_id,
        training_args.trn_config,
        torch_dtype=dtype,
        use_flash_attention_2=script_args.use_flash_attention_2,
    )

    config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["embed_tokens", "q_proj", "v_proj", "o_proj", "k_proj", "up_proj", "down_proj", "gate_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    args = training_args.to_dict()
    packing = True
    # Note: max_seq_length must be a multiple of 2048 to use the flash attention 2 kernel
    sft_config = NeuronSFTConfig(
        max_seq_length=8192,
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
        train_dataset=dataset,
        formatting_func=formatting_function,
    )

    # Start training
    train_result = trainer.train()

    trainer.save_model()  # Saves the tokenizer too for easy upload
    metrics = train_result.metrics
    xm.master_print(f"Model trained in {training_args.output_dir}")
    xm.master_print(metrics)


@dataclass
class ScriptArguments:
    model_id: str = field(
        default="Qwen/Qwen3-8B",
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
