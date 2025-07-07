from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    set_seed,
)

from optimum.neuron import NeuronHfArgumentParser as HfArgumentParser
from optimum.neuron import NeuronSFTConfig, NeuronSFTTrainer, NeuronTrainingArguments
from optimum.neuron.models.training import Qwen3ForCausalLM


def format_dolly_not_packing(examples):
    output_text = []
    for i in range(len(examples["instruction"])):
        instruction = f"### Instruction\n{examples['instruction'][i]}"
        context = f"### Context\n{examples['context'][i]}" if len(examples["context"][i]) > 0 else None
        response = f"### Answer\n{examples['response'][i]}"
        prompt = "\n\n".join([i for i in [instruction, context, response] if i is not None])
        output_text.append(prompt)
    return output_text


def format_dolly_packing(example):
    instruction = f"### Instruction\n{example['instruction']}"
    context = f"### Context\n{example['context']}" if len(example["context"]) > 0 else None
    response = f"### Answer\n{example['response']}"
    prompt = "\n\n".join([i for i in [instruction, context, response] if i is not None])
    return prompt


def training_function(script_args, training_args):
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    dataset = dataset.select([0] * 10000)

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # with lazy_load_for_parallelism(tensor_parallel_size=training_args.tensor_parallel_size):
    #     model = AutoModelForCausalLM.from_pretrained(script_args.model_id)
    dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    # model = LlamaForCausalLM.from_pretrained(
    #     script_args.model_id, training_args.trn_config, torch_dtype=dtype, use_flash_attention_2=True
    # )
    # model = GraniteForCausalLM.from_pretrained(
    #     script_args.model_id, training_args.trn_config, torch_dtype=dtype, use_flash_attention_2=True
    # )
    model = Qwen3ForCausalLM.from_pretrained(
        script_args.model_id, training_args.trn_config, torch_dtype=dtype, use_flash_attention_2=True
    )

    # config = LoraConfig(
    #     r=16,
    #     lora_alpha=32,
    #     lora_dropout=0.05,
    #     target_modules=["embed_tokens", "q_proj", "v_proj", "o_proj", "k_proj", "gate_proj", "up_proj", "down_proj"],
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )

    args = training_args.to_dict()
    packing = True
    sft_config = NeuronSFTConfig(
        max_seq_length=2048,
        # max_seq_length=512,
        packing=packing,
        **args,
    )

    trainer = NeuronSFTTrainer(
        args=sft_config,
        model=model,
        # peft_config=config,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=format_dolly_packing if packing else format_dolly_not_packing,
    )

    # Start training
    trainer.train()

    trainer.save_model()  # Saves the tokenizer too for easy upload


@dataclass
class ScriptArguments:
    model_id: str = field(
        default="meta-llama/Meta-Llama-3-8B",
        metadata={"help": "The model that you want to train from the Hugging Face hub."},
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
