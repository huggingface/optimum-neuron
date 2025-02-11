from dataclasses import dataclass, field

from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)

from optimum.neuron import NeuronHfArgumentParser as HfArgumentParser
from optimum.neuron import NeuronSFTConfig, NeuronSFTTrainer, NeuronTrainingArguments
from optimum.neuron.distributed import lazy_load_for_parallelism


def format_dolly(examples):
    output_text = []
    for i in range(len(examples["instruction"])):
        instruction = f"### Instruction\n{examples['instruction'][i]}"
        context = f"### Context\n{examples['context'][i]}" if len(examples["context"][i]) > 0 else None
        response = f"### Answer\n{examples['response'][i]}"
        prompt = "\n\n".join([i for i in [instruction, context, response] if i is not None])
        output_text.append(prompt)
    return output_text


def training_function(script_args, training_args):
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    with lazy_load_for_parallelism(tensor_parallel_size=training_args.tensor_parallel_size):
        model = AutoModelForCausalLM.from_pretrained(script_args.model_id)

    config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "gate_proj", "v_proj", "o_proj", "k_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    args = training_args.to_dict()
    sft_config = NeuronSFTConfig(
        max_seq_length=1024,
        packing=False,
        **args,
    )

    trainer = NeuronSFTTrainer(
        args=sft_config,
        model=model,
        peft_config=config,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=format_dolly,
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
