import numpy as np
import torch
import evaluate
import argparse
from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig, setup_chat_format
from optimum.neuron.models.training import LlamaForCausalLM
from optimum.neuron.models.training.config import TrainingNeuronConfig
from optimum.neuron import NeuronHfArgumentParser as HfArgumentParser
from optimum.neuron import NeuronSFTConfig, NeuronSFTTrainer, NeuronTrainingArguments
from optimum.neuron.utils import is_neuronx_distributed_available

if is_neuronx_distributed_available():
    from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_size

"""
To do a quick test, you can launch this with:

python training_tes_language_generation.py \
    --epochs=10 \
    --max_steps=40 \
    --per_device_train_batch_size 8 \
    --train_max_length 64

Inspired from: https://github.com/huggingface/smol-course/blob/main/1_instruction_tuning/notebooks/sft_finetuning_example.ipynb
"""


@dataclass
class ScriptArguments:
    model_id: str = field(
        default="HuggingFaceTB/SmolLM2-135M",
        # default="Qwen/Qwen3-0.6B",
        metadata={"help": "The model that you want to train from the Hugging Face hub."},
    )
    flash_attention_2: bool = field(
        default=False,
        metadata={"help": "Whether to use Flash Attention 2."},
    )

def parse_args():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument
    parser.add_argument(
        "--model_id",
        type=str,
        default="HuggingFaceTB/SmolLM2-135M",
        # default="Qwen/Qwen3-0.6B",
        help="Model id to use for training.",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs to train for.",
    )
    parser.add_argument(
        "-s",
        "--max_steps",
        type=int,
        default=-1,
        help="Number of steps to train for.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size to use for training.",
    )
    parser.add_argument(
        "--train_max_length",
        type=int,
        default=512,
        help="Maximum length of the training data.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate to use for training.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Logging steps to use for training.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save steps to use for training.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=50,
        help="Evaluation steps to use for training.",
    )
    parser.add_argument(
        "--packing",
        type=bool,
        default=True,
        help="Packing to use for training.",
    )
    parser.add_argument(
        "--flash_attention_2",
        type=bool,
        default=True,
        help="Use flash attention 2 for training.",
    )
    parser.add_argument(
        "--sequence_parallel_enabled",
        type=bool,
        default=False,
        help="Use sequence parallelism for training.",
    )
    args = parser.parse_args()
    return args


def training_test_language(args):
    model_id = args.model_id
    max_length = args.train_max_length
    batch_size = args.per_device_train_batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    max_steps = args.max_steps
    logging_steps = args.logging_steps
    save_steps = args.save_steps
    eval_steps = args.eval_steps
    packing = args.packing
    flash_attention_2 = args.flash_attention_2
    model_name = model_id.split("/")[-1]
    output_dir = f"{model_name}-finetuned"
    # TODO: remove this
    torch.distributed.init_process_group(backend="xla")
    tensor_parallel_size = get_tensor_model_parallel_size()
    sequence_parallel_enabled = args.sequence_parallel_enabled
    device = torch.device("xla")
    torch_dtype = torch.float32


    trn_config = TrainingNeuronConfig(
        tensor_parallel_size=tensor_parallel_size,
        sequence_parallel_enabled=sequence_parallel_enabled,
    )

    # Load the model and tokenizer
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_id, trn_config=trn_config, torch_dtype=torch_dtype, flash_attention_2=flash_attention_2
    ).to(device)


    print(model)
    return
    # model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_id, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Set up the chat format
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

    # Test the original model
    # question = "Can you tell me a quote from Shakespeare on these topics: love, death, and life?"
    # inputs = tokenizer(
    #     [INFERENCE_PROMPT_STYLE.format(question) + tokenizer.eos_token],
    #     return_tensors="pt"
    # ).to(model.device)

    # outputs = model.generate(
    #     **inputs,
    #     max_new_tokens=1200,
    #     eos_token_id=tokenizer.eos_token_id,
    #     use_cache=True,
    # )
    # response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # print(response[0].split("### Response:")[1])
    # return

    # Set up the chat format
    # model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

    # Let's test the base model before training
    # prompt = "Write a haiku about programming"

    # # Format with template
    # messages = [{"role": "user", "content": prompt}]
    # formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)

    # Generate response
    # inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    # outputs = model.generate(**inputs, max_new_tokens=100)
    # print("Before training:")
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    # Loading and prepare dataset
    dataset_id = "tengomucho/english_quotes_sanitized"
    print(f"Loading dataset {dataset_id}")
    quotes = load_dataset(dataset_id, split="train")
    quotes = quotes.train_test_split(test_size=0.2)
    quotes = quotes.flatten()
    print("Dataset loaded")

    def preprocess_function(examples):
        quotes = examples["quote"]
        authors = examples["author"]
        topics = examples["tags"]
        chats = []
        for quote, author, topics in zip(quotes, authors, topics):
            # Append the EOS token to the response if it's not already there
            if not quote.endswith(tokenizer.eos_token):
                quote += tokenizer.eos_token
            topics = ", ".join(topics)
            question = f"Can you give me a quote from {author} on one of these topics: {topics}?"
            chat = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": quote},
            ]
            tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=False
            )

            chats.append(chat)
        return {"messages": chats}

    dataset = quotes.map(
        preprocess_function, batched=True, remove_columns=quotes["train"].column_names
    )

    print(dataset["train"]["messages"][0])

    peft_config = LoraConfig(
        lora_alpha=16,  # Scaling factor for LoRA
        lora_dropout=0.05,  # Add slight dropout for regularization
        r=64,  # Rank of the LoRA update matrices
        bias="none",  # No bias reparameterization
        task_type="CAUSAL_LM",  # Task type: Causal Language Modeling
        target_modules="all-linear",  # Target modules for LoRA
    )

    model = get_peft_model(model, peft_config)

    # Configure the SFTTrainer
    sft_config = SFTConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        max_seq_length=max_length,
        packing=packing,
        use_mps_device=(
            True if device == "mps" else False
        ),  # Use MPS for mixed precision training
    )

    # Initialize the SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Training model")
    train_result = trainer.train()

    finetune_name = f"{model_name}-finetuned-output"
    trainer.save_model(f"./{finetune_name}")

    metrics = train_result.metrics
    print("Model trained")
    print(metrics)
    print("Evaluating model")
    eval_result = trainer.evaluate()
    print("Model evaluated")
    print(eval_result)


def main():
    # torch.distributed.init_process_group(backend="xla")
    # parser = HfArgumentParser([ScriptArguments, NeuronTrainingArguments])
    # script_args, training_args = parser.parse_args_into_dataclasses()

    # breakpoint()
    # return
    args = parse_args()
    training_test_language(args)


if __name__ == "__main__":
    main()
