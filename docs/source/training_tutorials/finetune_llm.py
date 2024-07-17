from dataclasses import dataclass, field
from functools import partial
from itertools import chain
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    set_seed,
)

from optimum.neuron import NeuronHfArgumentParser as HfArgumentParser
from optimum.neuron import NeuronTrainer as Trainer
from optimum.neuron import NeuronTrainingArguments as TrainingArguments
from optimum.neuron.distributed import lazy_load_for_parallelism


# Load dataset from the hub
dataset = load_dataset("databricks/databricks-dolly-15k", split="train")


def format_dolly(sample):
    instruction = f"### Instruction\n{sample['instruction']}"
    context = f"### Context\n{sample['context']}" if len(sample["context"]) > 0 else None
    response = f"### Answer\n{sample['response']}"
    # join all the parts together
    prompt = "\n\n".join([i for i in [instruction, context, response] if i is not None])
    return prompt


# empty list to save remainder from batches to use in next batch
remainder = {"input_ids": [], "attention_mask": [], "token_type_ids": []}


def pack_dataset(dataset, chunk_length=2048):
    print(f"Chunking dataset into chunks of {chunk_length} tokens.")

    def chunk(sample, chunk_length=chunk_length):
        # define global remainder variable to save remainder from batches to use in next batch
        global remainder
        # Concatenate all texts and add remainder from previous batch
        concatenated_examples = {k: list(chain(*sample[k])) for k in sample.keys()}
        concatenated_examples = {k: remainder[k] + concatenated_examples[k] for k in concatenated_examples.keys()}
        # get total number of tokens for batch
        batch_total_length = len(concatenated_examples[list(sample.keys())[0]])

        # get max number of chunks for batch
        if batch_total_length >= chunk_length:
            batch_chunk_length = (batch_total_length // chunk_length) * chunk_length

        # Split by chunks of max_len.
        result = {
            k: [t[i : i + chunk_length] for i in range(0, batch_chunk_length, chunk_length)]
            for k, t in concatenated_examples.items()
        }
        # add remainder to global variable for next batch
        remainder = {k: concatenated_examples[k][batch_chunk_length:] for k in concatenated_examples.keys()}
        # prepare labels
        result["labels"] = result["input_ids"].copy()
        return result

    # tokenize and chunk dataset
    lm_dataset = dataset.map(
        partial(chunk, chunk_length=chunk_length),
        batched=True,
    )
    print(f"Total number of samples: {len(lm_dataset)}")
    return lm_dataset


def create_and_save_dataset(model_id: str, dataset_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # template dataset to add prompt to each sample
    def template_dataset(sample):
        sample["text"] = f"{format_dolly(sample)}{tokenizer.eos_token}"
        return sample

    # apply prompt template per sample
    dataset = dataset.map(template_dataset, remove_columns=list(dataset.features))

    # tokenize dataset
    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]), batched=True, remove_columns=list(dataset.features)
    )

    # chunk dataset
    lm_dataset = pack_dataset(dataset, chunk_length=2048)  # We use 2048 as the maximum length for packing

    # save train_dataset to disk
    lm_dataset.save_to_disk(dataset_path)


def training_function(script_args, training_args):
    # load dataset
    dataset = load_from_disk(script_args.dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)
    with lazy_load_for_parallelism(tensor_parallel_size=training_args.tensor_parallel_size):
        model = AutoModelForCausalLM.from_pretrained(script_args.model_id)

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,  # no special collator needed since we stacked the dataset
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
    dataset_path: Optional[str] = field(
        metadata={"help": "Path to the preprocessed and tokenized dataset."},
        default=None,
    )


def main():
    parser = HfArgumentParser([ScriptArguments, TrainingArguments])
    script_args, training_args = parser.parse_args_into_dataclasses()

    if script_args.dataset_path is None:
        create_and_save_dataset(script_args.model_id, "tokenized_dolly")
        script_args.dataset_path = "tokenized_dolly"

    # set seed
    set_seed(training_args.seed)

    # run training function
    training_function(script_args, training_args)


if __name__ == "__main__":
    main()
