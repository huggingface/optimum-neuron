import argparse

from datasets import load_dataset
from huggingface_hub import HfFolder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument
    parser.add_argument(
        "--model_id",
        type=str,
        default="bert-base-uncased",
        help="Model id to use for training.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory to save the model.",
    )
    parser.add_argument(
        "--repository_id",
        type=str,
        default=None,
        help="Hugging Face Repository id for uploading models",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for.")
    parser.add_argument("--max_steps", type=int, default=-1, help="Number of steps to train for.")
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size to use for training.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size to use for validation.",
    )
    parser.add_argument(
        "--train_max_length",
        type=int,
        default=128,
        help="Maximum length of tokens to be used for training.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate to use for training.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for training.")
    parser.add_argument(
        "--hf_token",
        type=str,
        default=HfFolder.get_token(),
        help="Token to use for uploading models to Hugging Face Hub.",
    )
    args = parser.parse_known_args()
    return args


def training_function(args):
    # set seed
    set_seed(args.seed)

    # Load the dataset
    emotions = load_dataset("dair-ai/emotion")
    model_id = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Tokenize the dataset
    def tokenize_function(example):
        ret = tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=args.train_max_length,
        )
        return ret

    tokenized_emotions = emotions.map(tokenize_function, batched=True)

    num_labels = len(emotions["train"].features["label"].names)

    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
    )

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"{model_id}-finetuned"

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        do_train=True,
        bf16=True,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=500,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        # push to hub parameters
        push_to_hub=True if args.repository_id else False,
        hub_strategy="every_save",
        hub_model_id=args.repository_id if args.repository_id else None,
        hub_token=args.hf_token,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_emotions["train"],
        eval_dataset=tokenized_emotions["validation"],
        processing_class=tokenizer,
    )

    # Train the model
    train_result = trainer.train()
    metrics = train_result.metrics

    eval_dataset = tokenized_emotions["validation"]
    eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)
    metrics.update(eval_metrics)
    trainer.log_metrics("train", metrics)

    trainer.save_model(output_dir)
    trainer.create_model_card()
    if args.repository_id:
        trainer.push_to_hub(repository_id=args.repository_id, token=args.hf_token)


def main():
    args, _ = parse_args()
    training_function(args)


if __name__ == "__main__":
    main()