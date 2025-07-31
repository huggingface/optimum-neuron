# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class ScriptArguments:
    """Arguments for fine-tuning BERT on emotion classification."""

    model_id: str = field(default="bert-base-uncased", metadata={"help": "Model ID from Hugging Face Hub"})
    output_dir: str = field(default="bert-emotion-model", metadata={"help": "Output directory for the model"})
    epochs: int = field(default=3, metadata={"help": "Number of training epochs"})
    learning_rate: float = field(default=5e-5, metadata={"help": "Learning rate"})
    batch_size: int = field(default=8, metadata={"help": "Training batch size per device"})
    max_length: int = field(default=128, metadata={"help": "Maximum sequence length"})
    seed: int = field(default=42, metadata={"help": "Random seed"})


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================
def prepare_dataset(tokenizer, max_length):
    """Load and tokenize the emotion dataset."""
    # Load dataset
    dataset = load_dataset("dair-ai/emotion")

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    # Apply tokenization
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset


# =============================================================================
# Main Training Function
# =============================================================================
def main():
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # Set seed for reproducibility
    set_seed(args.seed)
    print(f"🚀 Starting BERT fine-tuning with model: {args.model_id}")

    # Load tokenizer and prepare dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    dataset = prepare_dataset(tokenizer, args.max_length)

    # Load model
    num_labels = len(dataset["train"].features["label"].names)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_id, num_labels=num_labels)

    # Training configuration
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        bf16=True,
        do_train=True,
        save_strategy="epoch",
        logging_steps=100,
        overwrite_output_dir=True,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        processing_class=tokenizer,
    )

    # Train the model
    print("⚡ Training started...")
    trainer.train()

    # Save model
    trainer.save_model()
    print(f"✅ Training completed! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
