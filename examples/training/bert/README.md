# BERT Text Classification Fine-tuning Example

This example demonstrates how to fine-tune BERT for text classification on AWS Trainium using the emotion dataset.

## Overview

This example fine-tunes a BERT model for emotion classification on the [emotion dataset](https://huggingface.co/datasets/dair-ai/emotion), which contains English Twitter messages labeled with six basic emotions: anger, fear, joy, love, sadness, and surprise.

## Files

- `fine_tune_bert.py` - Main training script with command-line interface
- `fine_tune_bert.sh` - Shell script for easy execution with recommended parameters
- `README.md` - This documentation file

## Requirements

- AWS Trainium instance (tested on `trn1.2xlarge`)
- Optimum Neuron with training dependencies installed:
  ```bash
  python -m pip install .[training]
  ```

## Usage

### Quick Start

Run the training with default parameters:

```bash
bash fine_tune_bert.sh
```

### Custom Training

Run with custom parameters:

```bash
torchrun --nproc_per_node=2 fine_tune_bert.py \
  --model_id bert-base-uncased \
  --learning_rate 5e-5 \
  --per_device_train_batch_size 8 \
  --epochs 3 \
  --output_dir my-bert-model
```

### Available Parameters

- `--model_id`: Hugging Face model ID (default: `bert-base-uncased`)
- `--learning_rate`: Learning rate (default: `5e-5`)
- `--per_device_train_batch_size`: Training batch size per device (default: `8`)
- `--per_device_eval_batch_size`: Evaluation batch size per device (default: `8`)
- `--epochs`: Number of training epochs (default: `3`)
- `--train_max_length`: Maximum sequence length for tokenization (default: `128`)
- `--output_dir`: Directory to save the trained model
- `--repository_id`: Hugging Face Hub repository ID for model upload
- `--seed`: Random seed for reproducibility (default: `42`)

## Expected Results

After training for 3 epochs, you should see results similar to:

```
***** train metrics *****
  epoch                    =        3.0
  eval_loss                =     0.1761
  eval_runtime             = 0:00:03.73
  eval_samples_per_second  =    267.956
  eval_steps_per_second    =     16.881
  total_flos               =  1470300GF
  train_loss               =     0.2024
  train_runtime            = 0:07:27.14
  train_samples_per_second =     53.674
  train_steps_per_second   =      6.709
```

## Performance

- **Training time**: ~7.5 minutes for 3 epochs
- **Cost**: Approximately $0.18 on `trn1.2xlarge` instance
- **Hardware utilization**: 2 Neuron cores with data parallelism

## Tutorial

For a detailed step-by-step guide, see the [Fine-tune BERT for Text Classification tutorial](https://huggingface.co/docs/optimum-neuron/training_tutorials/fine_tune_bert) in the Optimum Neuron documentation.