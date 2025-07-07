# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Optimum Neuron is the interface between the HuggingFace Transformers library and AWS Accelerators (Trainium and Inferentia). It provides tools for model loading, training, and inference on single- and multi-accelerator settings.

## Common Development Commands

### Code Quality and Formatting
```bash
# Run code quality checks
make style_check
ruff check .
ruff format . --diff

# Fix code style issues
make style
ruff check . --fix
ruff format .
```

To ensure code quality and style, check pyproject.toml for the configuration of `ruff`.
You can also check my vim configuration at ~/.config/nvim/init.lua to see how I like to format my python code.

### Testing

Most of the tests require a Neuron device to run, so it is not possible to run them on the machine running Claude code. 
Instead, provide a test script that can be run on an AWS Trainium instance along with the command line to execute it.
I will run the tests on my AWS Trainium instance and provide you with the results.

### Build

```bash

### Building and Distribution
```bash
# Build distribution packages
make build_dist
python -m build

# Create example scripts from Transformers
make transformers_examples

# Clean build artifacts
make clean
```

## Architecture Overview

### Core Components

**Main Module Structure (`optimum/neuron/`)**:
- `modeling.py` - Core Neuron model classes (NeuronModelForXXX) that wrap Transformers models for inference
- `modeling_traced.py` - Base traced model implementation (`NeuronTracedModel`)
- `modeling_decoder.py` - Decoder-specific models (`NeuronModelForCausalLM`) 
- `modeling_diffusion.py` - Stable Diffusion pipelines for Neuron
- `modeling_seq2seq.py` - Sequence-to-sequence models
- `trainers.py` - Training classes (`NeuronTrainer`, `NeuronSFTTrainer`, etc.), it is a bit dirty and would benefit refactoring
- `training_args.py` - Training argument classes

**Key Subsystems**:

- **`models/`** - Architecture-specific implementations
  - `auto_model.py` - Auto model loading
  - `bert/`, `clip/`, `whisper/`, `yolos/` - Model-specific classes
  - `inference/` - Inference backends (HLO, NXD)
  - `training/` - Training-specific model implementations, it is the subpackage for training models with Trainium. It provides custom implementations of models with the same name as their Transformers counterparts, optimized for Neuron devices, with support for distributed training and checkpoint consolidation.

- **`cache/`** - Compilation caching system
  - `hub_cache.py` - HuggingFace Hub integration for cached models
  - `traced.py` - Traced model caching

- **`generation/`** - Text generation utilities
- **`pipelines/`** - Pipeline implementations for diffusers and transformers
- **`peft/`** - Parameter-Efficient Fine-Tuning (LoRA) support for training, compatible with custom modeling implementations defined in `optimum.neuron.models.training`
- **`accelerate/`** - HuggingFace Accelerate integration, compatible with both regular Transformers models and custom Neuron implementations in `optimum.neuron.models.training`
- **`utils/`** - Utilities and helper functions, it is a bit of a mess and would benefit refactoring.

### Model Loading Pattern for Inference

The library follows a consistent pattern where `NeuronModelForXXX` classes mirror their Transformers counterparts but are optimized for Neuron devices:

```python
# Standard pattern
from optimum.neuron import NeuronModelForSequenceClassification
model = NeuronModelForSequenceClassification.from_pretrained("path/to/neuron/model")
```

### Model Loading Pattern for Training

The library provides custom implementation of models with the same name as their Transformers counterparts under `optimum.neuron.models.training`:

```python
from optimum.neuron.models.training import LlamaForCausalLM
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
```

These classes inherint from `NeuronModelMixin` defined in `optimum/neuron/models/training/modeling_utils.py` that defines:
-  `from_pretrained`: loads regular model weights from the Hugging Face Hub into the custom model implementation
-  `save_pretrained`: saves custom model weights with metadata to consolidate checkpoints to be usable by everyone on the Hugging Face Hub

### Training vs Inference

- **Training**: Use `NeuronTrainer` classes with distributed training support
- **Inference**: Use compiled/traced models via `NeuronModelForXXX` classes
- **Export**: Models must be exported to Neuron format before inference

### Distributed Training

The distributed training is handled as follows:
- Tensor Parallelism is directly implemented in the custom modeling code
- Pipeline Parallelism is handled by the `NeuronAccelerator.prepare_model` method
- Automatic checkpoint consolidation is handled thanks to the `ModelWeightTransformationSpec` classes and the `optimum.neuron.distributed.checkpointing` module

### Dependencies

Key dependencies managed in `setup.py`:
- `transformers ~= 4.51.0` 
- `accelerate == 0.29.2`
- `optimum ~= 1.23.3`
- `peft==0.14.0` for LoRA fine-tuning
- `trl==0.11.4` for reinforcement learning

### Test Structure

Tests are organized by functionality:
- `tests/decoder/` - Decoder model tests
- `tests/generation/` - Text generation tests
- `tests/inference/` - Inference tests
- `tests/training` - Training related tests, read the @tests/training/README.md in this directory for more information, and update it as changes are made if relevant.

Use pytest markers to run specific test categories based on hardware requirements.
