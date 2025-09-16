# Training Tests

This directory contains comprehensive tests for Optimum Neuron's training functionality on AWS Trainium devices. The tests validate distributed training, custom model implementations, and various optimization techniques.

## Test Files Overview

### Core Training Tests

These tests cover fundamental training features and custom model implementations, ensuring compatibility with the Transformers library and Neuron's distributed training capabilities.

#### `test_custom_modeling.py`
Validates custom Neuron model implementations against original Transformers models:
- **Model accuracy**: Ensures custom implementations produce identical outputs to Transformers
- **Parallel attention**: Tests GQA (Grouped Query Attention) and QKV fusion implementations
- **Model resizing**: Validates token embedding resizing with tensor parallelism
- **Query indexing**: Tests query head distribution across tensor parallel ranks

Key test scenarios:
- `test_custom_modeling_matches_original`: Compares custom vs original model outputs
- `test_custom_model_resize_embedding`: Tests embedding layer resizing

Because these tests are ran on tiny random models, they are fast so we can run all the combinaisons of what we support (fused qkv, gqa qkv, flash attention, eager attention, etc).

#### `test_overfit.py`
End-to-end training validation through overfitting tests:
- **Convergence validation**: Ensures models can overfit simple datasets
- **LoRA training**: Validates Parameter-Efficient Fine-Tuning
- **Monitoring integration**: Uses Weights & Biases for training tracking

#### `test_checkpointing.py`
Tests checkpoint consolidation and distributed checkpoint handling:
- **Checkpoint consolidation**: Validates model parallel checkpoint merging for both regular and LoRA models
- **Parameter loading**: Ensures pipeline parallel ranks only load relevant parameters
- **Weight averaging**: Tests LoRA checkpoint consolidation with proper weight averaging

Key test scenarios:
- `test_consolidate_custom_model_parallel_checkpoints`: Tests checkpoint consolidation across parallel ranks
- `test_consolidate_custom_lora_model_parallel_checkpoints`: Tests LoRA checkpoint consolidation

#### `test_optimizer.py`
Tests optimizer behavior across different parallelization strategies:
- **Gradient accumulation**: Validates gradient accumulation with different accumulation steps
- **Gradient clipping**: Tests gradient clipping behavior with model parallelism
- **Optimizer step**: Validates optimizer behavior with various configurations

These series of tests cover most of the training features we support and want to validate:

    1. Custom model implementation is correct
    2. The custom model can be trained
    3. The checkpoints produced by the custom model can be converted back to a Transformers compatible checkpoint

### Specialized Component Tests

These tests focus on specific Neuronx-distributed components and utilities, ensuring they work correctly in distributed training scenarios.

#### `test_flash_attn.py`
Tests the NKI (Neuron Kernel Interface) flash attention implementation:
- **Numerical correctness**: Compares flash attention output against eager attention
- **Model compatibility**: Tests with different model configurations (Llama, Granite)
- **Precision handling**: Validates behavior with different floating-point types

#### `test_linears.py`
Validates parallel linear layer implementations:
- **ColumnParallelLinear**: Tests column-wise weight partitioning
- **RowParallelLinear**: Tests row-wise weight partitioning with input parallelism
- **Data type support**: Tests float32, bfloat16, and mixed precision scenarios
- **Numerical accuracy**: Ensures parallel outputs match sequential linear layers

#### `test_zero1.py`
Tests ZeRO-1 optimizer integration and mixed precision:
- **ZeRO-1 creation**: Validates ZeRO-1 optimizer setup with proper sharding groups
- **Master weights**: Tests master weight configuration and FP32 gradient accumulation
- **Training integration**: Tests ZeRO-1 with NeuronTrainingArguments
- **Mixed precision**: Validates ZeRO-1 behavior with different mixed precision modes

#### `test_mixed_precision.py`
Tests mixed precision training capabilities:
- **Configuration validation**: Tests MixedPrecisionConfig validation and error handling
- **Model preparation**: Validates model dtype handling for different precision modes (NO, FULL_BF16, AUTOCAST_BF16)
- **Trainer integration**: Tests mixed precision with NeuronTrainer and autocast context managers
- **Environment setup**: Validates stochastic rounding environment variable handling

### Trainer Implementation Tests

#### `test_neuron_trainer.py`
Tests the core `NeuronTrainer` implementation:
- **Basic training**: Validates fundamental training loop functionality
- **Distributed training**: Tests trainer behavior across different parallelization strategies
- **Configuration handling**: Tests training argument processing and validation

#### `test_neuron_sft_trainer.py`
Tests the specialized `NeuronSFTTrainer` for supervised fine-tuning:
- **SFT-specific features**: Validates supervised fine-tuning specific functionality
- **Integration testing**: Tests SFT trainer with various model configurations

#### `test_modeling_auto.py`
Tests automatic model loading and configuration:
- **Auto model selection**: Validates automatic model class selection for training
- **Configuration compatibility**: Tests model configuration handling across different architectures


## Test Infrastructure

### Base Classes
- **`DistributedTest`**: Base class for multi-worker distributed tests
- **`launch_procs`**: Utility for launching distributed test processes

### Parallelization Strategies
Tests cover various combinations of:
- **Data Parallel (DP)**: Multiple replicas of the model
- **Tensor Parallel (TP)**: Model weights split across devices
- **Pipeline Parallel (PP)**: Model layers split across devices

Common test configurations:
- `dp=2`: 2 data parallel workers
- `tp=2`: 2 tensor parallel workers
- `pp=2`: 2 pipeline parallel stages
- `dp=4,tp=2,pp=4`: Mixed parallelism (32 total workers)

### Utilities
- **Test markers**: `@is_trainium_test` ensures tests only run on Trainium hardware

## Running Tests

These tests require AWS Trainium instances to run. The tests are designed to:
1. Validate numerical correctness against CPU/transformers baselines
2. Test distributed training scenarios
3. Ensure checkpoint compatibility
4. Validate training convergence

### Example Test Execution
```bash
# Run all training tests (requires Trainium instance)
pytest tests/training/

# Run specific test file
pytest tests/training/test_custom_modeling.py

# Run with specific parallelization
pytest tests/training/test_common.py::TestCommonTrainingFeatures::test_optimizer_step
```
