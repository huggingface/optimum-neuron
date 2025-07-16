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

#### `test_common.py`
Tests fundamental training features across different parallelization strategies:
- **Optimizer validation**: Tests gradient accumulation, clipping, and ZeRO-1 optimization
- **Checkpoint consolidation**: Validates model parallel checkpoint merging for both regular and LoRA models
- **Parameter loading**: Ensures pipeline parallel ranks only load relevant parameters

Key test scenarios:
- `test_optimizer_step`: Validates optimizer behavior with gradient accumulation and clipping
- `test_consolidate_custom_model_parallel_checkpoints`: Tests checkpoint consolidation across parallel ranks
- `test_consolidate_custom_lora_model_parallel_checkpoints`: Tests LoRA checkpoint consolidation with weight averaging


These series of tests cover most of the training features we support and want to validate:

    1. Custom model implementation is correct
    2. The custom model can be trained
    3. The checkpoints produced by the custom model can be converted back to a Transformers compatible checkpoint

### Specialized Component Tests

These tests focus on specfic Neuronx-distributed compotents and utilities, ensuring they work correctly in distributed training scenarios.

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


### Other Tests

#### `test_trainers.py`
Tests the various `NeuronTrainer`s implementation. It needs to be reworked.


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
