# Granite Inference Model Guide

This directory contains the Neuron-optimized Granite inference implementation. It follows [HF Transformers](https://github.com/huggingface/transformers) structure with only Neuron-specific graph changes from [NxDI](https://github.com/aws-neuron/neuronx-distributed-inference).

## HF reference
- Transformers Granite modeling: https://github.com/huggingface/transformers/tree/main/src/transformers/models/granite

## What differs vs HF
- **Parallel layers**: TP-aware `ColumnParallelLinear`/`RowParallelLinear` and `ParallelEmbedding`.
- **Attention base**: Uses `NeuronAttentionBase` utilities instead of HF attention.
- **Static shapes & KV cache**: Enforced by `neuron_config.json` and `KVCacheManager`.

## What was removed from [NxDI](https://github.com/aws-neuron/neuronx-distributed-inference)

Reference: [NxDI Granite implementation](https://github.com/aws-neuron/neuronx-distributed-inference/tree/main/src/neuronx_distributed_inference/models/granite)

Granite shares Llama's architecture and inherits similar removals:

### Removed NKI Kernel Implementations
- **MLP TKG kernel** - Small batch/sequence optimization with NKI primitives
- **Quantized MLP kernels** - INT8 quantization with fused RMSNorm-Quantize
- **NKI-based MLP** - Hand-optimized kernels using neuronxcc.nki APIs

**Why removed**: Requires unstable NKI private APIs. Optimum Neuron uses compiler auto-optimization.

### Removed Serving/Speculative Features
- **EAGLE/EAGLE3 draft models** - Speculative decoding support
- **Medusa heads** - Multi-head speculative sampling
- **Flash Decoding** - Multi-query decode parallelism
- **LoRA serving** - Dynamic adapter switching
- **Context parallelism (CP)** - MLP splitting for long contexts

**Why removed**: Production serving features delegated to vLLM integration.

### Removed Configuration/Infrastructure
- **NxDI config classes** (`InferenceConfig`, `NeuronConfig`) - Replaced by `NxDNeuronConfig`
- **Module registration** - Plugin architecture for implementation swapping
- **Preshard hooks** - Weight sharding orchestration
- **Custom RMSNorm variants** - CPU/NXD mode switching

**Why removed**: Optimum uses HF Transformers config base extended via `neuron_config.json`.

### What Optimum Neuron Keeps
- Core parallel layers (`ColumnParallelLinear`, `RowParallelLinear`, `ParallelEmbedding`)
- `NeuronAttentionBase` with GQA and flash attention
- Rotary embeddings with RoPE scaling
- State dict conversion (QKV fusion, rank tensors)
- Model structure (decoder layers, embeddings, norms)

Optimum Neuron prioritizes **stability and HF compatibility** over experimental optimizations.

## Key files
- [optimum/neuron/models/inference/granite/modeling_granite.py](modeling_granite.py)
- [optimum/neuron/models/inference/backend/modules/attention/attention_base.py](../backend/modules/attention/attention_base.py)
