# Llama4 Inference Model Guide

This directory contains the Neuron-optimized Llama4 inference implementation, aligned to [HF Transformers](https://github.com/huggingface/transformers) with Neuron-specific changes from [NxDI](https://github.com/aws-neuron/neuronx-distributed-inference).

## HF reference
- Transformers Llama4 modeling: https://github.com/huggingface/transformers/tree/main/src/transformers/models/llama4

## What differs vs HF
- **Parallel layers**: TP-aware linear/embedding layers replace HF modules.
- **Attention base**: `NeuronAttentionBase` enforces TP head counts and static shapes.
- **Static shapes & KV cache**: Uses `KVCacheManager` and fixed sequence shapes from `neuron_config.json`.

## What was removed from [NxDI](https://github.com/aws-neuron/neuronx-distributed-inference)

Reference: [NxDI Llama4 implementation](https://github.com/aws-neuron/neuronx-distributed-inference/tree/main/src/neuronx_distributed_inference/models/llama4)

Llama4 follows Llama architecture with shared removals:

### Removed NKI Kernel Implementations
- **MLP TKG kernel** - Advanced small batch optimization using NKI compiler primitives
- **Quantized MLP kernels** - INT8 quantization with fused operations
- **NKI-based standard MLP** - Hand-tuned MLP with optional fused residual add

**Why removed**: Requires neuronxcc.nki private APIs not stabilized for external use. Optimum relies on compiler auto-optimization.

### Removed Serving/Speculative Decoding
- **EAGLE draft models** - Speculative decoding with draft models
- **Medusa heads** - Multi-head speculative sampling
- **Flash Decoding** - Multi-query parallelism for decode phase
- **LoRA serving** - Dynamic adapter switching during inference
- **Context parallelism** - MLP splitting across cores for long contexts

**Why removed**: Advanced serving features for production deployments. vLLM integration handles production needs.

### Removed Configuration/Utilities
- **NxDI-specific configs** - Replaced by Optimum's `NxDNeuronConfig`
- **Module registration system** - Plugin architecture for swapping implementations
- **Preshard hooks** - Weight sharding orchestration before init
- **Custom RMSNorm variants** - CPU vs NXD mode switching
- **Quantization helpers** - Per-layer quantization control

**Why removed**: Optimum uses HF config base with `neuron_config.json` extensions.

### What Optimum Neuron Keeps
- Parallel layers with TP-aware sharding
- `NeuronAttentionBase` with GQA, flash attention, static KV cache
- Rotary embeddings with long context scaling
- State dict conversion (QKV fusion, vocab parallelism)
- Decoder layer composition

Optimum Neuron prioritizes **stability, maintainability, and HF ecosystem compatibility**.

## Key files
- [optimum/neuron/models/inference/llama4/modeling_llama4.py](modeling_llama4.py)
- [optimum/neuron/models/inference/backend/modules/attention/attention_base.py](../backend/modules/attention/attention_base.py)
