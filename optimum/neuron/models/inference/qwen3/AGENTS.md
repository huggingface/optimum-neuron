# Qwen3 Inference Model Guide

This directory contains the Neuron-optimized Qwen3 inference implementation. It follows [HF Transformers](https://github.com/huggingface/transformers) architecture with Neuron-specific changes derived from [NxDI](https://github.com/aws-neuron/neuronx-distributed-inference).

## HF reference
- Transformers Qwen3 modeling: https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3

## What differs vs HF
- **Parallel layers**: TP-aware linear/embedding layers replace HF modules.
- **Attention base**: `NeuronAttentionBase` for TP-aware heads and static shapes.
- **KV cache/static shapes**: Uses `KVCacheManager` and fixed shapes.

## What was removed from [NxDI](https://github.com/aws-neuron/neuronx-distributed-inference)

Reference: [NxDI Qwen3 implementation](https://github.com/aws-neuron/neuronx-distributed-inference/tree/main/src/neuronx_distributed_inference/models/qwen3)

Qwen3 follows Llama architecture with shared removals:

### Removed NKI Kernel Implementations
- **MLP TKG kernel** - Advanced small batch optimization
- **Quantized MLP kernels** - INT8 quantization with fused operations
- **NKI-based MLP** - Hand-tuned kernel implementations

**Why removed**: Requires unstable neuronxcc.nki APIs. Optimum uses compiler optimization.

### Removed Serving/Speculative Features
- **EAGLE draft models** - Speculative decoding support
- **Medusa heads** - Multi-head speculative sampling
- **Flash Decoding** - Multi-query decode parallelism
- **LoRA serving** - Dynamic adapter switching
- **Context parallelism** - MLP splitting for long contexts

**Why removed**: Production serving features handled by vLLM integration.

### Removed Configuration/Infrastructure
- **NxDI config classes** - Replaced by `NxDNeuronConfig`
- **Module registration system** - Plugin architecture
- **Preshard hooks** - Weight sharding orchestration
- **Custom RMSNorm variants** - CPU/NXD mode switching

**Why removed**: Optimum uses HF Transformers config with `neuron_config.json`.

### What Optimum Neuron Keeps
- Parallel layers with TP-aware sharding
- `NeuronAttentionBase` with GQA and flash attention
- Rotary embeddings with scaling
- State dict conversion (QKV fusion, vocab parallelism)
- Decoder layer composition

Optimum Neuron prioritizes **stability and HF ecosystem compatibility**.

## Key files
- [optimum/neuron/models/inference/qwen3/modeling_qwen3.py](modeling_qwen3.py)
- [optimum/neuron/models/inference/backend/modules/attention/attention_base.py](../backend/modules/attention/attention_base.py)
