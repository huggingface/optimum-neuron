# Qwen2 Inference Model Guide

This directory contains the Neuron-optimized Qwen2 inference implementation. It follows [HF Transformers](https://github.com/huggingface/transformers) architecture with only Neuron-specific changes from [NxDI](https://github.com/aws-neuron/neuronx-distributed-inference).

## HF reference
- Transformers Qwen2 modeling: https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen2

## What differs vs HF
- **Parallel layers**: TP-aware linear/embedding layers replace HF modules.
- **Attention base**: `NeuronAttentionBase` for TP-aware heads and static shapes.
- **KV cache/static shapes**: Uses `KVCacheManager` and fixed shapes from `neuron_config.json`.

## What was removed from [NxDI](https://github.com/aws-neuron/neuronx-distributed-inference)

Reference: [NxDI Qwen2 implementation](https://github.com/aws-neuron/neuronx-distributed-inference/tree/main/src/neuronx_distributed_inference/models/qwen2)

Qwen2 follows Llama-like architecture with shared removals:

### Removed NKI Kernel Implementations
- **MLP TKG kernel** - Small batch/sequence optimization
- **Quantized MLP kernels** - INT8 quantization with fused RMSNorm-Quantize
- **NKI-based MLP** - Hand-optimized MLP implementations

**Why removed**: Requires neuronxcc.nki private APIs. Optimum relies on compiler auto-optimization.

### Removed Serving/Speculative Features
- **EAGLE draft models** - Speculative decoding
- **Medusa heads** - Multi-head speculative sampling
- **Flash Decoding** - Multi-query parallelism
- **LoRA serving** - Dynamic adapter switching
- **Context parallelism** - MLP splitting for long contexts

**Why removed**: Production serving delegated to vLLM integration.

### Removed Configuration/Infrastructure
- **NxDI config classes** - Replaced by `NxDNeuronConfig`
- **Module registration** - Plugin architecture
- **Preshard hooks** - Weight sharding orchestration
- **Custom RMSNorm variants** - CPU/NXD mode switching

**Why removed**: Optimum uses HF config base with `neuron_config.json`.

### What Optimum Neuron Keeps
- Parallel layers with TP-aware sharding
- `NeuronAttentionBase` with GQA and flash attention
- Rotary embeddings
- State dict conversion (QKV fusion)
- Decoder layer structure

Optimum Neuron prioritizes **stability and HF compatibility** for Qwen2 models.

## Key files
- [optimum/neuron/models/inference/qwen2/modeling_qwen2.py](modeling_qwen2.py)
- [optimum/neuron/models/inference/backend/modules/attention/attention_base.py](../backend/modules/attention/attention_base.py)
