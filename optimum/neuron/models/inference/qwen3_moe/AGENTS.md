# Qwen3 MoE Inference Model Guide

This directory contains the Neuron-optimized Qwen3 MoE inference implementation. It follows [HF Transformers](https://github.com/huggingface/transformers) architecture with Neuron-specific changes from [NxDI](https://github.com/aws-neuron/neuronx-distributed-inference).

## HF reference
- Transformers Qwen3 MoE modeling: https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_moe

## What differs vs HF
- **Expert parallelism**: Routing and expert weights are sharded for TP/EP.
- **Static expert shapes**: Router logits and capacity are compiled into fixed shapes.
- **Parallel MLP/FFN**: Expert MLPs use parallel layers or fused kernels.
- **KV cache/static shapes**: Uses `KVCacheManager` and fixed shapes.

## What was removed from [NxDI](https://github.com/aws-neuron/neuronx-distributed-inference)

Reference: [NxDI Qwen3 MoE implementation](https://github.com/aws-neuron/neuronx-distributed-inference/tree/main/src/neuronx_distributed_inference/models/qwen3_moe)

Qwen3 MoE is a Mixture-of-Experts model with MoE-specific removals:

### Removed NKI Kernel Implementations
- **MLP TKG kernel for experts** - Small batch optimization for expert MLPs
- **Quantized expert kernels** - INT8 quantization for expert computation
- **NKI-based expert MLP** - Hand-optimized expert implementations
- **Fused routing + execution** - Combined router and expert kernels

**Why removed**: Requires unstable NKI APIs. Compiler handles MoE optimization.

### Removed MoE-Specific Serving Features
- **Dynamic expert capacity** - Runtime token allocation to experts
- **Expert load balancing** - Custom balancing strategies
- **Expert caching** - Cached activations for repeated patterns
- **Hierarchical EP/TP** - Multi-level expert parallelism
- **EAGLE/Medusa for MoE** - Speculative decoding with experts

**Why removed**: Complex MoE serving optimizations for production. Focus on foundational support.

### Removed Configuration/Infrastructure
- **NxDI MoE configs** - Custom expert parallelism settings
- **Expert module registration** - Plugin system for experts
- **Expert preshard hooks** - Weight distribution for experts
- **Per-expert quantization** - Fine-grained quantization control

**Why removed**: Optimum uses standardized config via `neuron_config.json`.

### What Optimum Neuron Keeps
- Expert parallel layers with TP/EP sharding
- Static expert routing with fixed shapes
- `NeuronAttentionBase` with GQA and flash attention
- Expert MLP parallelization
- State dict conversion for expert weights
- Shared expert computation

Optimum Neuron provides **stable foundational MoE support** with vLLM for advanced serving.

## Key files
- [optimum/neuron/models/inference/qwen3_moe/modeling_qwen3_moe.py](modeling_qwen3_moe.py)
- [optimum/neuron/models/inference/backend/modules/attention/attention_base.py](../backend/modules/attention/attention_base.py)
