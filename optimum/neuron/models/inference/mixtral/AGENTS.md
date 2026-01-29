# Mixtral Inference Model Guide (MoE)

This directory contains the Neuron-optimized Mixtral (MoE) inference implementation. It follows [HF Transformers](https://github.com/huggingface/transformers) architecture with Neuron-specific changes derived from [NxDI](https://github.com/aws-neuron/neuronx-distributed-inference).

## HF reference
- Transformers Mixtral modeling: https://github.com/huggingface/transformers/tree/main/src/transformers/models/mixtral

## What differs vs HF
- **Expert parallelism**: Routing and expert weights are sharded for TP/EP.
- **Static expert shapes**: Router logits and capacity are compiled into fixed shapes.
- **Parallel MLP/FFN**: Expert MLPs use parallel layers instead of per-expert `nn.Linear` stacks.
- **KV cache/static shapes**: Uses `KVCacheManager` and fixed `sequence_length`/`batch_size`.

## What was removed from [NxDI](https://github.com/aws-neuron/neuronx-distributed-inference)

Reference: [NxDI Mixtral implementation](https://github.com/aws-neuron/neuronx-distributed-inference/tree/main/src/neuronx_distributed_inference/models/mixtral)

Mixtral is a Mixture-of-Experts (MoE) model with additional MoE-specific removals:

### Removed NKI Kernel Implementations
- **MLP TKG kernel** - Small batch optimization for expert MLPs
- **Quantized MLP kernels** - INT8 quantization for experts
- **NKI-based expert MLP** - Hand-optimized expert computation
- **Fused expert routing kernels** - Combined routing + expert execution

**Why removed**: Requires unstable NKI APIs. Compiler handles expert optimization automatically.

### Removed MoE-Specific Serving Features
- **Dynamic expert capacity** - Runtime expert token allocation
- **Expert load balancing hooks** - Custom load balancing strategies
- **Expert caching** - Cached expert activations for repeated tokens
- **Hierarchical expert parallelism** - Multi-level EP/TP decomposition
- **EAGLE/Medusa with MoE** - Speculative decoding for MoE models

**Why removed**: Complex serving optimizations for production MoE deployments. Focus on foundational MoE export/inference.

### Removed Configuration/Infrastructure
- **NxDI MoE configs** - Custom expert parallelism configuration
- **Expert module registration** - Plugin system for expert implementations
- **Preshard hooks for experts** - Expert weight distribution logic
- **Expert quantization helpers** - Per-expert quantization control

**Why removed**: Optimum uses standardized MoE configuration via `neuron_config.json`.

### What Optimum Neuron Keeps
- Expert parallel layers with TP/EP sharding
- Static expert routing with compiled shapes
- `NeuronAttentionBase` with GQA and flash attention
- Expert MLP parallelization
- State dict conversion for expert weights
- Router logits computation

Optimum Neuron provides **stable foundational MoE support** while delegating advanced serving to vLLM.

## Key files
- [optimum/neuron/models/inference/mixtral/modeling_mixtral.py](modeling_mixtral.py)
- [optimum/neuron/models/inference/backend/modules/attention/attention_base.py](../backend/modules/attention/attention_base.py)
