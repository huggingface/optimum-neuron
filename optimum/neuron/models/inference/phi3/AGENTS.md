# Phi-3 Inference Model Guide

This directory contains the Neuron-optimized Phi-3 inference implementation, aligned with [HF Transformers](https://github.com/huggingface/transformers) and limited to Neuron-specific graph changes.

## HF reference
- Transformers Phi-3 modeling: https://github.com/huggingface/transformers/tree/main/src/transformers/models/phi3

## What differs vs HF
- **Parallel layers**: TP-aware linear/embedding layers replace HF modules.
- **Attention base**: `NeuronAttentionBase` utilities used for TP-aware attention.
- **Static shapes & KV cache**: Uses `KVCacheManager` and fixed shapes.

## What was removed from [NxDI](https://github.com/aws-neuron/neuronx-distributed-inference)

Reference: [NxDI Phi-3 implementation](https://github.com/aws-neuron/neuronx-distributed-inference/tree/main/src/neuronx_distributed_inference/models/phi3)

Phi-3 uses Llama-like architecture with shared removals:

### Removed NKI Kernel Implementations
- **MLP TKG kernel** - Small batch optimization with NKI primitives
- **Quantized MLP kernels** - INT8 quantization with fused operations
- **NKI-based MLP** - Hand-tuned kernel implementations

**Why removed**: Requires unstable neuronxcc.nki private APIs. Optimum uses compiler optimization.

### Removed Serving/Speculative Features
- **EAGLE draft models** - Speculative decoding support
- **Medusa heads** - Multi-head speculative sampling
- **Flash Decoding** - Multi-query decode parallelism
- **LoRA serving** - Dynamic adapter switching
- **Context parallelism** - MLP splitting for long contexts

**Why removed**: Production serving features handled by vLLM integration.

### Removed Configuration/Infrastructure
- **NxDI config classes** - Replaced by Optimum's `NxDNeuronConfig`
- **Module registration system** - Plugin architecture
- **Preshard hooks** - Weight sharding orchestration
- **Custom layer norm variants** - CPU/NXD mode switching (Phi-3 uses LayerNorm)

**Why removed**: Optimum uses HF Transformers config with `neuron_config.json` extensions.

### What Optimum Neuron Keeps
- Parallel layers (`ColumnParallelLinear`, `RowParallelLinear`, `ParallelEmbedding`)
- `NeuronAttentionBase` with GQA and flash attention
- Rotary embeddings (SuRoPE for Phi-3)
- State dict conversion (QKV fusion, padding)
- Decoder layer structure

Optimum Neuron prioritizes **stability and HF ecosystem compatibility** for Phi-3 models.

## Key files
- [optimum/neuron/models/inference/phi3/modeling_phi3.py](modeling_phi3.py)
- [optimum/neuron/models/inference/backend/modules/attention/attention_base.py](../backend/modules/attention/attention_base.py)
