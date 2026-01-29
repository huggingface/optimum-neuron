# SmolLM3 Inference Model Guide

This directory contains the Neuron-optimized SmolLM3 inference implementation. It follows [HF Transformers](https://github.com/huggingface/transformers) architecture with Neuron-specific changes from [NxDI](https://github.com/aws-neuron/neuronx-distributed-inference).

## HF reference
- Transformers SmolLM3 modeling: https://github.com/huggingface/transformers/tree/main/src/transformers/models/smollm3

## What differs vs HF
- **Parallel layers**: TP-aware linear/embedding layers replace HF modules.
- **Attention base**: `NeuronAttentionBase` for TP-aware heads and static shapes.
- **KV cache/static shapes**: Uses `KVCacheManager` and fixed shapes.

## What was removed from [NxDI](https://github.com/aws-neuron/neuronx-distributed-inference)

Reference: [NxDI SmolLM3 implementation](https://github.com/aws-neuron/neuronx-distributed-inference/tree/main/src/neuronx_distributed_inference/models/smollm3)

SmolLM3 follows Llama architecture with shared removals:

### Removed NKI Kernel Implementations
- **MLP TKG kernel** - Small batch optimization (particularly relevant for small models)
- **Quantized MLP kernels** - INT8 quantization with fused operations
- **NKI-based MLP** - Hand-tuned kernel implementations

**Why removed**: Requires neuronxcc.nki private APIs. Optimum relies on compiler auto-optimization.

### Removed Serving/Speculative Features
- **EAGLE draft models** - Speculative decoding (SmolLM3 often used as draft model itself)
- **Medusa heads** - Multi-head speculative sampling
- **Flash Decoding** - Multi-query parallelism
- **LoRA serving** - Dynamic adapter switching
- **Context parallelism** - MLP splitting for long contexts

**Why removed**: Production serving features handled by vLLM integration.

### Removed Configuration/Infrastructure
- **NxDI config classes** - Replaced by Optimum's `NxDNeuronConfig`
- **Module registration system** - Plugin architecture
- **Preshard hooks** - Weight sharding orchestration
- **Custom RMSNorm variants** - CPU/NXD mode switching

**Why removed**: Optimum uses HF Transformers config with `neuron_config.json` extensions.

### What Optimum Neuron Keeps
- Parallel layers with TP-aware sharding
- `NeuronAttentionBase` with GQA and flash attention
- Rotary embeddings
- State dict conversion (QKV fusion)
- Decoder layer structure

Optimum Neuron prioritizes **stability and HF compatibility** for SmolLM3 models.

## Key files
- [optimum/neuron/models/inference/smollm3/modeling_smollm3.py](modeling_smollm3.py)
- [optimum/neuron/models/inference/backend/modules/attention/attention_base.py](../backend/modules/attention/attention_base.py)
