# Llama Inference Model Guide

This directory contains the Neuron-optimized Llama inference implementation. It is based on [HF Transformers](https://github.com/huggingface/transformers) and trimmed from [NxDI](https://github.com/aws-neuron/neuronx-distributed-inference) to include only Neuron-specific graph changes.

## HF reference
- Transformers Llama modeling: https://github.com/huggingface/transformers/tree/main/src/transformers/models/llama

## What differs vs HF
- **Parallel layers**: HF `nn.Linear`/`Embedding` are replaced with `ColumnParallelLinear`, `RowParallelLinear`, `ParallelEmbedding`.
- **Attention base**: HF attention is replaced with `NeuronAttentionBase` for TP-aware heads and static shapes.
- **KV cache/static shapes**: Uses `KVCacheManager` and fixed `sequence_length`/`batch_size` from `neuron_config.json`.
- **State dict remaps**: QKV fusion/padding helpers (e.g., `convert_state_dict_to_fused_qkv`).

## What was removed from [NxDI](https://github.com/aws-neuron/neuronx-distributed-inference)

Reference: [NxDI Llama modeling_llama.py](https://github.com/aws-neuron/neuronx-distributed-inference/blob/main/src/neuronx_distributed_inference/models/llama/modeling_llama.py)

The Optimum Neuron port removes NxDI-specific components that are either experimental, infrastructure-specific, or replaced by Optimum's own systems:

### Removed NKI Kernel Implementations
- **MLP TKG (Tensor-Kernel-Grid) kernel** (`_trace_nki_mlp_tkg_kernel`, `_kernel_enabled_nki_mlp_tkg`) - Advanced small batch/sequence optimization using NKI compiler primitives with CC pipelining
- **Quantized MLP kernel integration** (`_kernel_enabled_quantized_mlp`, `rmsnorm_quant_isa_kernel`) - INT8 quantization with fused RMSNorm-Quantize for reduced memory bandwidth
- **Standard MLP kernel** (`_kernel_enabled_mlp`, `mlp_isa_kernel`, `mlp_fused_add_isa_kernel`) - NKI-based MLP with optional fused residual add
- **Quantization infrastructure**: `QuantizationType`, activation quantization, clamp bounds, quantized linear layer preprocessing

**Why removed**: These kernels require neuronxcc.nki private/pre-prod APIs not stabilized for external use. Optimum Neuron relies on stable NeuronX compiler auto-optimization rather than hand-tuned NKI kernels. Users needing these optimizations should use NxDI directly.

### Removed Serving/Speculative Decoding Features
- **EAGLE draft model support** (`is_eagle_draft`, `is_eagle3_draft`, `enable_eagle_draft_input_norm`, `hidden_norm`, `WeightGatheredColumnParallel`) - Speculative decoding with draft models
- **Medusa heads** (`is_medusa`, `num_medusa_heads`, `medusa_speculation_length`, `ResBlock`) - Multi-head speculative sampling
- **Flash Decoding** (`flash_decoding_enabled`, `calculate_num_cores_per_group`, `num_cores_per_group`) - Multi-query parallelism for decode phase
- **LoRA serving** (`is_lora_module`, adapter-aware forward paths) - Dynamic LoRA adapter switching during inference
- **Context parallelism (CP)** for MLP (`mlp_cp_degree`, `cte_mlp`, `tkg_mlp`, context parallel process groups) - Splits MLP across additional cores for long context encoding

**Why removed**: These are advanced serving features for production deployments. Optimum Neuron focuses on foundational model export/inference; vLLM integration handles production serving needs. Users requiring EAGLE/Medusa should use NxDI or wait for vLLM integration.

### Removed Configuration/Utility Code
- **NxDI-specific config classes** (`InferenceConfig`, `NeuronConfig`, `LlamaInferenceConfig`) - Replaced by Optimum's `NxDNeuronConfig`
- **Module registration system** (`_LLAMA_MODULE_MAP`, `_register_module`, `@register_module`) - NxDI's plugin architecture for swapping implementations
- **Quantization config helpers** (`get_modules_to_not_convert`, `get_updated_configs`) - Per-layer quantization control
- **Custom RMSNorm variants** (`LlamaRMSNormPadded`, `LlamaCustomRMSNormPadded`, `get_rmsnorm_cls`) - CPU vs NXD mode switching and padding handling
- **Preshard hooks** (`preshard_hook_fn`, per-module `preshard_hook`) - Weight sharding orchestration before model initialization
- **Sequence parallel mappings** (`reduce_scatter_to_sequence_parallel_region_tiled`, tile_cc control) - Advanced sequence parallelism collectives

**Why removed**: Optimum Neuron uses HF Transformers config as base and extends via `neuron_config.json`. NxDI's config system is replaced by Optimum's standardized approach. Preshard hooks are handled by Optimum's state dict conversion. Sequence parallel features may be added in future releases.

### Removed Experimental/Debug Features
- **Decorator peeling** (`peel_decorations`) - Workaround for NKI kernel tracing bugs
- **Platform target detection** (`get_platform_target`) - trn1 vs inf2 conditional paths
- **Kernel feature flags** (`MLP_TKG_FP8_ENABLED`, `use_quantization_type`) - Backward compatibility checks for kernel APIs
- **NKI grid configuration** (`CCPipeline`, `nc(logical_nc_config)`, `seq_len_threshold_for_cc_tiling`) - Manual NeuronCore allocation and tiling
- **Weight transpose utilities** (`transpose_parallel_linear_layer`) - For quantized weight layouts

**Why removed**: These are implementation details for NxDI's experimental kernel path. Optimum Neuron avoids exposing internal compiler/kernel internals, using higher-level abstractions instead.

### What Optimum Neuron Keeps from NxDI
- **Core parallel layer usage**: `ColumnParallelLinear`, `RowParallelLinear`, `ParallelEmbedding` with TP-aware sharding
- **Attention architecture**: `NeuronAttentionBase` with GQA support, flash attention, static KV cache
- **Rotary embeddings**: `RotaryEmbedding`, `Llama3RotaryEmbedding` for RoPE with long context scaling
- **State dict conversion**: QKV fusion (`convert_state_dict_to_fused_qkv`), rank tensor injection, vocab parallelism handling
- **Model structure**: Decoder layer composition (attention + MLP), embedding/LM head, layer norms

The Optimum Neuron implementation prioritizes **stability, maintainability, and HF ecosystem compatibility** over cutting-edge performance optimizations. For production deployments requiring maximum throughput, NxDI remains the reference implementation.

## Key files
- [optimum/neuron/models/inference/llama/modeling_llama.py](modeling_llama.py)
- [optimum/neuron/models/inference/backend/modules/attention/attention_base.py](../backend/modules/attention/attention_base.py)
