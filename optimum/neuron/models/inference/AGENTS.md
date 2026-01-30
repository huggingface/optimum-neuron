# Optimum Neuron Inference Models (NxD) Guide

This guide focuses on NxD inference models, decoder graphs, attention, and porting practices. For project-wide workflows see [AGENTS.md](../../../../AGENTS.md).

## NxD Decoder Models

### Three-Graph Architecture
- Context encoding graph: multi-token prompt → KV cache
- Token generation graph: 1 token in → 1 token out
- Speculation graph: optional multi-token proposal

Implementation details live in:
- [optimum/neuron/models/inference/backend/modules/decoder/modeling_decoder.py](backend/modules/decoder/modeling_decoder.py)
- [optimum/neuron/models/inference/backend/modules/decoder/decoder_builders.py](backend/modules/decoder/decoder_builders.py)
- [optimum/neuron/models/inference/backend/modules/decoder/decoder_wrappers.py](backend/modules/decoder/decoder_wrappers.py)

### KV Cache Management
KV cache is managed by `KVCacheManager` with BHSD layout and in-place aliasing:
- [optimum/neuron/models/inference/backend/modules/kvcache/kv_cache_manager.py](backend/modules/kvcache/kv_cache_manager.py)

### On-Device Sampling
Sampling on NeuronCores uses `nxd_topk`, `nxd_argmax`, NKI cumsum kernels:
- [optimum/neuron/models/inference/backend/modules/generation/sampling.py](backend/modules/generation/sampling.py)

### State Dict Conversion

**Method**: `convert_hf_to_neuron_state_dict()`

Converts HuggingFace checkpoint format to Neuron MoE format during model loading.

### Common Pitfalls
- Runtime shapes must match compiled shapes.
- Call context encoding before token generation.
- Weights NOT saved with export (HF Hub used as source).
- Export saves compiled model.pt with HF-format configs
- State dict conversion happens during `from_pretrained()`
- TP degree must match compiled model.
- Decoder graph changes require cache prune: `python tools/prune_test_models.py`.

## Attention Mechanisms

### Grouped Query Attention (GQA)
- Sharding strategy selection: REPLICATE_TO_TP_DEGREE vs CONVERT_TO_MHA
- Logic in [optimum/neuron/models/inference/backend/modules/attention/gqa.py](backend/modules/attention/gqa.py)

### Flash Attention on Neuron
- NKI kernel in inference: [optimum/neuron/models/inference/backend/modules/attention/attention_base.py](backend/modules/attention/attention_base.py)
- Training path uses `attn_implementation="flash_attention_2"`.

### Parallel Attention Layers
Parallel QKV and output projections use `ColumnParallelLinear`/`RowParallelLinear` in:
- [optimum/neuron/models/inference/backend/modules/attention/attention_base.py](backend/modules/attention/attention_base.py)

## Neuron vs HF Modeling Differences

### Llama-like models (decoder-only)
- Replace HF `nn.Linear`/`Embedding` with TP-aware parallel layers.
- Replace HF attention with `NeuronAttentionBase` for static shapes.
- Use `KVCacheManager` instead of HF dynamic cache.
- Optional fused QKV/MLP kernels (Neuron-only).
- State dict remaps (e.g., QKV concatenation).

See reference implementation:
- [optimum/neuron/models/inference/llama/modeling_llama.py](llama/modeling_llama.py)

### MoE models (Mixtral, Qwen3 MoE)
- Expert routing and sharding are TP/EP aware.
- Expert capacity and dispatch are statically shaped.
- Expert MLPs use parallel layers or fused kernels.
- State dict remaps for expert sharding when required.

## Porting from [NxDI](https://github.com/aws-neuron/neuronx-distributed-inference)

Use [NxDI](https://github.com/aws-neuron/neuronx-distributed-inference) for neuron-specific graph changes and HF [Transformers](https://github.com/huggingface/transformers) for base architecture.

The Optimum Neuron implementation prioritizes **stability, maintainability, and HF ecosystem compatibility** over cutting-edge performance optimizations. For production deployments requiring maximum throughput, NxDI remains the reference implementation.

### Extra neuron config parameters in NxDI and related code paths

By default, ignore all specific neuron config parameters defined in the NxDI port
that are not present in the `optimum/neuron/models/inference/backend/config.py` file.

### Debugging State Dict Issues

If weight loading fails with "Missing tensor" errors:

1. Check HF checkpoint keys: `list(state_dict.keys())`
2. Verify conversion function handles all keys
3. Add debug logging in `convert_hf_to_neuron_state_dict()`
4. Inspect sharded checkpoint structure after loading

### Per-Module Parity Tests
Track numerical differences using module-level tests before full graph tests:
- [tests/decoder/test_modules.py](../../../tests/decoder/test_modules.py) compares HF layers to Neuron equivalents using `nxd_testing.build_module()` and `validate_accuracy()`.
- [tests/decoder/test_attention.py](../../../../tests/decoder/test_attention.py) validates attention with explicit rotary embedding and mask handling.

These isolate drift or state-dict conversion issues early.
