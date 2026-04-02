# Gemma3 Inference Model Guide

This directory contains the Neuron-optimized Gemma3 inference implementation. It follows [HF Transformers](https://github.com/huggingface/transformers) architecture with Neuron-specific changes derived from [NxDI](https://github.com/aws-neuron/neuronx-distributed-inference).

## HF reference
- Transformers Gemma3 modeling: https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma3

## Gemma3-specific architecture

Gemma3 has several architectural differences from Llama that require special handling:

### Mixed sliding window / full attention
- Each decoder layer has an `attention_type` of either `"sliding_attention"` or `"full_attention"` (from `config.layer_types`).
- Two separate attention masks are computed per forward pass and routed per layer.
- Sliding attention layers use a banded causal mask of width `config.sliding_window`.
- Full attention layers use the standard full causal lower-triangular mask.

### Dual RoPE bases
- Sliding attention layers use `config.rope_local_base_freq` (e.g. 10000).
- Full attention layers use `config.rope_theta` (e.g. 1000000).
- RoPE cos/sin caches are maintained separately per attention type to avoid reuse of incorrect embeddings.

### Q-K normalization
- Query and key projections are normalized after linear projection using `NeuronGemma3RMSNorm`.
- HF names these `q_norm`/`k_norm`; they are renamed to `q_layernorm`/`k_layernorm` in the state dict conversion to match `NeuronAttentionBase` expectations.

### Custom RMSNorm: `NeuronGemma3RMSNorm`
- Applies `(1.0 + weight)` scaling instead of plain `weight` scaling.
- Weights are initialized to zeros (so the effective scale starts at 1.0).
- All layer norms in the model use this class: `input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm`, `q_layernorm`, `k_layernorm`, and the final `norm`.
- Do **not** replace with a standard `RMSNorm` — the weight convention would break numerical parity.

### Four normalization layers per decoder block
- `input_layernorm` → applied before attention
- `post_attention_layernorm` → applied to attention output before residual add
- `pre_feedforward_layernorm` → applied before MLP
- `post_feedforward_layernorm` → applied to MLP output before residual add

This differs from the Llama two-norm pattern (pre-attention + pre-MLP only).

### Scaled word embeddings: `NeuronGemma3TextScaledWordEmbedding`
- Wraps `ParallelEmbedding` and multiplies outputs by `sqrt(hidden_size)`.
- The nested structure requires a state dict remap: `embed_tokens.weight` → `embed_tokens.embedding.weight`.
- Tied weights (`lm_head.weight`) are copied from `embed_tokens.embedding.weight`.

### GELU activation in MLP
- Uses `nn.GELU(approximate="tanh")` instead of SiLU (as in Llama/Qwen3).

### head_dim > 128 bypass
- The NKI flash attention kernel supports `par_dim` up to 128 only.
- If `head_dim` is greater than 128 (for example, Gemma3-270m has `head_dim=256`, while models like Gemma3-27B use `head_dim=128`), flash attention is automatically disabled.
- This check lives in `NeuronAttentionBase.get_flash_attention_strategy()` and applies to all models; no Gemma3-specific override is needed.

## What differs vs HF

- **Parallel layers**: TP-aware linear/embedding layers replace HF modules.
- **Attention base**: `NeuronAttentionBase` for TP-aware heads and static shapes.
- **KV cache/static shapes**: Uses `KVCacheManager` and fixed shapes.
- **Mixed mask routing**: `NxDGemma3Model.forward()` computes two masks and dispatches per layer (not in HF).
- **Dual RoPE cache**: Separate `(cos, sin)` caches per attention type accumulated across layers.
- **Custom RMSNorm**: `NeuronGemma3RMSNorm` replaces HF `Gemma3RMSNorm` using the hardware-optimized `AwsNeuronRmsNorm` op.
- **Scaled embedding wrapper**: `NeuronGemma3TextScaledWordEmbedding` wraps `ParallelEmbedding`.

## State dict conversion

`Gemma3NxDModelForCausalLM.convert_hf_to_neuron_state_dict()` performs:

| From (HF key) | To (Neuron key) |
|---|---|
| `embed_tokens.weight` | `embed_tokens.embedding.weight` |
| `layers.*.self_attn.q_norm.weight` | `layers.*.self_attn.q_layernorm.weight` |
| `layers.*.self_attn.k_norm.weight` | `layers.*.self_attn.k_layernorm.weight` |
| All other norm/MLP/proj keys | unchanged |

## What was removed from [NxDI](https://github.com/aws-neuron/neuronx-distributed-inference)

Reference: [NxDI Gemma3 implementation](https://github.com/aws-neuron/neuronx-distributed-inference/tree/main/src/neuronx_distributed_inference/models/gemma3)

### Removed NKI Kernel Implementations
- **MLP TKG kernel** - Advanced small batch optimization
- **Quantized MLP kernels** - INT8 quantization with fused operations
- **NKI-based MLP** - Hand-tuned kernel implementations

**Why removed**: Requires neuronxcc.nki private APIs. Optimum relies on compiler auto-optimization.

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

**Why removed**: Optimum uses HF Transformers config with `neuron_config.json`.

### What Optimum Neuron Keeps
- Parallel layers with TP-aware sharding
- `NeuronAttentionBase` with GQA and flash attention
- Mixed sliding window / full attention per layer
- Dual RoPE base routing
- Q-K normalization
- State dict conversion (QKV fusion, embedding remap, norm rename)
- Decoder layer composition

Optimum Neuron prioritizes **stability and HF ecosystem compatibility**.

## VLM (image-text-to-text) support

Gemma3 supports vision-language inference via `Gemma3NxDModelForImageTextToText`, registered as `("gemma3", "image-text-to-text")`.

### Architecture

Two compiled bundles:
- **`model_vision.pt`**: `NeuronGemma3VisionEncoder` — SigLIP vision transformer + multi-modal projector
- **`model_text.pt`**: `NxDGemma3VLMDecoderModel` — Gemma3 text decoder with image embedding injection

Class hierarchy:
- `Gemma3NxDModelForImageTextToText` → `NxDModelForImageTextToText` (base VLM class)
- `NxDGemma3VLMDecoderModel` → `NxDGemma3Model` (inherits mixed sliding/full attention)
- `NeuronGemma3VisionEncoder` → contains `NeuronGemma3SigLIPVisionTransformer` + `NeuronGemma3MultiModalProjector`

### TP-sharded vision encoder

The SigLIP vision encoder (27 layers at 896×896) exceeds Neuron compiler instruction limits when `batch_size * max_num_images > 2` without tensor parallelism. The vision encoder uses `ColumnParallelLinear`/`RowParallelLinear` in attention Q/K/V/out and MLP fc1/fc2 to distribute across TP ranks.

Key classes:
- `NeuronGemma3SigLIPAttention` — TP-sharded multi-head attention
- `NeuronGemma3SigLIPMLP` — TP-sharded MLP
- `NeuronGemma3SigLIPEncoderLayer` / `NeuronGemma3SigLIPEncoder` — layer stack

### Multi-modal projector

`NeuronGemma3MultiModalProjector` downsamples vision features using `AvgPool2d` then projects via RMSNorm + linear from vision hidden dim to text hidden dim. Attribute names (`mm_input_projection_weight`, `mm_soft_emb_norm`) match HF for direct state dict loading.

### Position embeddings

Gemma3 uses standard sequential position IDs `[0, 1, ..., num_patches - 1]` (via `NeuronGemma3SigLIPVisionEmbeddings`), NOT Idefics3-style fractional-coordinate bucketing used by SmolVLM.

### Image injection

During context encoding, `NxDGemma3VLMDecoderModel.forward()` computes text embeddings, then replaces embeddings at `image_token_id` positions with vision features using `torch.where`.

### VLM state dict conversion

`Gemma3NxDModelForImageTextToText`:
- `_STATE_DICT_MODEL_PREFIX = "language_model.model."` — strips HF nesting for text weights
- `_get_vision_encoder_state_dict()`: remaps `vision_tower.vision_model.*` → `vision_model.*`, keeps `multi_modal_projector.*` as-is
- `convert_hf_to_neuron_state_dict()`: preserves `language_model.lm_head.weight` → `lm_head.weight`, removes remaining vision/projector/language_model keys, delegates to CausalLM converter

### Constraints

- **Chunked prefill not supported** (`_supports_chunked_prefill = False`): the mixed sliding/full attention masks are not compatible with chunked prefill. `prefill_chunk_size` is forced to 0.
- **No image tiling**: unlike SmolVLM/Idefics3, Gemma3 processes each image at full resolution. Each image produces exactly `mm_tokens_per_image` features.
- **`max_num_images`**: set to 5 in `_get_neuron_config()`. This determines the vision batch size (`batch_size * max_num_images`).

## Key files
- [optimum/neuron/models/inference/gemma3/modeling_gemma3.py](modeling_gemma3.py)
- [optimum/neuron/models/inference/backend/modules/attention/attention_base.py](../backend/modules/attention/attention_base.py)
