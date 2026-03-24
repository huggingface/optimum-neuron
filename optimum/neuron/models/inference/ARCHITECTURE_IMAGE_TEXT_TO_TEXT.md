# NxD Vision-Language Model Architecture

This document describes the class hierarchy, compilation flow, and runtime
dispatch for image-text-to-text inference models. It builds on the text-only
architecture described in [ARCHITECTURE_CAUSAL_LM.md](ARCHITECTURE_CAUSAL_LM.md)
and focuses on what changes and what stays the same.

---

## 1. How Image-Text-to-Text Differs from Text-Generation

Text generation (CausalLM) takes a token sequence and autoregressively produces more
tokens. Image-text-to-text (VLM) does the same, but first encodes one or more images
into embedding vectors and **injects** them into the token sequence before the
transformer layers run.

```
TEXT-ONLY (CausalLM):

  input_ids ──► embed_tokens() ──► transformer layers ──► lm_head ──► logits


VLM (ImageTextToText):

  pixel_values ──► vision_encoder() ──► image_embeds  ─┐
                                                       │  inject at
  input_ids ──► embed_tokens() ──► hidden_states ◄─────┘  <image> positions
                                        │
                                        ▼
                                transformer layers ──► lm_head ──► logits
```

Key differences:
- **Prefill only.** Image features are injected during context encoding / chunked
  prefill. Token generation is identical to text-only — the KV cache already
  contains the vision-injected representations.
- **Two weight sets.** The vision encoder has its own weights (separate from the
  text decoder). They live in a separate `model.pt`.
- **Extra inputs for context encoding.** The context encoding graph receives
  `image_embeds` and `image_token_mask` in addition to the standard 4 tensors.

---

## 2. Two Compiled Bundles

VLM models produce two `model.pt` files. The vision encoder has completely
different weights from the text decoder, so it must be a separate bundle.
The text decoder graphs share weights and stay in a single bundle.

`NxDPreTrainedModel` supports multiple bundles natively.
`create_graph_builders()` returns `dict[str, dict[str, NxDGraphBuilder]]` —
a dict of bundles, each bundle being a dict of graph builders. `compile()`,
`save()`, and `load_weights()` iterate over bundles automatically. For
single-bundle models (all existing CausalLM and Embedding models), the outer
dict has one key `"model"` and file naming is backward-compatible (`model.pt`).
Multi-bundle models use `model_{bundle_name}.pt`.

All decoder graphs within the text bundle share a **uniform 6-tensor signature**
(`input_ids`, `position_ids`, `seq_ids`, `sampling_params`, `image_embeds`,
`image_token_mask`). During context encoding / chunked prefill, image tensors
carry real data. During token generation, dummy tensors (all-false mask, empty
embeds) are passed so that the compiled graph signature is uniform and the
image injection is a no-op.

```
┌───────────────────────────────────────────────────────┐
│  model_vision.pt — Vision Encoder                     │
│  (separate weights, separate ModelBuilder.trace())    │
│                                                       │
│  ┌─────────────────────────────────────────────────┐  │
│  │ vision_encoder                                  │  │
│  │                                                 │  │
│  │ input:  pixel_values [B×N_img, 3, H, W]         │  │
│  │ output: image_features [B×N_img, seq_len, dim]  │  │
│  └─────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────┐
│  model_text.pt — Text Decoder (shared weights)        │
│  All graphs use uniform 6-tensor signature.           │
│                                                       │
│  ┌─────────────────────────┐  ┌─────────────────────┐ │
│  │ context_encoding        │  │ token_generation    │ │
│  │                         │  │                     │ │
│  │ 6 input tensors:        │  │ 6 input tensors:    │ │
│  │   input_ids             │  │   input_ids         │ │
│  │   position_ids          │  │   position_ids      │ │
│  │   seq_ids               │  │   seq_ids           │ │
│  │   sampling_params       │  │   sampling_params   │ │
│  │   image_embeds          │  │   image_embeds      │ │
│  │   image_token_mask      │  │   image_token_mask  │ │
│  │                         │  │   (dummy: all-false │ │
│  │                         │  │    mask, injection  │ │
│  │                         │  │    is a no-op)      │ │
│  └─────────────────────────┘  └─────────────────────┘ │
│                                                       │
│  ┌─────────────────────────┐                          │
│  │ chunked_prefill         │                          │
│  │ (replaces ctx_enc when  │                          │
│  │  prefill_chunk_size > 0)│                          │
│  │                         │                          │
│  │ 6 input tensors         │                          │
│  │ (same as ctx_enc above) │                          │
│  └─────────────────────────┘                          │
└───────────────────────────────────────────────────────┘

ModelBuilder requires a uniform forward() signature across all graphs
in the same bundle (the NxDModel ScriptModule has a single forward()
declaration). All graphs receive 6 tensors. Token generation passes
dummy (zero) image_embeds and image_token_mask — they are ignored at
runtime because the mask is all-False.

Note: speculation is not yet supported for VLM models.
VLM's create_graph_builders() does not include a speculation graph.
```

---

## 3. Class Hierarchy

### 3a. Public API layer

```
                          ┌──────────────────────────────┐
                          │  NeuronPreTrainedModel       │  (modeling_utils.py)
                          └─────────┬────────────────────┘
                                    │
              ┌─────────────────────┼────────────────────────┐
              │                                              │
    ┌─────────▼────────────────────┐     ┌───────────────────▼──────────────────┐
    │  NeuronModelForCausalLM      │     │  NeuronModelForImageTextToText       │
    │                              │     │                                      │
    │  task = "text-generation"    │     │  task = "image-text-to-text"         │
    └──────────────────────────────┘     └──────────────────────────────────────┘
```

### 3b. Orchestrator layer

```
    ┌─────────────────────────────────────────┐
    │  NxDModelForCausalLM                    │  (modeling_decoder.py)
    │                                         │
    │  _model_cls = None                      │
    │  _text_bundle_key = "model"             │
    │  _context_wrapper_cls =                 │
    │      NxDDecoderWrapperForCausalLM       │
    │  _chunked_prefill_wrapper_cls =         │
    │      NxDDecoderWrapperForCausalLM       │
    │  _token_generation_wrapper_cls =        │
    │      NxDDecoderWrapperForCausalLM       │
    │  _speculation_wrapper_cls =             │
    │      NxDDecoderWrapperForCausalLM       │
    │                                         │
    │  .context_encoding_model (4 tensors)    │
    │  .token_generation_model                │
    │  .speculation_model                     │
    │                                         │
    │  forward()                              │
    │  create_graph_builders()                │
    └────────────────┬────────────────────────┘
                     │
    ┌────────────────▼────────────────────────┐
    │  NxDModelForImageTextToText             │  (vlm_decoder.py)
    │                                         │
    │  _model_cls = None                      │
    │  _vision_encoder_cls = None             │
    │  _text_bundle_key = "text"              │
    │  _context_wrapper_cls =                 │
    │      NxDDecoderWrapperForImageTextToText│
    │  _chunked_prefill_wrapper_cls =         │
    │      NxDDecoderWrapperForImageTextToText│
    │  _token_generation_wrapper_cls =        │
    │      NxDTokenGenerationWrapperFor...    │
    │                                         │
    │  .context_encoding_model (6 tensors)    │
    │  .token_generation_model (6 tensors,    │
    │       dummy image args)                 │
    │  .speculation_model      (inherited,    │
    │       not compiled for VLM)             │
    │                                         │
    │  forward()               (overridden)   │
    │  generate()              (overridden)   │
    │  prefill_chunk()         (overridden)   │
    │  create_graph_builders() (overridden)   │
    │  create_vision_graph_builders()         │
    │  _prepare_image_injection_tensors()     │
    │  _encode_images()                       │
    │  get_checkpoint_loader_fn()             │
    └────────────────┬────────────────────────┘
                     │
    ┌────────────────▼────────────────────────┐
    │  SmolVLMNxDModelForImageTextToText      │  (smolvlm/modeling_smolvlm.py)
    │                                         │
    │  _model_cls = NxDSmolVLMDecoderModel    │
    │  _vision_encoder_cls =                  │
    │      NeuronIdefics3VisionEncoder        │
    │  _STATE_DICT_MODEL_PREFIX =             │
    │      "model.text_model."                │
    └─────────────────────────────────────────┘
```

### 3c. Traced module layer (nn.Module)

There is no shared `NxDDecoderModelForImageTextToText` base class. Each VLM
text decoder directly subclasses the appropriate architecture-specific model
(e.g. `NxDLlamaModel`) and overrides `forward()` to accept and inject image
embeddings.

The pattern: the VLM decoder overrides `forward()` to accept `image_embeds` +
`image_token_mask` as extra arguments, calls `compute_input_embeddings()` to
get text embeddings, injects image features using `torch.where()`, then
delegates to `_forward_from_embeddings()` for the rest of the transformer.

```
    ┌──────────────────────────────────────────────────────┐
    │  NxDDecoderModelForCausalLM                          │  (modeling_decoder.py)
    │                                                      │
    │  forward(input_ids, position_ids,                    │
    │          seq_ids, sampling_params)                   │
    │                                                      │
    │  compute_input_embeddings(input_ids):                │  ◄── hook
    │      return self.embed_tokens(input_ids)             │
    │                                                      │
    │  _forward_from_embeddings(hidden_states,             │
    │      position_ids, seq_ids, sampling_params)         │
    └──────────────────┬───────────────────────────────────┘
                       │
    ┌──────────────────▼───────────────────────────────────┐
    │  NxDLlamaModel (or other arch-specific model)        │
    │  (fills in embed_tokens, layers, norm, lm_head)      │
    └──────────────────┬───────────────────────────────────┘
                       │
    ┌──────────────────▼───────────────────────────────────┐
    │  NxDSmolVLMDecoderModel                              │  (smolvlm/modeling_smolvlm.py)
    │                                                      │
    │  forward(input_ids, position_ids,                    │
    │          seq_ids, sampling_params,                   │
    │          image_embeds=None, image_token_mask=None):  │
    │      h = self.compute_input_embeddings(input_ids)    │
    │      if is_context_encoding and image args present:  │
    │          mask = image_token_mask.bool().unsqueeze(-1)│
    │          h = torch.where(mask, image_embeds, h)      │
    │      return self._forward_from_embeddings(           │
    │               h, position_ids,                       │
    │               seq_ids, sampling_params)              │
    └──────────────────────────────────────────────────────┘
```

**What this buys:**
- All decoder graphs (context encoding, token generation) share the same traced
  module with a uniform 6-tensor signature. During token generation, dummy
  tensors (all-false mask) are passed, making the injection a no-op.
- The base class provides `compute_input_embeddings()` and
  `_forward_from_embeddings()` as clean extension points. VLM decoders only
  need to override `forward()` to insert the injection logic between them.

### 3d. Vision encoder (nn.Module, separate bundle)

The vision encoder is model-specific and has no relation to the decoder class
hierarchy. It is a plain `nn.Module` traced independently.

```
    ┌────────────────────────────────────────────────────────────┐
    │  NeuronIdefics3VisionEncoder (nn.Module)                   │
    │                                                            │
    │  .vision_model: NeuronSigLIPVisionTransformer              │
    │      .embeddings: NeuronSigLIPVisionEmbeddings             │
    │          .patch_embedding: Conv2d(3 → hidden, patch_size)  │
    │          .position_embedding: Embedding(num_patches, dim)  │
    │      .encoder: NeuronSigLIPEncoder                         │
    │          .layers: [NeuronSigLIPEncoderLayer × N]           │
    │              .self_attn: NeuronSigLIPAttention             │
    │              .mlp: NeuronSigLIPMLP                         │
    │      .post_layernorm: LayerNorm                            │
    │  .connector: Idefics3Connector (maps vision → text dim)    │
    │                                                            │
    │  forward(pixel_values) → image_features [B, seq_len, dim]  │
    └────────────────────────────────────────────────────────────┘
```

---

## 4. Base Class Design

### 4a. Multi-bundle support in `NxDPreTrainedModel`

`NxDPreTrainedModel` stores `_traced_models: dict[str, ScriptModule]`
and `graph_builders: dict[str, dict[str, NxDGraphBuilder]]`. The outer dict
keys are bundle names (`"model"` for single-bundle, `"vision"` / `"text"` for VLM).

| Method                    | Behavior                                             |
|---------------------------|------------------------------------------------------|
| `__init__`                | Takes `traced_models` dict                           |
| `compile()`               | Iterates bundles, one `ModelBuilder.trace()` each    |
| `save()`                  | Saves `model.pt` or `model_{name}.pt` per bundle     |
| `_from_pretrained()`      | Loads all bundles by name                            |
| `_load_weights_from_path` | Initializes each bundle separately                   |
| `shard_checkpoint()`      | Shards per bundle, per-bundle subdirs for multi      |
| `get_checkpoint_loader_fn(bundle_name)` | Overridable per-bundle loader          |

Single-bundle: shards stored in `weights_path/weights/`.
Multi-bundle: shards stored in `weights_path/{bundle_name}/weights/`.

`NxDModelForCausalLM` and `NxDModelForEmbedding` wrap their builders in
`{"model": {...}}` and extract `traced_models["model"]` in `__init__`.
Existing model subclasses are unaffected.

### 4b. Embedding hook in `NxDDecoderModelForCausalLM`

`NxDDecoderModelForCausalLM.forward()` calls `compute_input_embeddings()`
then delegates to `_forward_from_embeddings()`. This separation allows VLM
decoders to insert image injection between the two steps:

```python
def compute_input_embeddings(self, input_ids):
    return self.embed_tokens(input_ids)

def forward(self, input_ids, position_ids, seq_ids, sampling_params):
    hidden_states = self.compute_input_embeddings(input_ids)
    return self._forward_from_embeddings(hidden_states, position_ids, seq_ids, sampling_params)
```

### What stays unchanged from CausalLM

| Component                           | Changed for VLM?                    |
|-------------------------------------|-------------------------------------|
| `NxDDecoderModelForCausalLM`        | No (VLM subclasses override forward)|
| `NxDDecoderWrapperForCausalLM`      | No                                  |
| `NxDDecoderBuilderForCausalLM`      | No                                  |
| `NxDGenerationMixin._sample()`      | No                                  |
| Speculation graph                   | No (not compiled for VLM)           |
| All existing CausalLM subclasses    | No                                  |

---

## 5. Builder Layer

```
NxDGraphBuilder (graph_builder.py)  ← ABC
       │
       ├──► NxDDecoderBuilderForCausalLM                  (existing, unchanged)
       │       input_generator() → 4 tensors
       │
       ├──► NxDDecoderBuilderForImageTextToText            (vlm_builders.py)
       │       extends NxDDecoderBuilderForCausalLM
       │       input_generator() → 6 tensors:
       │         (input_ids, position_ids, seq_ids, sampling_params,
       │          image_embeds [B, active_tokens, hidden_size],
       │          image_token_mask [B, active_tokens])
       │
       ├──► NxDTokenGenerationBuilderForImageTextToText    (vlm_builders.py)
       │       extends NxDDecoderBuilderForCausalLM
       │       input_generator() → 6 tensors (dummy image tensors
       │         with all-false mask for uniform signature,
       │         active_tokens=1)
       │
       ├──► NxDChunkedPrefillBuilderForImageTextToText     (vlm_builders.py)
       │       extends NxDDecoderBuilderForImageTextToText
       │       active_tokens = prefill_chunk_size
       │
       └──► NxDVisionEncoderBuilder                        (vlm_builders.py)
               input_generator() → 1 tensor:
                 pixel_values [B × max_num_images, 3, image_size, image_size]
```

`NxDModelForImageTextToText.create_graph_builders()` returns:

```python
{
    "vision": {
        "vision_encoder": NxDVisionEncoderBuilder(...)
    },
    "text": {
        "context_encoding": NxDDecoderBuilderForImageTextToText(...),               # 6 tensors
        # OR (mutually exclusive):
        "chunked_prefill": NxDChunkedPrefillBuilderForImageTextToText(...),         # 6 tensors
        "token_generation": NxDTokenGenerationBuilderForImageTextToText(...),       # 6 tensors (dummy)
    }
}
```

`context_encoding` and `chunked_prefill` are mutually exclusive — the VLM
orchestrator creates one or the other based on `neuron_config.prefill_chunk_size`.

Each outer key becomes a separate `ModelBuilder.trace()` call and a separate
`model_{name}.pt` file. Graphs within the same bundle share weights.
`compile()`, `save()`, and `load_weights()` iterate over bundles automatically
via `NxDPreTrainedModel`.

The VLM class overrides `get_checkpoint_loader_fn(bundle_name)` to return
different weight extraction logic for `"vision"` vs `"text"` bundles.

---

## 6. Wrapper Layer

```
NxDModelWrapper (model_wrapper.py)
       │
       ├──► NxDDecoderWrapperForCausalLM                       (existing, unchanged)
       │
       ├──► NxDDecoderWrapperForImageTextToText                 (vlm_wrappers.py)
       │       Extends NxDDecoderWrapperForCausalLM.
       │       Used for context_encoding / chunked_prefill.
       │       Overrides forward() and _forward() to pass image_embeds +
       │       image_token_mask to the traced model. Handles padding of
       │       image tensors alongside input_ids for both context encoding
       │       and chunked prefill paths.
       │
       └──► NxDTokenGenerationWrapperForImageTextToText         (vlm_wrappers.py)
               Extends NxDDecoderWrapperForCausalLM.
               Used for token_generation.
               Inherits the base forward() for 4-tensor batching/padding.
               Overrides _forward() to create and cache dummy image_embeds
               (zeros, shape [B, 1, hidden_size]) and image_token_mask
               (all-false, shape [B, 1]). Passes them alongside the
               standard 4 input tensors to maintain the uniform 6-tensor
               compiled signature. The all-false mask ensures the injection
               in the traced model is a no-op.
```

---

## 7. Forward Dispatch During Generation

### 7a. Orchestrator dispatch (`NxDModelForImageTextToText.forward`)

The orchestrator's `forward()` handles the vision/text split. The
`generate()` method captures `pixel_values` in `_current_pixel_values`
before delegating to the base `NxDGenerationMixin.generate()`, which
calls `_sample()`, which calls `forward()` or `prefill_chunk()`. This
avoids any changes to `_sample()` or `NxDGenerationMixin`.

```
forward(input_ids, position_ids, seq_ids, sampling_params, pixel_values=None)
   │
   │  pixel_values ← self._current_pixel_values if not passed directly
   │
   │  ┌─ Once per prompt (prefill, chunk_size == 0) ────────────┐
   │  │                                                          │
   │  │  1. Run vision encoder (model_vision.pt):                │
   │  │     image_features = _encode_images(pixel_values)        │
   │  │                                                          │
   │  │  2. Prepare injection tensors:                           │
   │  │     image_embeds, image_token_mask, _ =                  │
   │  │         _prepare_image_injection_tensors(                │
   │  │             input_ids, image_features)                   │
   │  │                                                          │
   │  │  3. Run context encoding (model_text.pt, 6 tensors):     │
   │  │     context_encoding_model(                              │
   │  │         input_ids, position_ids, seq_ids,                │
   │  │         sampling_params, image_embeds, image_token_mask) │
   │  │     set kv_cache_populated = True                        │
   │  │                                                          │
   │  └──────────────────────────────────────────────────────────┘
   │
   │  ┌─ Once per prompt (prefill, chunk_size > 0) ──────────────┐
   │  │                                                          │
   │  │  1. Run vision encoder (once):                           │
   │  │     image_features = _encode_images(pixel_values)        │
   │  │                                                          │
   │  │  2. Iterate chunks with feature_offset tracking:         │
   │  │     for chunk_start in range(0, prompt_len, chunk_size): │
   │  │       chunk_embeds, chunk_mask, feature_offset =         │
   │  │           _prepare_image_injection_tensors(              │
   │  │               chunk_ids, image_features, feature_offset) │
   │  │       chunked_prefill_model(chunk_ids, chunk_pos,        │
   │  │           seq_ids, sampling_params,                      │
   │  │           chunk_embeds, chunk_mask)  → 6 tensors         │
   │  │     set kv_cache_populated = True                        │
   │  │                                                          │
   │  └──────────────────────────────────────────────────────────┘
   │
   │  ┌─ Decode loop ────────────────────────────────────────────┐
   │  │                                                          │
   │  │  token_generation_model(                                 │
   │  │      input_ids, position_ids, seq_ids, sampling_params)  │
   │  │  (wrapper adds dummy image tensors internally)           │
   │  │                                                          │
   │  │  KV cache already has vision-injected representations.   │
   │  │                                                          │
   │  └──────────────────────────────────────────────────────────┘
```

Note: `forward()` contains both the full-context and chunked-prefill paths.
When `_sample()` drives generation with `prefill_chunk_size > 0`, it calls
`prefill_chunk()` per chunk instead of `forward()` (see §7d). The chunk loop
inside `forward()` exists for any caller that passes a full prompt directly.
The vLLM integration does **not** use this path — see §7d Path B.

### 7b. Inside the traced graph (context encoding path)

```
NxDSmolVLMDecoderModel.forward(                        ← VLM decoder
    input_ids, position_ids, seq_ids, sampling_params,
    image_embeds, image_token_mask)
   │
   │  h = self.compute_input_embeddings(input_ids)
   │      │
   │      │  return self.embed_tokens(input_ids)
   │
   │  if is_context_encoding and image args present:
   │      mask = image_token_mask.bool().unsqueeze(-1)
   │      h = torch.where(mask, image_embeds, h)       ← injection
   │
   │  return self._forward_from_embeddings(
   │      h, position_ids, seq_ids, sampling_params)
   │      │
   │      │  ... layer loop, norm, lm_head (unchanged) ...
```

### 7c. Inside the traced graph (token generation path)

Token generation uses the same traced module and 6-tensor signature,
but the wrapper passes dummy tensors (all-false mask), so
`_is_context_encoding()` returns `False` and the injection is skipped:

```
NxDSmolVLMDecoderModel.forward(                        ← same VLM decoder
    input_ids, position_ids, seq_ids, sampling_params,
    dummy_image_embeds, dummy_image_token_mask)
   │
   │  h = self.compute_input_embeddings(input_ids)
   │
   │  is_context_encoding = False  (single token)
   │  → injection skipped
   │
   │  return self._forward_from_embeddings(
   │      h, position_ids, seq_ids, sampling_params)
```

### 7d. Chunked prefill with VLM

There are two chunked prefill paths depending on the caller:

**Path A: via `_sample()` → `prefill_chunk()` (generation loop)**

When `_sample()` drives generation with `prefill_chunk_size > 0`, it iterates
chunks itself and calls `self.prefill_chunk()` per chunk.
`NxDModelForImageTextToText` overrides `prefill_chunk()` to inject image features.
The vision encoder runs on the first call and is cached for subsequent chunks.
A running `_image_feature_offset` tracks consumed image feature rows.

```
_sample():
   │
   │  for chunk_start in range(0, prompt_len, chunk_size):
   │     self.prefill_chunk(chunk_ids, chunk_pos, seq_ids, sampling_params)
   │
   └→ NxDModelForImageTextToText.prefill_chunk():
         │
         │  if _cached_image_features is None:       ← first chunk
         │      image_features = _encode_images(pixel_values)
         │      _cached_image_features = image_features
         │      _image_feature_offset = 0
         │
         │  image_embeds, image_token_mask, _image_feature_offset =
         │      _prepare_image_injection_tensors(
         │          input_ids, _cached_image_features, _image_feature_offset)
         │
         │  chunked_prefill_model(
         │      input_ids, position_ids, seq_ids,
         │      sampling_params, image_embeds, image_token_mask)
         │  → 6 tensors
         │
         │  set kv_cache_populated = True
```

**Path B: via vLLM runner → `prefill_chunk_vllm()` → `prefill_chunk()`**

The vLLM integration drives chunking from its own runner
(`optimum/neuron/vllm/runner.py: _execute_chunked_prefill`). The runner
iterates chunks itself and calls `prefill_chunk_vllm()` per chunk — a
thin wrapper in `optimum/neuron/vllm/model_loader.py` that delegates to
the model's `prefill_chunk()` method and reshapes the output:

```
vLLM runner._execute_chunked_prefill():
   │
   │  for each sequence:
   │    for chunk_start in range(0, prompt_len, chunk_size):
   │       model.prefill_chunk_vllm(chunk_ids, chunk_pos, seq_ids, samp)
   │         │
   │         └→ model.prefill_chunk(chunk_ids, chunk_pos, seq_ids, samp)
   │              → [1, 1, vocab]  squeezed to [1, vocab]
```

This means the vLLM path uses the same underlying `prefill_chunk()`
override as Path A, with its vision encoder caching and
`_image_feature_offset` tracking. The `forward()` internal chunk loop
(§7a) is **not** used by vLLM.

In all chunked prefill paths, the chunked prefill graph always receives
6 tensors (same traced signature as context encoding). When a chunk has
no image tokens, the mask is all-False and the injection in the traced
model is a no-op.

### 7e. Generation loop and `NxDGenerationMixin`

`NxDModelForImageTextToText.generate()` captures `pixel_values` in
`self._current_pixel_values` before calling `super().generate()`.

The base `NxDGenerationMixin._sample()` is unchanged — it calls:
- `self.prefill_chunk()` per chunk when `prefill_chunk_size > 0`
  (VLM overrides `prefill_chunk()` to inject image features)
- `self.forward()` for full context encoding when `prefill_chunk_size == 0`
- `self.forward()` per token in the decode loop

The orchestrator's `forward()` retrieves `_current_pixel_values` and
handles the vision/text split:
- On prefill: runs the vision encoder, prepares injection tensors, calls
  context encoding with 6 real tensors.
- On decode: calls token generation with 4 tensors (the wrapper adds
  dummy image tensors).

After `generate()` completes, cached state is cleaned up in a `finally`
block: `_current_pixel_values`, `_cached_image_features`, and
`_image_feature_offset` are all reset.

---

## 8. End-to-End: From HF Model to Token Output

```
User code:
  model = NeuronModelForImageTextToText.export(
      "HuggingFaceTB/SmolVLM-256M-Instruct", neuron_config)
  model.generate(input_ids, pixel_values=pixel_values, ...)

 ┌──────────────────────────────────────────────────────────────────────┐
 │  EXPORT (compile time)                                               │
 │                                                                      │
 │  NeuronModelForImageTextToText.export()                              │
 │    → resolves to SmolVLMNxDModelForImageTextToText                   │
 │      (registered via @register_neuron_model_for_inference            │
 │       for "idefics3" / "image-text-to-text")                         │
 │    → _export()                                                       │
 │                                                                      │
 │    1. create_graph_builders() returns:                               │
 │       {"vision": {"vision_encoder": ...},                            │
 │        "text":   {"context_encoding": ...,                           │
 │                   "token_generation": ...}}                          │
 │                                                                      │
 │    2. NxDPreTrainedModel.compile() iterates bundles:                 │
 │       "vision" → ModelBuilder.trace() → model_vision.pt              │
 │       "text"   → ModelBuilder.trace() → model_text.pt                │
 │                                                                      │
 │    3. SmolVLMNxDModelForImageTextToText.__init__()                   │
 │       → context_encoding_model  (VLM wrapper, traced_models["text"]) │
 │       → token_generation_model  (VLM tkgen wrapper,                  │
 │                                  traced_models["text"])              │
 │                                                                      │
 │    4. load_weights() iterates bundles:                               │
 │       "vision" → get_checkpoint_loader_fn("vision") → vision weights │
 │       "text"   → get_checkpoint_loader_fn("text")   → text weights   │
 │                                                                      │
 └──────────────────────────────────────────────────────────────────────┘

 ┌────────────────────────────────────────────────────────────────────┐
 │  GENERATE (runtime)                                                │
 │                                                                    │
 │  model.generate(input_ids, pixel_values=...)                       │
 │    → NxDModelForImageTextToText.generate()                         │
 │        captures pixel_values in _current_pixel_values              │
 │        calls super().generate() → NxDGenerationMixin.generate()    │
 │          calls reset() → kv_cache_populated = False                │
 │          calls super().generate() (HF GenerationMixin)             │
 │            → calls _sample()                                       │
 │                                                                    │
 │  _sample() drives prefill + decode:                                │
 │                                                                    │
 │    ┌─ Prefill (chunk_size == 0) ───────────────────────────┐       │
 │    │ self.forward(input_ids, pos_ids, seq_ids, samp)       │       │
 │    │   → NxDModelForImageTextToText.forward():             │       │
 │    │     pixel_values ← self._current_pixel_values         │       │
 │    │     1. _encode_images(pixel_values) → image_features  │       │
 │    │        runs vision traced model (model_vision.pt)     │       │
 │    │     2. _prepare_image_injection_tensors(              │       │
 │    │            input_ids, image_features)                 │       │
 │    │        → image_embeds, image_token_mask               │       │
 │    │     3. context_encoding_model(input_ids, pos_ids,     │       │
 │    │            seq_ids, samp, image_embeds, mask)         │       │
 │    │        → 6 tensors, model_text.pt                     │       │
 │    └───────────────────────────────────────────────────────┘       │
 │                                                                    │
 │    ┌─ Prefill (chunk_size > 0) ────────────────────────────┐       │
 │    │ _sample() iterates chunks and calls per chunk:        │       │
 │    │   self.prefill_chunk(chunk_ids, chunk_pos,            │       │
 │    │                      seq_ids, samp)                   │       │
 │    │   → NxDModelForImageTextToText.prefill_chunk():       │       │
 │    │     1. _encode_images() on first chunk (cached)       │       │
 │    │     2. _prepare_image_injection_tensors(              │       │
 │    │            chunk_ids, image_features, feature_offset) │       │
 │    │        → chunk_embeds, chunk_mask, new_offset         │       │
 │    │     3. chunked_prefill_model(chunk_ids, chunk_pos,    │       │
 │    │            seq_ids, samp, chunk_embeds, chunk_mask)   │       │
 │    │        → 6 tensors, model_text.pt                     │       │
 │    └───────────────────────────────────────────────────────┘       │
 │                                                                    │
 │    ┌─ Decode loop ─────────────────────────────────────────┐       │
 │    │ self.forward(one_token, pos_ids, ...)                 │       │
 │    │   → NxDModelForImageTextToText.forward():             │       │
 │    │     dispatches to token_generation_model(             │       │
 │    │         input_ids, pos_ids, seq_ids, samp)            │       │
 │    │     wrapper adds dummy image tensors → 6 tensors      │       │
 │    │   No vision encoder call.                             │       │
 │    └───────────────────────────────────────────────────────┘       │
 │                                                                    │
 │  After generate() returns (finally block):                         │
 │    _current_pixel_values = None                                    │
 │    _cached_image_features = None                                   │
 │    _image_feature_offset = 0                                       │
 │                                                                    │
 └────────────────────────────────────────────────────────────────────┘
```

---

## 9. Config Extension

`NxDVLMNeuronConfig` extends `NxDNeuronConfig` with vision-specific fields:

| Field             | Type  | Default | Description                                   |
|-------------------|-------|---------|-----------------------------------------------|
| `max_num_images`  | int   | 1       | Max images per input (affects vision batch)   |
| `image_size`      | int   | 512     | Input image resolution (static for tracing)   |
| `image_seq_len`   | int   | 64      | Tokens produced per image by vision encoder   |

Text decoder config specialization per graph is identical to CausalLM
(see [ARCHITECTURE_CAUSAL_LM.md §9](ARCHITECTURE_CAUSAL_LM.md#9-config-specialization-per-graph)).

---

## 10. File Reference

```
optimum/neuron/models/inference/
├── modeling_utils.py                             # NeuronModelForImageTextToText
├── ARCHITECTURE_CAUSAL_LM.md                     # text-only architecture doc
├── ARCHITECTURE_IMAGE_TEXT_TO_TEXT.md             # this document
├── backend/
│   ├── config.py                                 # NxDVLMNeuronConfig
│   ├── pretrained_model.py                       # multi-bundle support
│   └── modules/
│       └── decoder/
│           ├── modeling_decoder.py               # NxDDecoderModelForCausalLM (base)
│           │                                     # NxDModelForCausalLM (orchestrator)
│           ├── vlm_decoder.py                    # NxDModelForImageTextToText (orchestrator)
│           ├── vlm_builders.py                   # NxDDecoderBuilderForImageTextToText
│           │                                     # NxDTokenGenerationBuilderForImageTextToText
│           │                                     # NxDChunkedPrefillBuilderForImageTextToText
│           │                                     # NxDVisionEncoderBuilder
│           ├── vlm_wrappers.py                   # NxDDecoderWrapperForImageTextToText
│           │                                     # NxDTokenGenerationWrapperForImageTextToText
│           ├── decoder_wrappers.py               # NxDDecoderWrapperForCausalLM (base)
│           └── decoder_builders.py               # NxDDecoderBuilderForCausalLM (base)
├── smolvlm/
│   └── modeling_smolvlm.py                       # SmolVLMNxDModelForImageTextToText
│                                                 # NxDSmolVLMDecoderModel
│                                                 # NeuronIdefics3VisionEncoder
│                                                 # NeuronSigLIP* layers
│                                                 # SmolVLMNeuronModelForImageTextToText
│                                                 #   (registered model alias)
└── ...
```
