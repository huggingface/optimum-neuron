# NxD Vision-Language Model Architecture

This document describes the **proposed** class hierarchy, compilation flow, and runtime
dispatch for image-text-to-text inference models. It builds on the text-only architecture
described in [ARCHITECTURE_CAUSAL_LM.md](ARCHITECTURE_CAUSAL_LM.md) and focuses on what
changes and what stays the same.

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

  pixel_values ──► vision_encoder() ──► image_embeds ─┐
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
  The token generation graph signature is unchanged.

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

```
┌──────────────────────────────────────────────────────┐
│  model_vision.pt — Vision Encoder                     │
│  (separate weights, separate ModelBuilder.trace())    │
│                                                       │
│  ┌─────────────────────────────────────────────────┐  │
│  │ vision_encoding                                  │  │
│  │                                                  │  │
│  │ input:  pixel_values [B×N_img, 3, H, W]         │  │
│  │ output: image_features [B×N_img, seq_len, dim]  │  │
│  └─────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  model_text.pt — Text Decoder (shared weights)        │
│                                                       │
│  ┌─────────────────────────┐  ┌─────────────────────┐ │
│  │ context_encoding        │  │ token_generation    │ │
│  │                         │  │                     │ │
│  │ 6 input tensors:        │  │ 4 input tensors:    │ │
│  │   input_ids             │  │   input_ids         │ │
│  │   position_ids          │  │   position_ids      │ │
│  │   seq_ids               │  │   seq_ids           │ │
│  │   sampling_params       │  │   sampling_params   │ │
│  │   image_embeds ◄── NEW  │  │                     │ │
│  │   image_token_mask ◄──  │  │ (UNCHANGED from     │ │
│  │                         │  │  text-only CausalLM) │ │
│  └─────────────────────────┘  └─────────────────────┘ │
│                                                       │
│  ┌─────────────────────────┐  ┌─────────────────────┐ │
│  │ chunked_prefill         │  │ speculation         │ │
│  │ (replaces ctx_enc)      │  │ (optional)          │ │
│  │                         │  │                     │ │
│  │ 6 input tensors         │  │ 4 input tensors     │ │
│  │ (same as ctx_enc above) │  │ (same as token_gen) │ │
│  └─────────────────────────┘  └─────────────────────┘ │
└──────────────────────────────────────────────────────┘

ModelBuilder dispatches by input tensor count + shapes:
  6 tensors → context_encoding / chunked_prefill
  4 tensors → token_generation / speculation
No dummy tensors needed. No uniform-signature constraint.
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
    │                                         │
    │  .vision_encoder (separate wrapper)     │
    │  .context_encoding_model (6 tensors)    │  ◄── overridden
    │  .token_generation_model (inherited)    │
    │  .speculation_model      (inherited)    │
    │                                         │
    │  forward()               (overridden)   │
    │  create_graph_builders() (overridden)   │
    │  create_vision_graph_builders()         │
    │  _prepare_image_injection_tensors()     │
    └────────────────┬────────────────────────┘
                     │
    ┌────────────────▼────────────────────────┐
    │  SmolVLMNxDModelForImageTextToText      │  (smolvlm/modeling_smolvlm.py)
    │                                         │
    │  _model_cls = NxDSmolVLMTextModel       │
    │  _vision_encoder_cls =                  │
    │      NeuronIdefics3VisionEncoder        │
    └─────────────────────────────────────────┘
```

### 3c. Traced module layer (nn.Module)

The key insight: the VLM text decoder subclasses `NxDDecoderModelForCausalLM` and
overrides `forward()` to accept `image_embeds` + `image_token_mask`. It stashes them
on `self` and delegates to `super().forward()`, which calls the overridden
`compute_input_embeddings()` hook.

```
    ┌──────────────────────────────────────────────────────┐
    │  NxDDecoderModelForCausalLM                          │  (modeling_decoder.py)
    │                                                      │
    │  forward(input_ids, position_ids,                    │
    │          seq_ids, sampling_params)                    │
    │                                                      │
    │  compute_input_embeddings(input_ids):                │  ◄── new hook
    │      return self.embed_tokens(input_ids)             │
    └──────────────────┬───────────────────────────────────┘
                       │
    ┌──────────────────▼───────────────────────────────────┐
    │  NxDDecoderModelForImageTextToText                   │  (vlm_decoder.py)
    │                                                      │
    │  forward(input_ids, position_ids,                    │
    │          seq_ids, sampling_params,                    │
    │          image_embeds, image_token_mask):             │
    │      self._image_embeds = image_embeds               │
    │      self._image_token_mask = image_token_mask       │
    │      return super().forward(input_ids, position_ids, │
    │                             seq_ids, sampling_params) │
    │                                                      │
    │  compute_input_embeddings(input_ids):                │
    │      h = self.embed_tokens(input_ids)                │
    │      h[self._image_token_mask] = self._image_embeds  │
    │      return h                                        │
    └──────────────────┬───────────────────────────────────┘
                       │
    ┌──────────────────▼───────────────────────────────────┐
    │  NxDSmolVLMTextModel                                 │  (smolvlm/modeling_smolvlm.py)
    │                                                      │
    │  (fills in embed_tokens, layers, norm, lm_head       │
    │   — architecture-specific, like NxDLlamaModel)       │
    └──────────────────────────────────────────────────────┘
```

**Why self-stashing works:** `torch.jit.trace` captures the attribute write as part
of the graph. We verified that calling the traced model with different `image_embeds`
values produces correct (non-cached) results.

**What this buys:**
- Token generation graph is traced via `super().forward()` with 4 args — it never
  sees image args, never calls `compute_input_embeddings()` with stashed state.
- Context encoding graph is traced via the overridden `forward()` with 6 args —
  the tracer captures the stash + injection path.
- The base class change is a one-line refactor: `self.embed_tokens(input_ids)` →
  `self.compute_input_embeddings(input_ids)`.

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
    │              .self_attn: NeuronSigLIPAttention              │
    │              .mlp: NeuronSigLIPMLP                          │
    │      .post_layernorm: LayerNorm                            │
    │  .connector: Idefics3Connector (maps vision → text dim)   │
    │                                                            │
    │  forward(pixel_values) → image_features [B, seq_len, dim] │
    └────────────────────────────────────────────────────────────┘
```

---

## 4. Changes Required to the Base Classes

### 4a. Multi-bundle support in `NxDPreTrainedModel` (implemented)

`NxDPreTrainedModel` stores `_traced_models: dict[str, ScriptModule]`
and `graph_builders: dict[str, dict[str, NxDGraphBuilder]]`. The outer dict
keys are bundle names (`"model"` for single-bundle, `"vision"` / `"text"` for VLM).

| Method                    | Change                                               |
|---------------------------|------------------------------------------------------|
| `__init__`                | `traced_model` → `traced_models` dict                |
| `compile()`               | Iterates bundles, one `ModelBuilder.trace()` each    |
| `save()`                  | Saves `model.pt` or `model_{name}.pt` per bundle     |
| `_from_pretrained()`      | Loads all bundles by name                            |
| `_load_weights_from_path` | Initializes each bundle separately                   |
| `shard_checkpoint()`      | Shards per bundle, per-bundle subdirs for multi      |
| New: `get_checkpoint_loader_fn(bundle_name)` | Overridable per-bundle loader    |

`NxDModelForCausalLM` and `NxDModelForEmbedding` wrap their builders in
`{"model": {...}}` and extract `traced_models["model"]` in `__init__`.
Existing model subclasses are unaffected.

### 4b. Embedding hook in `NxDDecoderModelForCausalLM`

One method extraction (no-op for text-only models):

```python
# modeling_decoder.py, line 224

# BEFORE:
inputs_embeds = self.embed_tokens(input_ids)

# AFTER:
inputs_embeds = self.compute_input_embeddings(input_ids)

# New method:
def compute_input_embeddings(self, input_ids):
    return self.embed_tokens(input_ids)
```

### What stays unchanged

| Component                           | Change needed?                      |
|-------------------------------------|-------------------------------------|
| `NxDDecoderModelForCausalLM`        | +1 method extraction (line 224)     |
| `NxDDecoderWrapperForCausalLM`      | None                                |
| `NxDDecoderBuilderForCausalLM`      | None                                |
| `NxDGenerationMixin._sample()`      | None                                |
| Token generation graph              | None                                |
| Speculation graph                   | None                                |
| All existing model subclasses       | None                                |

---

## 5. Builder Layer

```
NxDGraphBuilder (graph_builder.py)  ← ABC
       │
       ├──► NxDDecoderBuilderForCausalLM         (existing, unchanged)
       │       input_generator() → 4 tensors
       │
       ├──► NxDDecoderBuilderForImageTextToText   (new)
       │       extends NxDDecoderBuilderForCausalLM
       │       input_generator() → 6 tensors:
       │         (input_ids, position_ids, seq_ids, sampling_params,
       │          image_embeds [B, active_tokens, hidden_size],
       │          image_token_mask [B, active_tokens])
       │
       └──► NxDVisionEncoderBuilder               (new)
               input_generator() → 1 tensor:
                 pixel_values [B × max_num_images, 3, image_size, image_size]
```

`NxDModelForImageTextToText.create_graph_builders()` returns:

```python
{
    "vision": {
        "vision_encoding": NxDVisionEncoderBuilder(...)
    },
    "text": {
        "context_encoding": NxDDecoderBuilderForImageTextToText(...),  # 6 tensors
        "token_generation": NxDDecoderBuilderForCausalLM(...),         # 4 tensors
    }
}
```

Each outer key becomes a separate `ModelBuilder.trace()` call and a separate
`model_{name}.pt` file. Graphs within the same bundle share weights.
`compile()`, `save()`, and `load_weights()` iterate over bundles automatically
via the refactored `NxDPreTrainedModel`.

The VLM class overrides `get_checkpoint_loader_fn(bundle_name)` to return
different weight extraction logic for `"vision"` vs `"text"` bundles.

---

## 6. Wrapper Layer

```
NxDModelWrapper (model_wrapper.py)
       │
       ├──► NxDDecoderWrapperForCausalLM         (existing, unchanged)
       │       Used as-is for token_generation and speculation.
       │
       ├──► NxDDecoderWrapperForImageTextToText   (new, extends above)
       │       Used for context_encoding / chunked_prefill.
       │       Overrides _forward() to pass image_embeds + image_token_mask
       │       to the traced model.
       │
       └──► NxDVisionEncoderWrapper               (new)
               Wraps the vision encoder traced model.
               forward(pixel_values) → image_features
```

---

## 7. Forward Dispatch During Generation

### 7a. Orchestrator dispatch (`NxDModelForImageTextToText.forward`)

```
forward(input_ids, position_ids, seq_ids, pixel_values=None, ...)
   │
   │  ┌─ Once per prompt (prefill) ──────────────────────────────┐
   │  │                                                          │
   │  │  1. Run vision encoder (model_vision.pt):                     │
   │  │     image_features = vision_encoder(pixel_values)        │
   │  │                                                          │
   │  │  2. Prepare injection tensors:                           │
   │  │     image_embeds, image_token_mask =                     │
   │  │         _prepare_image_injection_tensors(                │
   │  │             input_ids, image_features)                   │
   │  │                                                          │
   │  │  3. Run context encoding (model_text.pt, 6 tensors):       │
   │  │     context_encoding_model(                              │
   │  │         input_ids, position_ids, seq_ids,                │
   │  │         sampling_params, image_embeds, image_token_mask) │
   │  │     set kv_cache_populated = True                        │
   │  │                                                          │
   │  └──────────────────────────────────────────────────────────┘
   │
   │  ┌─ Decode loop (identical to CausalLM) ───────────────────┐
   │  │                                                          │
   │  │  token_generation_model(                                 │
   │  │      input_ids, position_ids, seq_ids, sampling_params)  │
   │  │                                                          │
   │  │  No image args. No dummy tensors.                        │
   │  │  KV cache already has vision-injected representations.   │
   │  │                                                          │
   │  └──────────────────────────────────────────────────────────┘
```

### 7b. Inside the traced graph (context encoding path)

```
NxDDecoderModelForImageTextToText.forward(
    input_ids, position_ids, seq_ids, sampling_params,
    image_embeds, image_token_mask)
   │
   │  self._image_embeds = image_embeds           ← stash
   │  self._image_token_mask = image_token_mask   ← stash
   │  │
   │  super().forward(input_ids, position_ids, seq_ids, sampling_params)
   │      │
   │      │  ... mask computation (unchanged) ...
   │      │
   │      │  inputs_embeds = self.compute_input_embeddings(input_ids)
   │      │      │
   │      │      │  h = self.embed_tokens(input_ids)
   │      │      │  h[self._image_token_mask] = self._image_embeds  ← injection
   │      │      │  return h
   │      │
   │      │  ... layer loop, norm, lm_head (unchanged) ...
```

### 7c. Inside the traced graph (token generation path)

```
NxDDecoderModelForCausalLM.forward(                    ← base class, not VLM
    input_ids, position_ids, seq_ids, sampling_params)
   │
   │  ... mask computation (unchanged) ...
   │
   │  inputs_embeds = self.compute_input_embeddings(input_ids)
   │      │
   │      │  return self.embed_tokens(input_ids)       ← base implementation
   │
   │  ... layer loop, norm, lm_head (unchanged) ...
```

The token generation graph is traced through `NxDDecoderModelForCausalLM.forward()`
directly (4 args). It never enters the VLM override. The traced NEFF has no
image-related ops.

### 7d. Chunked prefill with VLM

Chunked prefill splits a long prompt into fixed-size chunks and processes them
sequentially, writing to the KV cache one chunk at a time. For VLM, the vision
encoder runs **once** before chunking — it processes whole images, not text
chunks. Then each chunk gets its own slice of `image_embeds` and
`image_token_mask`, because only some chunks contain `<image>` placeholder
tokens.

```
generate(input_ids, pixel_values=...)
   │
   │  1. Vision encoding (once):
   │     image_features = vision_encoder(pixel_values)
   │
   │  2. Prepare FULL injection tensors (over entire prompt):
   │     full_image_embeds [B, prompt_len, hidden_dim]
   │     full_image_token_mask [B, prompt_len]
   │
   │  3. Chunk loop:
   │     for chunk_start in range(0, prompt_len, chunk_size):
   │       │
   │       │  chunk_ids   = input_ids[:, start:end]
   │       │  chunk_pos   = position_ids[:, start:end]
   │       │  chunk_embeds = full_image_embeds[:, start:end, :]
   │       │  chunk_mask   = full_image_token_mask[:, start:end]
   │       │
   │       │  ┌─ Has any image tokens in this chunk? ──────────┐
   │       │  │                                                 │
   │       │  │  chunk_mask.any() == True:                      │
   │       │  │    chunked_prefill_model(                       │
   │       │  │        chunk_ids, chunk_pos, seq_ids,           │
   │       │  │        sampling_params, chunk_embeds,           │
   │       │  │        chunk_mask)                              │
   │       │  │    → 6 tensors, image injection happens         │
   │       │  │                                                 │
   │       │  │  chunk_mask.any() == False:                     │
   │       │  │    chunked_prefill_model(                       │
   │       │  │        chunk_ids, chunk_pos, seq_ids,           │
   │       │  │        sampling_params, zeros, zeros)           │
   │       │  │    → 6 tensors, injection is a no-op            │
   │       │  │      (mask is all-False, embed values ignored)  │
   │       │  │                                                 │
   │       │  └─────────────────────────────────────────────────┘
   │
   │  4. Decode loop (identical to CausalLM, 4 tensors)
```

Note: the chunked prefill graph always receives 6 tensors (same traced
signature as context encoding). When a chunk has no image tokens, the mask
is all-False and the injection line `h[mask] = embeds` is a no-op —
zeros are fine for the embeds tensor since they're never written.

### 7e. Generation loop and `NxDGenerationMixin`

The text-only `NxDGenerationMixin._sample()` does not know about
`pixel_values`. The VLM orchestrator needs to hook into the generation
flow to run the vision encoder and thread image tensors into prefill.

There are two clean options:

**Option A: Override `_sample()` in `NxDModelForImageTextToText`**

```
NxDGenerationMixin._sample()              ← text-only, unchanged
       │
       ▼
NxDVLMGenerationMixin._sample()           ← VLM override
       │
       │  1. Extract pixel_values from model_kwargs
       │     (HF GenerationMixin passes them through)
       │
       │  2. Run vision encoder once:
       │     image_features = self.vision_encoder(pixel_values)
       │
       │  3. Prepare full image_embeds + image_token_mask
       │     from image_features + input_ids
       │
       │  4. Stash on self:
       │     self._image_embeds = image_embeds
       │     self._image_token_mask = image_token_mask
       │
       │  5. Call super()._sample(...)
       │     which calls self.forward() for prefill
       │     which passes stashed image tensors to
       │     context_encoding_model
       │
       │  Decode loop runs unchanged — forward() sees
       │  single-token input, dispatches to
       │  token_generation_model (4 tensors, no images)
```

**Option B: Override `forward()` in the orchestrator only**

The orchestrator's `forward()` (not the traced module's) already decides
which sub-model to call. The VLM orchestrator overrides it to:
- On first call (prefill): run vision encoder, prepare injection tensors,
  call `context_encoding_model` with 6 args.
- On subsequent calls (decode): call `token_generation_model` with 4 args.

This keeps `_sample()` completely untouched. The generation loop just calls
`self.forward()` as usual, and the orchestrator's `forward()` handles the
vision/text split transparently.

```
NxDModelForImageTextToText.forward(input_ids, position_ids, seq_ids,
                                    pixel_values=None, ...)
   │
   ├─ Prefill (input_ids.shape[-1] > 1):
   │     image_features = self.vision_encoder(pixel_values)
   │     image_embeds, mask = self._prepare_injection(...)
   │     context_encoding_model(input_ids, pos, seq, samp,
   │                            image_embeds, mask)
   │
   └─ Decode (single token):
         token_generation_model(input_ids, pos, seq, samp)
         (inherited from NxDModelForCausalLM.forward)
```

Option B is simpler — it requires no changes to `_sample()` or
`NxDGenerationMixin` at all. The only override is at the orchestrator
`forward()` level, which already exists as a dispatch point.

For chunked prefill, the same principle applies: `prefill_chunk()` is
overridden in the VLM orchestrator to pass the per-chunk image slices.

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
 │    → _export()  [inherited from NxDPreTrainedModel]                  │
 │                                                                      │
 │    1. create_graph_builders() returns:                               │
 │       {"vision": {"vision_encoding": ...},                          │
 │        "text":   {"context_encoding": ...,                          │
 │                   "token_generation": ...}}                         │
 │                                                                      │
 │    2. NxDPreTrainedModel.compile() iterates bundles:                │
 │       "vision" → ModelBuilder.trace() → model_vision.pt             │
 │       "text"   → ModelBuilder.trace() → model_text.pt               │
 │                                                                      │
 │    3. SmolVLMNxDModelForImageTextToText.__init__()                   │
 │       → vision_encoder_wrapper  (wraps traced_models["vision"])     │
 │       → context_encoding_model  (VLM wrapper, traced_models["text"])│
 │       → token_generation_model  (std wrapper, traced_models["text"])│
 │                                                                      │
 │    4. load_weights() iterates bundles:                               │
 │       "vision" → get_checkpoint_loader_fn("vision") → vision weights│
 │       "text"   → get_checkpoint_loader_fn("text")   → text weights  │
 │                                                                      │
 └──────────────────────────────────────────────────────────────────────┘

 ┌────────────────────────────────────────────────────────────────────┐
 │  GENERATE (runtime)                                                │
 │                                                                    │
 │  model.generate(input_ids, pixel_values=...)                       │
 │    → NxDGenerationMixin.generate()                                 │
 │        calls reset() → kv_cache_populated = False                  │
 │        calls super().generate() (HF GenerationMixin)               │
 │          → calls _sample()                                         │
 │                                                                    │
 │  _sample() calls self.forward() — the orchestrator handles          │
 │  the vision/text split transparently (Option B from §7e):          │
 │                                                                    │
 │    ┌─ Prefill (first forward() call) ──────────────────────┐       │
 │    │ self.forward(input_ids, pos_ids, ..., pixel_values)    │       │
 │    │   → NxDModelForImageTextToText.forward():              │       │
 │    │     1. vision_encoder(pixel_values) → image_features   │       │
 │    │        runs on Neuron device (model_vision.pt)        │       │
 │    │     2. prepare_injection(input_ids, image_features)    │       │
 │    │        → image_embeds, image_token_mask                │       │
 │    │     3. context_encoding_model(input_ids, pos_ids,      │       │
 │    │            seq_ids, samp, image_embeds, mask)           │       │
 │    │        → 6 tensors, model_text.pt                      │       │
 │    │        → embed + inject + layers + norm + lm_head      │       │
 │    └────────────────────────────────────────────────────────┘       │
 │                                                                    │
 │    ┌─ Prefill with chunked_prefill ────────────────────────┐       │
 │    │ (alternative when prefill_chunk_size > 0)              │       │
 │    │                                                        │       │
 │    │ self.forward(input_ids, pos_ids, ..., pixel_values)    │       │
 │    │   → vision_encoder(pixel_values) → image_features      │       │
 │    │   → prepare full image_embeds, mask for whole prompt   │       │
 │    │   → for each chunk:                                    │       │
 │    │       slice image_embeds/mask for this chunk range     │       │
 │    │       prefill_chunk(chunk_ids, chunk_pos, seq_ids,     │       │
 │    │           samp, chunk_embeds, chunk_mask)               │       │
 │    │       → 6 tensors, model_text.pt                       │       │
 │    └────────────────────────────────────────────────────────┘       │
 │                                                                    │
 │    ┌─ Decode loop (identical to CausalLM) ─────────────────┐       │
 │    │ self.forward(one_token, pos_ids, ...)                  │       │
 │    │   → NxDModelForImageTextToText.forward():              │       │
 │    │     no pixel_values → dispatches to                    │       │
 │    │     token_generation_model(4 tensors, model_text.pt)  │       │
 │    │   No vision encoder call. No image tensors.            │       │
 │    └────────────────────────────────────────────────────────┘       │
 │                                                                    │
 └────────────────────────────────────────────────────────────────────┘
```

---

## 9. Config Extension

`NxDVLMNeuronConfig` extends `NxDNeuronConfig` with vision-specific fields:

| Field             | Type  | Default | Description                                   |
|-------------------|-------|---------|-----------------------------------------------|
| `max_num_images`  | int   | 1       | Max images per input (affects vision batch)    |
| `image_size`      | int   | 512     | Input image resolution (static for tracing)    |
| `image_seq_len`   | int   | 64      | Tokens produced per image by vision encoder    |

Text decoder config specialization per graph is identical to CausalLM
(see [ARCHITECTURE_CAUSAL_LM.md §9](ARCHITECTURE_CAUSAL_LM.md#9-config-specialization-per-graph)).

---

## 10. Refactoring Suggestions for the `smolvlm-with-refactor` Branch

The current branch implementation differs from this proposed design in several ways.
Here are the suggested changes, ordered by impact:

### 10a. Multi-bundle support in `NxDPreTrainedModel` (implemented)

**Current (branch):** `NxDPreTrainedModel` stores a list of traced models with a
custom wrapper class pairing each model with its builders.

**Implemented:** `_traced_models` is a `dict[str, ScriptModule]` keyed by
bundle name (`"model"`, `"vision"`, `"text"`). `graph_builders` is
`dict[str, dict[str, NxDGraphBuilder]]`. `compile()`, `save()`, and
`load_weights()` iterate over bundles automatically. No wrapper class needed —
parallel dicts stay in sync because they're produced from the same
`create_graph_builders()` return value. Per-bundle weight loading is supported
via overridable `get_checkpoint_loader_fn(bundle_name)`.

### 10b. Remove dummy image tensors from token generation

**Current (branch):** Token generation wrapper creates dummy `image_embeds` and
`image_token_mask` tensors to match a uniform 6-tensor graph signature.

**Proposed:** Token generation uses the standard 4-tensor `NxDDecoderBuilderForCausalLM`
and `NxDDecoderWrapperForCausalLM`. The ModelBuilder router distinguishes context
encoding (6 tensors) from token generation (4 tensors) by input count and shape.

### 10c. Use `compute_input_embeddings()` hook instead of modifying the base forward

**Current (branch):** `vlm_decoder.py` duplicates or heavily patches
`NxDDecoderModelForCausalLM.forward()` to add image injection.

**Proposed:** Extract a one-line `compute_input_embeddings()` method in the base
class. VLM subclass overrides `forward()` to stash image args on `self`, calls
`super().forward()`, and overrides `compute_input_embeddings()` for injection.
This was validated to work with `torch.jit.trace`.

### 10d. Rename classes to match HF task convention

**Current (branch):** `NxDVLMModelForCausalLM`, `NxDVLMContextDecoderWrapper`, etc.

**Proposed:** Use `ImageTextToText` suffix consistently:
- `NxDModelForImageTextToText`
- `NxDDecoderModelForImageTextToText`
- `NxDDecoderBuilderForImageTextToText`
- `NxDDecoderWrapperForImageTextToText`

### 10e. Eliminate VLM-specific wrapper subclasses where possible

**Current (branch):** `NxDVLMContextDecoderWrapper` and `NxDVLMTokenGenerationWrapper`
are separate classes.

**Proposed:**
- Token generation: use `NxDDecoderWrapperForCausalLM` as-is (no subclass).
- Context encoding: one subclass (`NxDDecoderWrapperForImageTextToText`) that
  overrides `_forward()` to pass the two extra tensors.

---

## 11. File Reference (Proposed)

```
optimum/neuron/models/inference/
├── modeling_utils.py                             # + NeuronModelForImageTextToText
├── ARCHITECTURE_CAUSAL_LM.md                     # text-only architecture doc
├── ARCHITECTURE_IMAGE_TEXT_TO_TEXT.md             # this document
├── backend/
│   ├── config.py                                 # + NxDVLMNeuronConfig
│   ├── pretrained_model.py                       # multi-bundle support (implemented)
│   └── modules/
│       └── decoder/
│           ├── modeling_decoder.py               # + compute_input_embeddings() hook
│           ├── vlm_decoder.py                    # NxDModelForImageTextToText (orchestrator)
│           │                                     # NxDDecoderModelForImageTextToText (traced)
│           ├── vlm_builders.py                   # NxDDecoderBuilderForImageTextToText
│           │                                     # NxDVisionEncoderBuilder
│           ├── vlm_wrappers.py                   # NxDDecoderWrapperForImageTextToText
│           │                                     # NxDVisionEncoderWrapper
│           ├── decoder_wrappers.py               # (unchanged)
│           └── decoder_builders.py               # (unchanged)
├── smolvlm/
│   └── modeling_smolvlm.py                       # SmolVLMNxDModelForImageTextToText
│                                                 # NxDSmolVLMTextModel
│                                                 # NeuronIdefics3VisionEncoder
│                                                 # NeuronSigLIP* layers
└── ...
```
