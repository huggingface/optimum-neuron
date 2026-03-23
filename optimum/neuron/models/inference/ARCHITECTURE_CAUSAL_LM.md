# NxD Decoder Model Architecture

This document describes the current class hierarchy, compilation flow, and runtime
dispatch for decoder-only NxD inference models in `optimum/neuron/models/inference/`.

It is validated against the current branch implementation in:

- `backend/pretrained_model.py`
- `backend/modules/decoder/modeling_decoder.py`
- `backend/modules/decoder/decoder_builders.py`
- `backend/modules/decoder/decoder_wrappers.py`
- `backend/modules/generation/generation_utils.py`
- `auto_models.py`

---

## 1. Class Hierarchy Overview

```
                          ┌──────────────────────────────┐
                          │  NeuronModel                 │  (modeling_base.py)
                          │  (ABC)                       │
                          └─────────┬────────────────────┘
                                    │
                          ┌─────────▼────────────────────┐
                          │  NeuronPreTrainedModel       │  (modeling_utils.py)
                          │  (ABC)                       │
                          │                              │
                          │  .export()                   │
                          │  .from_pretrained()          │
                          │  .get_neuron_config()        │
                          └─────────┬────────────────────┘
                                    │
                          ┌─────────▼────────────────────┐
                          │  NeuronModelForCausalLM      │  (modeling_utils.py)
                          │                              │
                          │  task = "text-generation"    │
                          └─────────┬────────────────────┘
                                    │
          ┌─────────────────────────┼──────────────────────────┐
          │                         │                          │
┌─────────▼────────────┐  ┌────────▼────────────┐  ┌──────────▼────────────┐
│ NxDGenerationMixin   │  │ NxDPreTrainedModel  │  │ NeuronModelForCausalLM│
│ (generation_utils)   │  │ (pretrained_model)  │  │ (modeling_utils)      │
│                      │  │                     │  │                       │
│ generate()           │  │ _export()           │  │ public task type      │
│ _sample()            │  │ _from_pretrained()  │  │                       │
│ _assisted_decoding() │  │ compile()           │  │                       │
│                      │  │ save()              │  │                       │
│                      │  │ load_weights()      │  │                       │
└─────────┬────────────┘  └────────┬────────────┘  └──────────┬────────────┘
          │                        │                          │
          └────────────────────────┼──────────────────────────┘
                                   │
                         ┌─────────▼────────────────────────┐
                         │  NxDModelForCausalLM             │
                         │  (modeling_decoder.py)           │
                         │                                  │
                         │  Owns graph wrappers:            │
                         │    .context_encoding_model       │  or
                         │    .chunked_prefill_model        │
                         │    .token_generation_model       │
                         │    .speculation_model            │  (optional)
                         │                                  │
                         │  forward()                       │
                         │  prefill_chunk()                 │
                         │  reset()                         │
                         │  create_graph_builders()         │
                         └─────────┬────────────────────────┘
                                   │
      ┌────────────────────────────┼──────────────────────────────────────────────┐
      │                            │                                              │
┌─────▼──────────────────┐  ┌──────▼──────────────────┐  ┌────────────────────────▼───────┐
│ LlamaNxDModelForCausalLM│ │ MixtralNxDModelForCausalLM│ │ Qwen3MoeNxDModelForCausalLM   │
│ GraniteNxDModelFor...   │ │ Gemma3NxDModelFor...     │ │ Llama4NxDModelForCausalLM      │
│ Qwen2NxDModelFor...     │ │                          │ │                                │
│ Qwen3NxDModelFor...     │ │                          │ │                                │
│ Phi3NxDModelFor...      │ │                          │ │                                │
│ SmolLM3NxDModelFor...   │ │                          │ │                                │
└─────────────────────────┘  └──────────────────────────┘  └────────────────────────────────┘
```

The concrete model class mostly provides `_model_cls`, state-dict conversion,
and model-specific compiler/config overrides. Graph creation, wrapper setup,
runtime dispatch, and generation loops live in `NxDModelForCausalLM`.


## 2. Traced Model Layer

```
                         ┌──────────────────────────────────────────────────┐
                         │  nn.Module                                       │
                         └──────────┬───────────────────────────────────────┘
                                    │
                         ┌──────────▼───────────────────────────────────────┐
                         │  NxDDecoderModelForCausalLM                      │
                         │  (modeling_decoder.py)                           │
                         │                                                  │
                         │  .embed_tokens                                   │
                         │  .layers                                         │
                         │  .norm                                           │
                         │  .lm_head                                        │
                         │  .kv_mgr: KVCacheManager                         │
                         │  .sampler: Sampler | None                        │
                         │                                                  │
                         │  forward(input_ids, position_ids,                │
                         │          seq_ids, sampling_params)               │
                         │  _forward_from_embeddings(...)                   │
                         └──────────┬───────────────────────────────────────┘
                                    │
                         ┌──────────▼───────────────────────────────────────┐
                         │  NxDLlamaModel                                   │
                         │  NxDGraniteModel                                 │
                         │  NxDQwen3MoeModel                                │
                         │  ...                                             │
                         └──────────────────────────────────────────────────┘
```

`NxDDecoderModelForCausalLM` is the traced `nn.Module` that is compiled by
`neuronx_distributed.trace.ModelBuilder`.

Model-specific traced classes populate the architecture pieces only:

- embeddings
- decoder layers
- final norm
- output head

The traced base class owns:

- KV cache management
- attention-mask construction for each graph shape
- optional on-device sampling
- optional `output_logits`
- the `_forward_from_embeddings()` entry used by VLM-style subclasses


## 3. Graph Variants and Compatibility

Current text-generation models compile one bundle named `model` containing:

- `context_encoding` or `chunked_prefill`
- `token_generation`
- `speculation_model` optionally

Runtime wrappers map those graph tags to attributes:

- `context_encoding_model`
- `chunked_prefill_model`
- `token_generation_model`
- `speculation_model`

### Context encoding

- Active tokens = `max_context_length`
- Batch size = `ctx_batch_size`
- No prior KV cache is read
- Uses a full lower-triangular causal mask
- Used for regular prompt prefill when `prefill_chunk_size == 0`

### Chunked prefill

- Active tokens = `prefill_chunk_size`
- Batch size = `ctx_batch_size`
- Reads existing KV cache and scatters per-position updates
- Uses a prior-cache mask plus an in-chunk causal `active_mask`
- On-device sampling is forcibly disabled for this graph
- Replaces context encoding when `prefill_chunk_size > 0`

### Token generation

- Active tokens = `1`
- Batch size = `tkg_batch_size`
- Reads KV cache and updates one position
- Uses only the prior-cache mask
- Enables continuous-batching reorder logic in the wrapper
- Uses `priority_model_idx=0` during compilation

### Speculation

- Active tokens = `speculation_length`
- Batch size = `tkg_batch_size`
- Reads KV cache and updates multiple new positions
- Uses a prior-cache mask plus a causal `active_mask`
- Uses `priority_model_idx=0` during compilation

### Important compatibility rules

`NxDNeuronConfig` enforces:

- speculative decoding is incompatible with on-device sampling
- speculative decoding is incompatible with chunked prefill

So a model can expose at most one of these runtime families:

- context encoding + token generation (+ optional speculation)
- chunked prefill + token generation


## 4. Wrapper Layer

`NxDModelWrapper` in `backend/model_wrapper.py` is an empty `nn.Module` base.

The real runtime adapter is `NxDDecoderWrapperForCausalLM` in
`backend/modules/decoder/decoder_wrappers.py`.

Its responsibilities are:

1. Convert `int64` inputs to `int32`.
2. Pad sequence length to the static shape expected by the compiled graph.
3. Split large runtime batches into compiled-size chunks.
4. Pad undersized batches safely when needed.
5. Reorder token-generation inputs for continuous batching.

### Sequence padding behavior

- Context encoding pads `input_ids` to `max_context_length` using `pad_token_id`.
- Context encoding pads `position_ids` to `max_context_length` using `1`.
- Chunked prefill pads by repeating the last real token and last real position.

That chunked-prefill padding is intentional: it keeps logit gathering anchored on
the last real token and makes duplicate KV scatter writes a no-op overwrite.

### Batch handling behavior

If runtime batch size equals compiled batch size, the wrapper calls the traced
model directly.

If runtime batch size is smaller than compiled batch size, the wrapper pads:

- tensor inputs by repeating the first batch row
- `seq_ids` with unused cache slots rather than zeros

If runtime batch size is larger than compiled batch size, the wrapper executes
multiple compiled calls and concatenates the outputs.

### Continuous batching behavior

Reordering is only applied for `token_generation_model` when
`continuous_batching=True`.

The wrapper sorts by `seq_ids` before execution and restores the original order
afterward.


## 5. Builder Layer and Compilation

`NxDModelForCausalLM.create_graph_builders()` returns:

```python
{
    "model": {
        "context_encoding": ...,      # or "chunked_prefill"
        "token_generation": ...,
        "speculation_model": ...,     # optional
    }
}
```

Each graph is an `NxDDecoderBuilderForCausalLM`.

The builder provides:

- `active_tokens` for the traced graph shape
- `max_tokens` used to size `n_positions`
- example inputs for tracing
- `DecoderModelInstanceForCausalLM`, which sets input/output aliasing for KV cache tensors

### Example-input generation

`input_generator()` traces with:

- `input_ids`: zeros of shape `[batch_size, active_tokens]`
- `position_ids`:
  - `arange(active_tokens)` for context encoding
  - zeros for token generation, chunked prefill, and speculation
- `seq_ids`: `arange(batch_size)`
- `sampling_params`: zero tensor sized from `prepare_sampling_params(1)`

### KV-cache aliasing

`DecoderModelInstanceForCausalLM.get()` aliases each KV cache input tensor to the
corresponding traced output index so runtime cache updates stay in-place.

If `output_logits=True`, the alias offsets move by one because traced outputs become:

- sampled tokens or logits
- gathered full logits
- KV cache outputs


## 6. Config Specialization Per Graph

`NxDModelForCausalLM` creates graph-specific deep copies of `NxDNeuronConfig`.

| Parameter            | context_encoding            | chunked_prefill             | token_generation      | speculation            |
|----------------------|-----------------------------|-----------------------------|-----------------------|------------------------|
| `batch_size`         | `ctx_batch_size`            | `ctx_batch_size`            | `tkg_batch_size`      | `tkg_batch_size`       |
| `active_tokens`      | `max_context_length`        | `prefill_chunk_size`        | `1`                   | `speculation_length`   |
| `max_tokens`         | `max_context_length`        | `sequence_length`           | `sequence_length`     | `sequence_length`      |
| `on_device_sampling` | inherited                   | forced to `False`           | inherited             | inherited              |
| `prefill_chunk_size` | inherited                   | inherited                   | forced to `0`         | inherited              |
| `priority_model_idx` | `None`                      | `None`                      | `0`                   | `0`                    |

Important nuance:

- `ctx_batch_size` is `1` only when `continuous_batching=True`
- otherwise `ctx_batch_size == batch_size`
- `tkg_batch_size` is always `batch_size`


## 7. Runtime Dispatch

`NxDModelForCausalLM.forward()` dispatches by sequence length and prompt position:

```text
if input_ids.shape[-1] > 1 and position_ids.min() == 0:
    if prefill_chunk_size > 0:
        raise ValueError
    else:
        use context_encoding_model

elif input_ids.shape[-1] == speculation_length:
    use speculation_model

else:
    use token_generation_model
```

Notes:

- multi-token `forward()` is only valid for regular context encoding
- when chunked prefill is enabled, callers must use `generate()` or `prefill_chunk()`
- successful prompt prefill sets `kv_cache_populated = True`


## 8. Generation Loop

`NxDGenerationMixin.generate()` is intentionally stateless: it always calls
`reset()` before delegating to Hugging Face `GenerationMixin.generate()`.

`_sample()` handles the real runtime loop.

### Prompt setup

- Rejects inputs longer than compiled `sequence_length`
- Rejects runtime batch size larger than compiled `batch_size`
- Rejects left padding
- Builds `position_ids` from `attention_mask` when provided
- Builds `seq_ids` as `arange(batch_size)` when absent

### Sampling path selection

CPU sampling is used when:

- on-device sampling is disabled, or
- chunked prefill is enabled

When CPU sampling is used, the loop removes HF temperature/top-k/top-p warpers
and applies the fused Optimum warper instead.

### Prefill behavior

Without chunked prefill:

- `forward(full_prompt, ...)`
- sample next token from logits or device-produced tokens

With chunked prefill:

- iterate prompt chunks of size `prefill_chunk_size`
- stop early once a right-padded chunk is all PAD
- call `prefill_chunk()` sequentially
- always sample the first next token on CPU from the last chunk outputs

### Decode behavior

After the first next token is chosen:

- append it to `input_ids`
- increment `position_ids`
- repeatedly call `forward(next_tokens, ...)`
- stop when `stopping_criteria` marks all sequences finished

`_assisted_decoding()` is also implemented, but it requires:

- assistant model without on-device sampling
- batch size 1
- greedy decoding


## 9. Attention and KV Cache Semantics Inside the Traced Graph

`NxDDecoderModelForCausalLM._forward_from_embeddings()` identifies the graph type
from the input shape:

- chunked prefill: `chunk_size > 0` and `seq_len == chunk_size`
- context encoding: `seq_len > 1` and `seq_len != speculation_length`
- speculation: `seq_len == speculation_length`
- otherwise token generation

### Context encoding

- `past_key_values = None`
- attention mask is a full lower-triangular matrix of shape
  `[batch, 1, n_positions, n_positions]`
- `active_mask = None`
- after decoding, only the hidden state at `max(position_ids)` is projected

### Chunked prefill

- starts from `kv_mgr.get_cache(cache_size)`
- if compiled batch is smaller than KV cache batch dimension, the read path is indexed by `seq_ids`
- prior-cache attention mask covers already-written positions
- `active_mask` is a causal triangle masked further by `actual_tokens_mask`
- cache updates use scatter semantics through `kv_mgr.update_cache(...)`

### Token generation

- reads the existing KV cache
- builds a prior-cache attention mask of shape `[batch, 1, 1, n_positions]`
- uses no explicit `active_mask`

### Speculation

- reads the existing KV cache
- expands the prior-cache mask to `[batch, 1, speculation_length, n_positions]`
- builds a causal `active_mask` over the speculative window

For all graph types, the traced path is:

1. embeddings
2. decoder layers
3. final norm
4. KV cache update
5. LM head
6. optional on-device sampling
7. optional gathered logits when `output_logits=True`


## 10. Export and Load Lifecycle

### Export path

`NeuronModelForCausalLM.export()`:

1. resolves the concrete model class through the registry in `auto_models.py`
2. calls `NxDPreTrainedModel._export()`
3. creates graph builders through `create_graph_builders()`
4. compiles each bundle with `ModelBuilder.trace()`
5. instantiates the NxD model wrapper object
6. optionally loads weights onto device

Compiled bundle naming is:

- `model.pt` for the default single bundle
- `model_<bundle_name>.pt` for multi-bundle models

### Load path

`from_pretrained()`:

1. loads the serialized `NeuronConfig`
2. checks current instance compatibility
3. rebuilds graph builders from config
4. loads compiled bundles from disk or Hub
5. instantiates the model
6. loads pre-sharded or raw checkpoint weights


## 11. Current Model Coverage

The current branch registers these causal-LM inference models in `auto_models.py`:

- Gemma3
- Granite
- Llama
- Llama4
- Mixtral
- Phi3
- Qwen2
- Qwen3
- Qwen3 MoE
- SmolLM3

Related but separate inference architectures also exist for:

- Qwen3 embeddings
- SmolVLM image-text-to-text


## 12. File Reference

```
optimum/neuron/models/inference/
├── modeling_utils.py                          # NeuronPreTrainedModel, NeuronModelForCausalLM
├── auto_models.py                             # registry wiring for concrete inference classes
├── backend/
│   ├── config.py                              # NxDNeuronConfig
│   ├── graph_builder.py                       # NxDGraphBuilder base
│   ├── model_wrapper.py                       # NxDModelWrapper base
│   ├── pretrained_model.py                    # compile / load / save lifecycle
│   └── modules/
│       ├── decoder/
│       │   ├── modeling_decoder.py            # traced decoder + runtime orchestrator
│       │   ├── decoder_builders.py            # graph tracing setup and KV aliasing
│       │   ├── decoder_wrappers.py            # runtime shape adaptation and batching
│       │   └── vlm_decoder.py                 # VLM-specific extension of the same pattern
│       ├── generation/
│       │   ├── generation_utils.py            # generate / _sample / assisted decoding
│       │   └── sampling.py                    # sampling params and on-device sampler
│       ├── attention/
│       │   ├── attention_base.py              # base Neuron attention path
│       │   └── gqa.py                         # head sharding strategy
│       └── kvcache/
│           └── kv_cache_manager.py            # KV cache layout and updates
├── llama/
├── llama4/
├── mixtral/
├── gemma3/
├── granite/
├── phi3/
├── qwen2/
├── qwen3/
├── qwen3_moe/
└── smollm3/
```
