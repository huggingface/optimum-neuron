# NxD Decoder Model Architecture

This document describes the class hierarchy, compilation flow, and runtime dispatch
for decoder-based inference models in `optimum/neuron/models/inference/backend/`.

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
                          │  .task: str                  │
                          │  .export()                   │
                          │  .from_pretrained()          │
                          │  .get_neuron_config()        │
                          └─────────┬────────────────────┘
                                    │
                          ┌─────────▼────────────────────┐
                          │  NeuronModelForCausalLM      │  (modeling_utils.py)
                          │                              │
                          │  task = "text-generation"    │
                          │  generate() [abstract]       │
                          └─────────┬────────────────────┘
                                    │
          ┌─────────────────────────┼──────────────────────────┐
          │                         │                          │
┌─────────▼────────────┐  ┌────────▼────────────┐  ┌──────────▼────────────┐
│ NxDGenerationMixin   │  │ NxDPreTrainedModel  │  │ NeuronModelForCausalLM│
│ (generation_utils)   │  │ (pretrained_model)  │  │ (modeling_utils)      │
│                      │  │                     │  │                       │
│ generate()           │  │ _export()           │  │ [provides task]       │
│ _sample()            │  │ _from_pretrained()  │  │                       │
│ _assisted_decoding() │  │ compile()           │  │                       │
│                      │  │ save()              │  │                       │
│                      │  │ load_weights()      │  │                       │
└─────────┬────────────┘  └────────┬────────────┘  └──────────┬────────────┘
          │                        │                          │
          └────────────────────────┼──────────────────────────┘
                                   │  (triple inheritance via MRO)
                         ┌─────────▼────────────────────────┐
                         │  NxDModelForCausalLM             │  (modeling_decoder.py:304)
                         │                                  │
                         │  _model_cls = None               │
                         │                                  │
                         │  Owns sub-model wrappers:        │
                         │    .context_encoding_model       │  ─── OR ───
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
                    ┌──────────────▼───────────────────┐
                    │  LlamaNxDModelForCausalLM        │  (llama/modeling_llama.py:310)
                    │  _model_cls = NxDLlamaModel      │
                    ├─────────────────────────────────-┤
                    │  Qwen2NxDModelForCausalLM        │  (_model_cls = ...)
                    │  Qwen3NxDModelForCausalLM        │
                    │  GraniteNxDModelForCausalLM      │
                    │  SmolLM3NxDModelForCausalLM      │
                    │  Phi3NxDModelForCausalLM         │
                    └──────────────────────────────────┘
```

The concrete model class (e.g. `LlamaNxDModelForCausalLM`) only sets `_model_cls`.
Everything else — graph creation, sub-model wrappers, forward dispatch, generation
loops — is inherited from `NxDModelForCausalLM`.


## 2. The Traced Model Layer (nn.Module)

```
                         ┌──────────────────────────────────────────────────┐
                         │  nn.Module                                       │
                         └──────────┬───────────────────────────────────────┘
                                    │
                         ┌──────────▼───────────────────────────────────────┐
                         │  NxDDecoderModelForCausalLM                      │  (modeling_decoder.py:59)
                         │                                                  │
                         │  .embed_tokens  [abstract]                       │  ◄── set by model subclass
                         │  .layers        [abstract]                       │  ◄── set by model subclass
                         │  .norm          [abstract]                       │  ◄── set by model subclass
                         │  .lm_head       [abstract]                       │  ◄── set by model subclass
                         │  .kv_mgr: KVCacheManager                         │
                         │  .sampler: Sampler | None                        │
                         │                                                  │
                         │  forward(input_ids, position_ids,                │
                         │          seq_ids, sampling_params)               │
                         └──────────┬───────────────────────────────────────┘
                                    │
                         ┌──────────▼───────────────────────────────────────┐
                         │  NxDLlamaModel                                   │  (llama/modeling_llama.py:279)
                         │                                                  │
                         │  self.embed_tokens = ParallelEmbedding(...)      │
                         │  self.lm_head = ColumnParallelLinear(...)        │
                         │  self.layers = [NeuronLlamaDecoderLayer(...)]    │
                         │  self.norm = NeuronRMSNorm(...)                  │
                         └──────────────────────────────────────────────────┘
```

`NxDDecoderModelForCausalLM` is the class that gets **traced by neuronx-distributed**.
Its `forward()` is compiled into a static Neuron graph (NEFF). The model-specific
subclass (e.g. `NxDLlamaModel`) only fills in the architecture-specific modules.


## 3. The Four Graph Variants

Each compiled model can contain up to 4 traced graphs, identified by **model tags**.
The graphs share the same weights but differ in input shapes and attention logic.

```
┌──────────────────────────────────────────────────────────────┐
│            Compiled model.pt (torch.jit.ScriptModule)        │
│                                                              │
│  ┌──────────────────────────┐  ┌──────────────────────────┐  │
│  │ context_encoding         │  │ token_generation         │  │
│  │                          │  │                          │  │
│  │ active_tokens =          │  │ active_tokens = 1        │  │
│  │   max_context_length     │  │ batch_size = tkg_bs      │  │
│  │ batch_size = ctx_bs      │  │ KV: scatter 1 token      │  │
│  │ KV: write full seq       │  │ Mask: prior only         │  │
│  │ Mask: causal triangle    │  │ priority_model_idx=0     │  │
│  └──────────────────────────┘  └──────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────┐  ┌──────────────────────────┐  │
│  │ chunked_prefill          │  │ speculation              │  │
│  │ (replaces ctx_enc)       │  │ (optional)               │  │
│  │                          │  │                          │  │
│  │ active_tokens =          │  │ active_tokens =          │  │
│  │   prefill_chunk_size     │  │   speculation_length     │  │
│  │ batch_size = ctx_bs      │  │ batch_size = tkg_bs      │  │
│  │ KV: scatter per-pos      │  │ KV: scatter per-pos      │  │
│  │ Mask: prior + causal     │  │ Mask: prior + causal     │  │
│  │ ODS always off           │  │ priority_model_idx=0     │  │
│  └──────────────────────────┘  └──────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘

Mutual exclusion:
  prefill_chunk_size > 0  →  chunked_prefill  (no context_encoding)
  prefill_chunk_size == 0 →  context_encoding  (no chunked_prefill)
  speculation_length > 0  →  speculation graph is compiled
```


## 4. The Wrapper Layer (Runtime)

Between the orchestrator (`NxDModelForCausalLM`) and the traced JIT model sits
a wrapper that adapts dynamic runtime shapes to the compiled static shapes.

```
NxDModelWrapper (model_wrapper.py)  ← torch.nn.Module, empty base
       │
       ▼
NxDDecoderWrapperForCausalLM (decoder_wrappers.py:33)
       │
       │  One instance per graph variant:
       │    .tag = "context_encoding_model" | "token_generation_model"
       │         | "chunked_prefill_model"  | "speculation_model"
       │    .model = traced_model (shared jit.ScriptModule)
       │    .neuron_config = graph-specific deepcopy of NxDNeuronConfig
       │
       │  forward() pipeline:
       │    1. convert_int64_to_int32()
       │    2. Pad seq dim:
       │         context_encoding → pad to max_context_length
       │         chunked_prefill  → pad to chunk_size (repeat last token)
       │    3. Batch splitting/padding loop:
       │         if input_batch == compiled_batch → _forward()
       │         if input_batch < compiled_batch  → _forward_with_pad()
       │         if input_batch > compiled_batch  → split into chunks
       │    4. _forward():
       │         continuous_batching + token_gen → sort by seq_ids
       │         call traced model
       │         unsort outputs
       │    5. Slice off padding, return
```


## 5. The Builder Layer (Compilation)

```
NxDGraphBuilder (graph_builder.py)  ← ABC
       │
       ▼
NxDDecoderBuilderForCausalLM (decoder_builders.py:26)
       │
       │  Created by NxDModelForCausalLM.create_graph_builders()
       │  One per graph variant. Parameters:
       │    .active_tokens   = {max_context_length, 1, chunk_size, spec_length}
       │    .max_tokens      = sequence_length (KV cache size)
       │    .model_cls       = e.g. NxDLlamaModel
       │    .priority_model_idx = 0 for token_gen/speculation (weight layout opt)
       │
       │  input_generator():
       │    Returns example (input_ids, position_ids, seq_ids, sampling_params)
       │    with correct static shapes for tracing.
       │
       │  get_model_instance():
       │    Returns DecoderModelInstanceForCausalLM which:
       │      - load_module(): instantiates model_cls, sets n_positions
       │      - get(): returns (module, input_output_aliases)
       │        where aliases map KV cache tensors → output indices
       │        (enables in-place KV update without copies)
       │
       ▼
ModelBuilder (neuronx_distributed)
       │
       │  .add(key, model_instance, example_inputs, compiler_args)
       │  .trace() → torch.jit.ScriptModule (the compiled model.pt)
```


## 6. Forward Dispatch During Generation

### 6a. Orchestrator dispatch (`NxDModelForCausalLM.forward`)

```
forward(input_ids, position_ids, seq_ids, ...)
   │
   ├─ input_ids.shape[-1] > 1 AND position_ids starts at 0?
   │     ├─ prefill_chunk_size > 0 → raise ValueError
   │     │    ("use generate() or prefill_chunk() instead")
   │     │
   │     └─ else → context_encoding_model(...)
   │              set kv_cache_populated = True
   │
   ├─ input_ids.shape[-1] == speculation_length?
   │     └─ speculation_model(...)
   │
   └─ else (single token)
         └─ token_generation_model(...)
```

### 6b. Generation loop (`NxDGenerationMixin._sample`)

```
_sample(input_ids, logits_processor, stopping_criteria, ...)
   │
   │  ┌──── PREFILL PHASE ───────────────────────────────────────┐
   │  │                                                          │
   │  │  prefill_chunk_size > 0?                                 │
   │  │    YES:                                                  │
   │  │      for chunk_start in range(0, prompt_len, chunk_sz):  │
   │  │        chunk_ids = input_ids[:, start:end]               │
   │  │        chunk_pos = position_ids[:, start:end]            │
   │  │        skip if chunk is all PAD                          │
   │  │        outputs = prefill_chunk(chunk_ids, chunk_pos,     │
   │  │                                seq_ids, sampling_params) │
   │  │      sample next_tokens (always CPU, is_ods=False)       │
   │  │                                                          │
   │  │    NO:                                                   │
   │  │      outputs = forward(input_ids, position_ids, ...)     │
   │  │      sample next_tokens (ODS or CPU per config)          │
   │  │                                                          │
   │  └──────────────────────────────────────────────────────────┘
   │
   │  ┌──── DECODE LOOP ────────────────────────────────────────┐
   │  │                                                         │
   │  │  while not finished:                                    │
   │  │    position_ids += 1                                    │
   │  │    outputs = forward(next_tokens, position_ids, ...)    │
   │  │      └─► dispatches to token_generation_model           │
   │  │          (or speculation_model if shape matches)        │
   │  │    sample next_tokens                                   │
   │  │    append to input_ids                                  │
   │  │    check stopping_criteria                              │
   │  │                                                         │
   │  └─────────────────────────────────────────────────────────┘
   │
   └─ return input_ids
```


## 7. Attention Mask Computation Inside the Traced Graph

`NxDDecoderModelForCausalLM.forward()` computes attention masks differently
depending on which graph variant is executing:

```
forward(input_ids, position_ids, seq_ids, sampling_params)
   │
   │  Detect graph type by input shape:
   │    _is_chunked_prefill()  →  chunk_size > 0 AND seq_len == chunk_size
   │    _is_context_encoding() →  seq_len > 1 AND seq_len != speculation_length
   │    _is_for_speculation()  →  seq_len == speculation_length
   │    else                   →  token generation (seq_len == 1)
   │
   ├─── CONTEXT ENCODING ────────────────────────────────────────────┐
   │    past_key_values = None (fresh, no KV cache)                  │
   │    attention_mask = lower-triangle [n_pos × n_pos]              │
   │    active_mask = None                                           │
   │    KV write: bulk write (is_for_context_encoding=True)          │
   │    Logits: gather hidden_states at max(position_ids) per batch  │
   │                                                                 │
   ├─── CHUNKED PREFILL ─────────────────────────────────────────────┤
   │    past_key_values = kv_mgr.get_cache()[seq_ids]                │
   │    attention_mask = prior mask (pos < first_chunk_pos)          │
   │        shape: [batch, 1, chunk_size, n_positions]               │
   │    active_mask = causal triangle [chunk × chunk]                │
   │        AND actual_tokens_mask (blocks padded positions)         │
   │    KV write: scatter per-position (is_chunked_prefill=True)     │
   │                                                                 │
   ├─── TOKEN GENERATION ────────────────────────────────────────────┤
   │    past_key_values = kv_mgr.get_cache()                         │
   │    attention_mask = prior mask (pos < current_pos)              │
   │        shape: [batch, 1, 1, n_positions]                        │
   │    active_mask = None (implicit single token)                   │
   │    KV write: scatter single position                            │
   │                                                                 │
   ├─── SPECULATION ─────────────────────────────────────────────────┤
   │    past_key_values = kv_mgr.get_cache()                         │
   │    attention_mask = prior mask (pos < first_spec_pos)           │
   │        shape: [batch, 1, spec_len, n_positions]                 │
   │    active_mask = causal triangle [spec_len × spec_len]          │
   │    KV write: scatter per-position                               │
   └─────────────────────────────────────────────────────────────────┘
   │
   │  Then for all paths:
   │    hidden = embed_tokens(input_ids)
   │    for layer in layers:
   │      hidden, new_kv, cos, sin = layer(hidden, attention_mask,
   │                                        position_ids, past_kv,
   │                                        active_mask, cos, sin)
   │    hidden = norm(hidden)
   │    kv_mgr.update_cache(...)
   │    logits = lm_head(hidden)
   │    if on_device_sampling: sample on device
   │    return [logits_or_tokens, *updated_kv_cache]
```


## 8. End-to-End: From HF Model to Token Output

```
User code:
  model = NeuronModelForCausalLM.export("meta-llama/Llama-3.1-8B", neuron_config)
  model.generate(input_ids, ...)

 ┌───────────────────────────────────────────────────────────────────┐
 │  EXPORT (compile time)                                            │
 │                                                                   │
 │  NeuronModelForCausalLM.export()                                  │
 │    → resolves to LlamaNxDModelForCausalLM via auto_model registry │
 │    → LlamaNxDModelForCausalLM._export()  [inherited from NxDPTM]  │
 │        1. create_graph_builders()                                 │
 │           builds NxDDecoderBuilderForCausalLM per graph variant   │
 │           each builder holds model_cls = NxDLlamaModel            │
 │        2. NxDPreTrainedModel.compile()                            │
 │           → get_builder() → ModelBuilder.trace()                  │
 │           → traces NxDLlamaModel.forward() with example inputs    │
 │           → produces torch.jit.ScriptModule (model.pt)            │
 │        3. LlamaNxDModelForCausalLM.__init__()                     │
 │           creates NxDDecoderWrapperForCausalLM per graph variant  │
 │           all wrappers share the same traced_model                │
 │        4. load_weights() shards HF checkpoint → Neuron device     │
 │                                                                   │
 └───────────────────────────────────────────────────────────────────┘

 ┌───────────────────────────────────────────────────────────────────┐
 │  GENERATE (runtime)                                               │
 │                                                                   │
 │  model.generate(input_ids)                                        │
 │    → NxDGenerationMixin.generate()                                │
 │        calls reset() → kv_cache_populated = False                 │
 │        calls super().generate() (HF GenerationMixin)              │
 │          → calls _sample()                                        │
 │                                                                   │
 │  _sample():                                                       │
 │    ┌─ Prefill ─────────────────────────────────────────────┐      │
 │    │ forward(full_prompt)                                  │      │
 │    │   → NxDModelForCausalLM.forward()                     │      │
 │    │     → context_encoding_model(input_ids, pos_ids, ...) │      │
 │    │       → NxDDecoderWrapperForCausalLM.forward()        │      │
 │    │         → pad to max_context_length                   │      │
 │    │         → traced_model(...)  [runs on Neuron device]  │      │
 │    │           → NxDLlamaModel.forward() [compiled graph]  │      │
 │    │             → embed → layers → norm → lm_head         │      │
 │    │         → slice off padding → return logits/tokens    │      │
 │    └───────────────────────────────────────────────────────┘      │
 │                                                                   │
 │    ┌─ Decode loop ─────────────────────────────────────────┐      │
 │    │ forward(one_token)                                    │      │
 │    │   → token_generation_model(...)                       │      │
 │    │     → pad batch if needed                             │      │
 │    │     → reorder by seq_ids (continuous batching)        │      │
 │    │     → traced_model(...)  [runs on Neuron device]      │      │
 │    │     → unsort, slice, return                           │      │
 │    └───────────────────────────────────────────────────────┘      │
 │                                                                   │
 └───────────────────────────────────────────────────────────────────┘
```


## 9. Config Specialization per Graph

`NxDModelForCausalLM` creates a **deepcopy** of `NxDNeuronConfig` for each graph,
with these overrides:

| Parameter            | context_encoding       | chunked_prefill        | token_generation      | speculation            |
|----------------------|------------------------|------------------------|-----------------------|------------------------|
| `batch_size`         | `ctx_batch_size`       | `ctx_batch_size`       | `tkg_batch_size`      | `tkg_batch_size`       |
| `active_tokens`      | `max_context_length`   | `prefill_chunk_size`   | `1`                   | `speculation_length`   |
| `on_device_sampling` | (inherited)            | **always False**       | (inherited)           | (inherited)            |
| `prefill_chunk_size` | n/a                    | (inherited)            | **forced to 0**       | (inherited)            |
| `priority_model_idx` | None                   | None                   | **0** (weight layout) | **0** (weight layout)  |

`ctx_batch_size` defaults to 1 (process one prompt at a time).
`tkg_batch_size` equals the user-specified `batch_size` (concurrent decoding).


## 10. File Reference

```
optimum/neuron/models/inference/
├── modeling_utils.py                          # NeuronPreTrainedModel, NeuronModelForCausalLM (public API)
├── backend/
│   ├── config.py                              # NxDNeuronConfig
│   ├── graph_builder.py                       # NxDGraphBuilder (ABC)
│   ├── model_wrapper.py                       # NxDModelWrapper (empty nn.Module base)
│   ├── pretrained_model.py                    # NxDPreTrainedModel (compile/save/load lifecycle)
│   └── modules/
│       ├── decoder/
│       │   ├── modeling_decoder.py            # NxDDecoderModelForCausalLM (traced forward)
│       │   │                                  # NxDModelForCausalLM (orchestrator)
│       │   ├── decoder_wrappers.py            # NxDDecoderWrapperForCausalLM (runtime padding/reorder)
│       │   │                                  # Model tags (CONTEXT_ENCODING_MODEL_TAG, etc.)
│       │   └── decoder_builders.py            # NxDDecoderBuilderForCausalLM (tracing setup)
│       ├── generation/
│       │   ├── generation_utils.py            # NxDGenerationMixin (_sample, _assisted_decoding)
│       │   └── sampling.py                    # Sampler, on-device sampling utils
│       ├── attention/
│       │   ├── attention_base.py              # NeuronAttentionBase (perform_prefill, compute_for_token_gen)
│       │   └── gqa.py                         # GQA sharding strategy
│       └── kvcache/
│           └── kv_cache_manager.py            # KVCacheManager (BHSD layout, aliased updates)
├── llama/
│   └── modeling_llama.py                      # NxDLlamaModel, LlamaNxDModelForCausalLM
├── qwen2/
│   └── modeling_qwen2.py                      # Qwen2NxDModelForCausalLM (extends Llama)
├── qwen3/
│   └── modeling_qwen3.py                      # Qwen3NxDModelForCausalLM (extends Llama)
└── ...                                        # granite, phi3, smollm3 (all extend Llama)
```
