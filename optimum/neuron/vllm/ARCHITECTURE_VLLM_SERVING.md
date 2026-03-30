# vLLM Serving with Optimum Neuron

This document describes how vLLM integrates with optimum-neuron to serve
models on AWS Trainium/Inferentia. It covers the plugin mechanism,
class hierarchy, request lifecycle, batch state management, prefill
strategies (including chunked prefill and VLM), sampling, and
data-parallel serving.

It is validated against the current branch implementation in:

- `plugin.py`
- `platform.py`
- `worker.py`
- `runner.py`
- `model_loader.py`
- `sampler.py`
- `server_manager.py`
- `reverse_proxy.py`

---

## 1. Overview

The optimum-neuron vLLM integration is a **plugin-based backend** that
lets vLLM serve Neuron-compiled models through the standard OpenAI-compatible
API. The key design constraints are:

- Neuron models are compiled for **static shapes** — input dimensions must
  match what was compiled.
- KV cache is managed **internally** by the Neuron model, not by vLLM's
  block manager.
- Tensor parallelism is handled **inside the Neuron runtime** — vLLM sees
  a single worker process per model instance.
- Prefill and decode use **different compiled graphs** and cannot be batched
  together.

```
                          ┌──────────────────────────┐
                          │  vLLM Engine (V1)        │
                          │                          │
                          │  Scheduler → SchedulerOutput
                          └────────┬─────────────────┘
                                   │
                          ┌────────▼─────────────────┐
                          │  OptimumNeuronWorker     │
                          │  (WorkerBase)            │
                          └────────┬─────────────────┘
                                   │
                          ┌────────▼─────────────────┐
                          │  OptimumNeuronModelRunner │
                          │  (batch state + dispatch) │
                          └────────┬─────────────────┘
                                   │
                          ┌────────▼─────────────────┐
                          │  OptimumNeuronModel      │
                          │  (nn.Module wrapper)     │
                          └────────┬─────────────────┘
                                   │
                          ┌────────▼─────────────────┐
                          │  NxD Model               │
                          │  (compiled graphs on     │
                          │   Neuron hardware)       │
                          └──────────────────────────┘
```


## 2. Plugin Discovery and Platform Registration

vLLM discovers the Neuron backend through Python entry points.

### Entry point

In `pyproject.toml`:

```toml
[project.entry-points."vllm.platform_plugins"]
optimum_neuron = "optimum.neuron.vllm.plugin:register"
```

`plugin.register()` returns the string
`"optimum.neuron.vllm.platform.OptimumNeuronPlatform"`, which vLLM
instantiates as the active platform.

### Platform configuration

`OptimumNeuronPlatform` extends `UnspecifiedPlatform` and sets:

| Attribute                | Value                            | Why                                                    |
|--------------------------|----------------------------------|--------------------------------------------------------|
| `device_name`            | `"neuron"`                       | Device identifier                                      |
| `device_type`            | `"cpu"`                          | Prevents vLLM from moving tensors to XLA prematurely   |
| `ray_device_key`         | `"neuron_cores"`                 | Ray resource scheduling key                            |
| `device_control_env_var` | `"NEURON_RT_VISIBLE_CORES"`      | Controls which Neuron cores are visible                 |

`check_and_update_config()` applies these overrides to the vLLM config:

1. Set worker class to `OptimumNeuronWorker`
2. Set distributed executor backend to `"uni"` when `world_size > 1`
3. Disable prefix caching (unsupported)
4. Set `block_size = max_model_len` (no KV cache fragmentation)
5. Fix `max_num_batched_tokens = max(current, max_model_len)` to prevent
   long prompts from being stuck (see section 8)
6. Reject MLA (Multi-Layer Attention) configurations
7. Patch `ModelConfig.verify_with_parallel_config()` to skip the
   attention-heads-divisible-by-TP check (Neuron uses padding)


## 3. Class Hierarchy

### Full class tree

```
vllm.platforms.interface.UnspecifiedPlatform
  └── OptimumNeuronPlatform                          (platform.py)

vllm.worker.worker_base.WorkerBase
  └── OptimumNeuronWorker                            (worker.py)

ABC
  └── OptimumNeuronModelRunner                       (runner.py)
        ├── OptimumNeuronModelRunnerForCausalLM
        │     └── OptimumNeuronModelRunnerForImageTextToText
        └── OptimumNeuronModelRunnerForEmbedding

nn.Module
  └── OptimumNeuronModel                             (model_loader.py)
        ├── OptimumNeuronModelForCausalLM
        ├── OptimumNeuronModelForImageTextToText
        └── OptimumNeuronModelForEmbedding

vllm.v1.sample.sampler.Sampler
  └── NeuronSampler                                  (sampler.py)
```

### Runner factory

`OptimumNeuronModelRunner.create(vllm_config)` dispatches to the
concrete runner based on task and modality:

```python
task = vllm_config.model_config.task or "generate"

if task == "generate":
    if model_config.is_multimodal_model:
        → OptimumNeuronModelRunnerForImageTextToText
    else:
        → OptimumNeuronModelRunnerForCausalLM

elif task == "embed":
    → OptimumNeuronModelRunnerForEmbedding
```

### Model wrapper mapping

| Runner class                               | Model wrapper class                        | NxD auto class                         |
|--------------------------------------------|--------------------------------------------|----------------------------------------|
| `OptimumNeuronModelRunnerForCausalLM`      | `OptimumNeuronModelForCausalLM`            | `NeuronModelForCausalLM`               |
| `OptimumNeuronModelRunnerForImageTextToText`| `OptimumNeuronModelForImageTextToText`     | `NeuronModelForImageTextToText`        |
| `OptimumNeuronModelRunnerForEmbedding`     | `OptimumNeuronModelForEmbedding`           | `NeuronModelForEmbedding`              |


## 4. Configuration Flow

vLLM configuration parameters map to Neuron compilation parameters:

| vLLM parameter              | Neuron parameter              | Notes                                      |
|-----------------------------|-------------------------------|--------------------------------------------|
| `max_num_seqs`              | `batch_size`                  | Must be ≤ compiled batch_size              |
| `max_model_len`             | `sequence_length`             | Must match compiled sequence_length        |
| `tensor_parallel_size`      | `tp_degree`                   | Must match compiled TP degree              |
| `dtype`                     | `torch_dtype`                 | Must match compiled dtype                  |

### Task mapping

vLLM task names map to HuggingFace task names:

```python
"generate"            → "text-generation"
"embed"               → "feature-extraction"
"image-text-to-text"  → "image-text-to-text"
```

### Platform-level fixes

vLLM V1 defaults `enable_chunked_prefill=True`, which sets
`max_num_batched_tokens` to 2048. The engine core later disables
vLLM-level chunked prefill because Neuron returns empty `kv_cache_groups`.
This leaves a 2048-token budget with no ability to chunk, causing prompts
longer than 2048 tokens to be stuck in the scheduler forever.

The platform fixes this by setting
`max_num_batched_tokens = max(current, max_model_len)`.

Neuron-level chunked prefill (when `prefill_chunk_size > 0`) is handled
entirely inside the runner, not by vLLM's scheduler.


## 5. Model Loading

`OptimumNeuronModel.create()` follows a three-tier lookup:

```
1. Try loading NeuronConfig from model directory
   ├── Found → model is pre-compiled → from_pretrained()
   └── Not found ↓

2. Search hub cache for compatible pre-compiled artifacts
   via select_hub_cached_entries(model, task, batch_size,
       sequence_length, tp, dtype)
   ├── Found → log warning, export using cached config
   └── Not found ↓

3. If allow_non_cached_model:
   └── Export on the fly (with warning)
   Else:
   └── Raise ValueError with hub cache URL
```

After loading, the factory validates:

- `max_num_seqs ≤ neuron_config.batch_size` — the scheduler cannot
  exceed compiled batch capacity since vLLM's block manager has no
  visibility into Neuron's internal KV cache.


## 6. Request Lifecycle

### End-to-end flow

```
┌──────────────────────────────────────────────────────────────────────┐
│ vLLM Scheduler                                                       │
│                                                                      │
│  Produces SchedulerOutput:                                           │
│    .scheduled_new_reqs     → new prompts                             │
│    .scheduled_cached_reqs  → ongoing decode sequences                │
│    .finished_req_ids       → completed sequences                     │
└──────────────┬───────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│ OptimumNeuronWorker.execute_model(scheduler_output)                  │
│   └── model_runner.execute_model(scheduler_output)                   │
└──────────────┬───────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│ OptimumNeuronModelRunnerForCausalLM.execute_model()                  │
│                                                                      │
│  1. Remove finished requests → free KV cache slots                   │
│  2. Decode phase (if cached requests exist):                         │
│     └── _prepare_decode() → _get_next_tokens()                      │
│  3. Prefill phase (if new requests exist):                           │
│     └── _execute_prefill()                                           │
│         ├── standard: _prepare_prompt() → _get_next_tokens()         │
│         └── chunked:  _execute_chunked_prefill()                     │
│  4. Cache sampled tokens in output_token_ids                         │
│  5. Return ModelRunnerOutput                                         │
└──────────────────────────────────────────────────────────────────────┘
```

### Important: prefill and decode use different graphs

Prefill and decode requests are **never batched together**. The runner
always processes decode first (token generation graph), then prefill
(context encoding or chunked prefill graph). The results are concatenated
before returning.


## 7. Batch State Management

### OptimumNeuronCachedBatch

vLLM's block manager has no visibility into Neuron's internal KV cache.
Instead, `OptimumNeuronCachedBatch` maintains a fixed-size array of
request slots — one per compiled sequence position.

```
cached_requests: [slot_0, slot_1, ..., slot_{max_num_seqs-1}]
                    │        │
                    │        └── OptimumNeuronCachedRequest or None
                    └── OptimumNeuronCachedRequest or None
```

Each `OptimumNeuronCachedRequest` tracks:

| Field              | Type              | Purpose                                |
|--------------------|-------------------|----------------------------------------|
| `req_id`           | `str`             | vLLM request identifier                |
| `seq_id`           | `int`             | Slot index = KV cache position         |
| `sampling_params`  | `SamplingParams`  | Per-request sampling config            |
| `prompt_token_ids` | `list[int]`       | Original prompt tokens                 |
| `output_token_ids` | `list[int]`       | Generated tokens (appended each step)  |

### Slot allocation

`add_request()` scans for the first `None` slot and assigns the request.
The slot index becomes `seq_id`, which maps directly to the KV cache
position in the Neuron model.

### Slot deallocation

`remove_requests()` sets finished slots to `None`, freeing them for
reuse by future requests.

### Why the runner caches tokens

vLLM's scheduler does not send previously generated tokens back to the
worker. The runner must cache `output_token_ids` itself to:

1. Know the last generated token for the next decode step
2. Compute `position_id = len(prompt) + len(output) - 1`
3. Build sampling metadata with full output history (for penalties)

### Temperature conversion

vLLM represents greedy decoding as `temperature=0.0`. Neuron requires
`top_k=1` instead. The conversion happens once at request creation:

```python
# In OptimumNeuronCachedRequest.__post_init__():
if temperature == 0.0:
    top_k = 1
    top_p = 1.0
    temperature = 1.0
```


## 8. Prefill Dispatch

`_execute_prefill()` routes to one of two paths:

```
_execute_prefill(scheduler_output)
    │
    ├── prefill_chunk_size > 0
    │     └── _execute_chunked_prefill()
    │
    └── prefill_chunk_size == 0
          └── _prepare_prompt() → _get_next_tokens()
```

### Standard prefill (context encoding)

When `prefill_chunk_size == 0`:

1. Collect new requests via `_collect_new_requests()`
2. Pad all prompt token sequences to the same length
3. Build `input_ids`, `position_ids`, `seq_ids` tensors
4. Call `model.forward()` — uses the context encoding graph
5. Sample the first token (on-device or CPU)

This can process multiple sequences in one batch call, up to
`ctx_batch_size`.

### Chunked prefill

When `prefill_chunk_size > 0`:

Processing is done **one sequence at a time**. This is not only simpler
but also faster and consumes less device memory than batching multiple
sequences (benchmarked on trn1.32xlarge with Llama 3.1 8B).

For each new sequence:

1. Divide prompt into chunks of `prefill_chunk_size` tokens
2. For each chunk:
   - Pad to `chunk_size` using repeat-last-token (same as the NxD wrapper)
   - Call `model.prefill_chunk_vllm(chunk_ids, chunk_pos, seq_id, sampling_params)`
   - Each chunk writes its KV entries into the cache at the positions
     given by `chunk_pos`; duplicate positions from padding overwrite
     the same slot (harmless no-op)
3. Keep only the **last chunk's** logits — earlier chunks populate the
   KV cache but their logits are not needed
4. After all sequences, sample first tokens from collected logits via
   `NeuronSampler`

```
Example: prompt = 5 tokens, chunk_size = 3

              KV cache state
              (positions filled)
              ─────────────────
Chunk 0:  input_ids  = [t0, t1, t2]
          position_ids = [ 0,  1,  2]
          ───────────────────────────────────────────────
          → prefill_chunk_vllm()
          → KV cache now holds positions 0, 1, 2
          → logits returned but discarded

Chunk 1:  input_ids  = [t3, t4, t4]    ← last token repeated as padding
          position_ids = [ 3,  4,  4]  ← last position repeated
          ───────────────────────────────────────────────
          → prefill_chunk_vllm()
          → KV cache now holds positions 0, 1, 2, 3, 4
            (position 4 written twice — second write is a no-op overwrite)
          → logits returned and KEPT (this is the last chunk)

Sample first generated token from chunk 1 logits.
The sampled token will be generated at position 5 during decode.
```

Chunked prefill always uses CPU sampling — on-device sampling is forcibly
disabled for the chunked prefill graph. This only affects the **first
generated token** of each sequence; all subsequent tokens are produced by
the token generation graph, which uses on-device sampling when enabled.


## 9. VLM Prefill

`OptimumNeuronModelRunnerForImageTextToText` extends the CausalLM runner
with multimodal input handling.

### Image extraction

`_extract_pixel_values(new_request_data)` extracts pixel values from
vLLM's multimodal features:

1. Filter for `"image"` modality entries in `mm_features`
2. Extract `"pixel_values"` tensors
3. Handle multi-tile images: `[N_tiles, C, H, W]` → unsqueeze to 4D
4. Concatenate all tiles → `[1, N_total_tiles, C, H, W]`

### VLM prefill flow

The VLM runner overrides `_execute_prefill()` to handle pixel values.
Each sequence is processed independently:

```
For each new VLM request:
    │
    ├── Extract pixel_values from mm_features
    │
    ├── If chunked (prefill_chunk_size > 0):
    │     1. Pad input_ids to multiple of chunk_size
    │     2. model.prepare_vlm_prefill(padded_ids, pixel_values)
    │        → pre-computes image embeddings once
    │     3. Loop over chunks:
    │        └── model.prefill_chunk_vllm(chunk) → logits
    │     4. model.reset_vlm_prefill()
    │        → clears cached embeddings
    │
    └── If non-chunked:
          1. model.forward(input_ids, ..., pixel_values=pixel_values)
          2. Returns logits or token IDs directly
```

The key optimization is that image embeddings are **pre-computed once**
before chunked processing and reused across all chunks. The
`prepare_vlm_prefill()` / `reset_vlm_prefill()` bracket ensures
embeddings are cached only for the duration of one sequence's prefill.

### Decode phase

Token generation for VLM is **identical to text-only** — the image
information is already captured in the KV cache after prefill. The
CausalLM decode path is reused without modification.


## 10. Decode (Token Generation)

`_prepare_decode()` builds tensors for cached sequences:

For each scheduled cached request:

1. Look up `OptimumNeuronCachedRequest` by `req_id`
2. Extract `input_token = output_token_ids[-1]` (last generated token)
3. Compute `position_id = num_tokens() - 1`
4. Collect `seq_id` and `sampling_params`

All decode requests are batched into a single `_get_next_tokens()` call
that uses the token generation graph.

### Sequence sorting

Before calling the Neuron model, the model wrapper sorts inputs by
`seq_id` for cache locality and restores the original order afterward:

```python
# In OptimumNeuronModelForCausalLM.forward():
sorted_seq_ids, sorted_indices = torch.sort(seq_ids)
# ... forward with sorted inputs ...
restored_indices = torch.argsort(sorted_indices)
output = torch.index_select(output, 0, restored_indices)
```


## 11. Sampling

### Two sampling paths

| Condition                               | Sampling path    | Sampler used        |
|-----------------------------------------|------------------|---------------------|
| `on_device_sampling=True` and not chunked prefill | On-device | NxD on-device sampler |
| `on_device_sampling=False`              | CPU              | `NeuronSampler`     |
| Chunked prefill (any config)            | CPU              | `NeuronSampler`     |

### On-device sampling

When enabled, the compiled graph includes a fused sampler kernel. The
model returns **token IDs** `[batch, 1]` directly instead of logits.
The runner unsqueezes to match the CPU sampler output shape.

Sampling parameters are passed as a tensor `[batch, 3]` with columns
`[top_k, top_p, temperature]`.

### CPU sampling (NeuronSampler)

`NeuronSampler` extends vLLM's `Sampler` base class and uses the
Neuron-optimized `NeuronTopkToppSampler` for random sampling:

```
NeuronSampler.sample(logits, sampling_metadata):
    │
    ├── If all_greedy: return greedy_sample(logits)
    │
    ├── If all_random:
    │     1. Apply argmax-invariant logits processors
    │     2. Call NeuronTopkToppSampler(logits, [top_k, top_p, temp])
    │     └── Return random samples
    │
    └── Mixed batch:
          1. Compute greedy samples
          2. Compute random samples (as above)
          3. Select per-row based on temperature < epsilon
```

### Sampling metadata

`create_sampling_metadata()` builds vLLM's `SamplingMetadata` from
cached requests. It tracks:

- Temperature, top_k, top_p per request
- Frequency, presence, repetition penalties
- Prompt and output token histories (padded to max length)
- Allowed token ID masks
- Bad word token IDs
- Greedy vs random flags for batch optimization

Unsupported logits processor features (not yet implemented):
`min_p`, `logits_bias`, `min_length`.


## 12. Data-Parallel Serving

When `--data-parallel-size N` is passed to `optimum-cli neuron serve`,
multiple independent vLLM servers are spawned behind a reverse proxy.

### Architecture

```
                       ┌────────────────────────────┐
                       │  Client                    │
                       └────────┬───────────────────┘
                                │  :8080
                       ┌────────▼───────────────────┐
                       │  RoundRobinProxy            │
                       │  (aiohttp)                  │
                       │                             │
                       │  /health → aggregate check  │
                       │  /*      → round-robin fwd  │
                       └───┬─────────────┬───────────┘
                           │             │
              :8081        │             │        :8082
         ┌─────────────────▼──┐   ┌──────▼─────────────────┐
         │  vLLM Server (DP0) │   │  vLLM Server (DP1)     │
         │  cores 0-7         │   │  cores 8-15            │
         └────────────────────┘   └────────────────────────┘
```

### VLLMServerManager

Spawns and manages N independent vLLM server subprocesses:

1. **Core partitioning**: `_core_range(rank)` partitions
   `NEURON_RT_VISIBLE_CORES` across DP ranks. Each rank gets
   `tp_size` contiguous cores.

2. **Process spawning**: Each server runs
   `python -m vllm.entrypoints.openai.api_server` with:
   - `NEURON_RT_VISIBLE_CORES` set to its core range
   - `VLLM_WORKER_MULTIPROC_METHOD=spawn`
   - `--host 127.0.0.1` (only proxy-accessible)

3. **Shutdown**: SIGTERM → 10s grace period → SIGKILL

**Critical**: Subprocesses are spawned **without** `start_new_session=True`.
They inherit the parent's process group so that vLLM can clean up its own
EngineCore workers on SIGTERM. Isolating them in separate process groups
causes Neuron core leaks.

### RoundRobinProxy

An aiohttp-based reverse proxy that:

- Distributes requests round-robin across upstream servers
- Streams responses back to clients
- Provides an aggregated `/health` endpoint (200 only if ALL backends healthy)
- Waits up to 600s for all backends to become healthy at startup

### Distributed initialization

In DP mode, the worker pre-initializes `torch.distributed` with the
correct topology (`dp_size` participants, gloo backend) before vLLM's
own distributed initialization (which runs with `world_size=1` since
Neuron handles TP internally).


## 13. Unsupported Features and Constraints

| Feature                        | Status          | Reason                                          |
|--------------------------------|-----------------|-------------------------------------------------|
| Prefix caching                 | Not supported   | Neuron manages KV cache internally              |
| Pipeline parallelism           | Not supported   | Only tensor parallelism                         |
| LoRA adapters                  | Not supported   | Asserted in worker constructor                  |
| Speculative decoding           | Not supported   | Asserted in worker constructor                  |
| MLA (Multi-Layer Attention)    | Not supported   | Rejected in platform config                     |
| Pin memory                     | Not available   | Neuron does not support pinned memory           |
| `min_p` logits processor       | Not yet         | Empty LogitsProcessors passed                   |
| `logits_bias`                  | Not yet         | Empty LogitsProcessors passed                   |
| `min_length`                   | Not yet         | Empty LogitsProcessors passed                   |
| Mixed prefill + decode batch   | Not possible    | Different compiled graphs                       |

### Static shape constraints

- Runtime `max_num_seqs` must be ≤ compiled `batch_size`
- Runtime prompt length must fit within compiled `sequence_length`
- TP degree must match compiled model exactly


## 14. Worker Lifecycle

`OptimumNeuronWorker` implements vLLM's `WorkerBase` contract with
several no-ops due to Neuron's internal management:

| WorkerBase method            | Neuron implementation                          |
|------------------------------|------------------------------------------------|
| `init_device()`              | Initialize distributed env (gloo), set seed    |
| `load_model()`               | Delegate to `model_runner.load_model()`        |
| `get_kv_cache_spec()`        | Return `{}` (Neuron manages KV cache)          |
| `initialize_cache()`         | No-op (assert 1 GPU block, 0 CPU blocks)       |
| `initialize_from_config()`   | No-op                                          |
| `compile_or_warm_up_model()` | No-op (compilation happens during load)         |
| `execute_dummy_batch()`      | No-op (not needed for Neuron DP sync)           |
| `execute_model()`            | Delegate to `model_runner.execute_model()`     |


## 15. File Reference

```
optimum/neuron/vllm/
├── plugin.py               # vLLM entry point: register() → platform class path
├── platform.py             # OptimumNeuronPlatform: config validation and patching
├── worker.py               # OptimumNeuronWorker: WorkerBase with Neuron no-ops
├── runner.py               # Model runners: batch state, prefill, decode, VLM
│                           #   OptimumNeuronCachedRequest
│                           #   OptimumNeuronCachedBatch
│                           #   OptimumNeuronModelRunner (ABC)
│                           #   OptimumNeuronModelRunnerForCausalLM
│                           #   OptimumNeuronModelRunnerForEmbedding
│                           #   OptimumNeuronModelRunnerForImageTextToText
│                           #   create_sampling_metadata()
├── model_loader.py         # Model wrappers: load, sort, forward
│                           #   OptimumNeuronModel (base)
│                           #   OptimumNeuronModelForCausalLM
│                           #   OptimumNeuronModelForImageTextToText
│                           #   OptimumNeuronModelForEmbedding
├── sampler.py              # NeuronSampler: CPU-side top-k/top-p sampling
├── server_manager.py       # VLLMServerManager: DP subprocess spawning
└── reverse_proxy.py        # RoundRobinProxy: aiohttp load balancer

optimum/commands/neuron/
└── serve.py                # CLI: optimum-cli neuron serve

pyproject.toml              # Entry point: vllm.platform_plugins → plugin:register
```
