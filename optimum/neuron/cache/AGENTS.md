# Cache Subsystem Agent Guide

## Module Overview

| File | Purpose | Neuronx required? |
|------|---------|:-:|
| `bucket_utils.py` | Bucket resolution, model ID encoding, config hashing, path translation | No |
| `bucket_cache.py` | Context manager (`hub_neuronx_cache`), `fetch_cache`, `sync_cache`, `lookup_cache` | No |
| `bucket_cli.py` | Standalone subprocess for bucket API (runs via `uv run --with "huggingface_hub>=1.0"`) | No |
| `hub_cache.py` | Public API shim — re-exports `hub_neuronx_cache`, `synchronize_hub_cache`, `select_hub_cached_entries` | No |
| `cleanup.py` | Local compile cache inspection and cleanup (`get_local_cache_status`, `cleanup_local_cache`) | No |
| `entries/cache_entry.py` | `ModelCacheEntry` dataclass — kept for backwards compat with callers | No |
| `entries/single_model.py`, `entries/multi_model.py` | Concrete entry types — kept for backwards compat | No |
| `training.py` | Training cache wrapper | Yes |
| `traced.py` | Deprecated — traced model caching now handled by bucket context | No |
| `optimum_neuron_cc_wrapper.py` | Compiler wrapper hooks | Yes |

CLI commands live in `optimum/commands/neuron/cache.py` (not in this directory).

## Architecture: Bucket-Based Cache

NEFFs are stored in HF Storage Buckets (not Git repos). The bucket API requires
`huggingface_hub >= 1.0`, which conflicts with `transformers < 1.0` constraint.
This is solved by running bucket operations in an isolated subprocess via
`uv run --with "huggingface_hub>=1.0"` (see `bucket_cli.py`).

### Bucket Layout

```
aws-neuron/optimum-neuron-neff-cache/
  neuronxcc-{compiler_version}/              # full version with hash (e.g. 2.28.4405.0+abc123)
    MODULE_{hlo_hash}+{flags_hash}/          # flat NEFFs (hf-mount compatible)
      model.neff
      model.done
      model.hlo_module.pb
      compile_flags.json
    {encoded_model_id}/                      # per-model area (fetch/lookup/sync)
      exports/{optimum_neuron_version}/      # export records (advisory, for lookup only)
        {8-char-hash}.json                   # flat neuron config dict
      MODULE_{hlo_hash}+{flags_hash}/        # same NEFFs (Xet dedup = zero storage cost)
        ...
```

- **Flat area**: matches local cache layout. Used by hf-mount.
- **Per-model area**: organized by model. Used by `fetch_cache()` and `lookup_cache()`.
- **Xet dedup**: identical files stored once, referenced from both locations.
- **model_id encoding**: `/` replaced with `--` (e.g. `meta-llama--Llama-3.1-8B`).

### Data Flow

```
with hub_neuronx_cache(model_id, export_config):
    # __enter__: snapshot MODULE dirs, fetch from bucket (if not mounted)
    ... compilation ...
    # __exit__: diff MODULE dirs, upload new ones + export record to bucket
```

The context manager is the primary API. It handles:
- **Fetch on enter** (non-mount mode): downloads cached NEFFs from per-model area
- **Sync on exit**: uploads new MODULE dirs to both flat + per-model areas
- **Export record**: flat neuron config uploaded last (signals export completed)

No monkey-patching of `libneuronxla` — the compiler uses its normal `CompileCacheFs`.

### Bucket Resolution

Priority: `NEURON_CACHE_BUCKET` env var > locally saved (`~/.cache/huggingface/optimum_neuron_cache_bucket`) > default (`aws-neuron/optimum-neuron-neff-cache`).

## Local Compile Cache

### Path Resolution
1. `cache_dir` kwarg (explicit)
2. `NEURON_COMPILE_CACHE_URL` env var (what libneuronxla uses)
3. `NEURON_CC_FLAGS --cache_dir=` flag
4. Default: `/var/tmp/neuron-compile-cache`

### Directory Layout
```
/var/tmp/neuron-compile-cache/
  neuronxcc-<compiler_version>/
    MODULE_<hlo_hash>+<flags_hash>/
      model.hlo_module.pb          # serialized HLO graph
      compile_flags.json           # compiler flags
      model.neff                   # compiled binary (success only)
      model.done                   # empty marker (success only)
      model.log                    # error log (failure only)
      model.hlo_module.pb.lock    # lock file (active compilation only)
```

### Entry States

| State | neff | done | log | lock | Action |
|-------|:---:|:---:|:---:|:---:|--------|
| Success | yes | yes | — | — | Keep |
| Failed | — | — | yes | — | **Poisoned** — remove to allow recompilation |
| Locked | — | — | — | yes | Active compilation in progress |
| Stale lock | — | — | — | yes | Orphaned from crash — remove (if no `neuronx-cc` running) |
| Empty | — | — | — | — | HLO uploaded, never compiled — harmless |

Failed entries permanently block recompilation of the same graph. This is the primary reason `cleanup_local_cache()` exists.

## Export Records

Export records are flat neuron config dicts stored in the bucket under
`{compiler}/{model_id}/exports/{optimum_neuron_version}/{hash}.json`.

They are **advisory only** — used by `lookup_cache()` and `select_hub_cached_entries()`
to show what was previously exported. Not used for NEFF selection (the compiler handles that).

Key fields: `static_batch_size`, `static_sequence_length`, `tensor_parallel_size`,
`instance_type`, `float_dtype`, `task`, `compiler_version`, `optlevel`.

## hf-mount Support

With `hf-mount` and `--upperdir` (requires CAP_SYS_ADMIN):
- Mount provides lazy-loaded NEFFs from the flat bucket area
- No fetch needed (mount replaces it)
- Sync still uploads new NEFFs via API on context exit
- Mount detection: `os.path.ismount(cache_dir)` or `NEURON_CACHE_MOUNTED=0|1` override

## CLI Commands

All defined in `optimum/commands/neuron/cache.py`:

| Command | Description |
|---------|-------------|
| `cache create` | Create/verify a cache bucket |
| `cache set` | Set active cache bucket locally |
| `cache synchronize` | Upload deferred-sync MODULE dirs (scans for `.bucket_meta.json`) |
| `cache fetch` | Pre-warm local cache from bucket for a model |
| `cache lookup` | List cached export configs for a model |
| `cache status` | Show local cache entry counts, sizes, compiler versions |
| `cache cleanup` | Remove failed/locked/empty entries from local cache |

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `NEURON_CACHE_BUCKET` | Override cache bucket ID |
| `NEURON_CACHE_MOUNTED` | Override mount detection (`0`/`1`) |
| `NEURON_COMPILE_CACHE_URL` | Local cache directory (read by libneuronxla) |
| `NEURON_CC_FLAGS` | Compiler flags (may include `--cache_dir=`) |

## Testing

CPU-only unit tests:
- `tests/cache/test_bucket_utils.py` — path helpers, config hash, bucket resolution
- `tests/cache/test_bucket_cache.py` — fetch/sync/lookup against `dacorvo/neuron-compile-cache` test bucket (requires `uv`)
- `tests/decoder/test_cache_cleanup.py` — local cache status and cleanup

Integration tests (require Neuron hardware + Hub access):
- `tests/decoder/test_cache.py` — end-to-end cache: compile, sync, verify
