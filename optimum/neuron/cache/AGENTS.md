# Cache Subsystem Agent Guide

## Module Overview

| File | Purpose | Neuronx required? |
|------|---------|:-:|
| `cleanup.py` | Local compile cache inspection and cleanup (`get_local_cache_status`, `cleanup_local_cache`) | No |
| `hub_cache.py` | HF Hub cache proxy, monkey-patching, registry, sync with retry | Yes (runtime) |
| `entries/cache_entry.py` | `ModelCacheEntry` dataclass — serialization, arch digest, hash | No |
| `entries/single_model.py`, `entries/multi_model.py` | Concrete entry types | No |
| `training.py`, `traced.py` | Legacy cache wrappers for training and traced model paths | Yes |
| `optimum_neuron_cc_wrapper.py` | Compiler wrapper hooks | Yes |

CLI commands live in `optimum/commands/neuron/cache.py` (not in this directory).

## Local Compile Cache

### Path Resolution
1. `cache_dir` kwarg (explicit)
2. `NEURON_COMPILE_CACHE_URL` env var (local path or S3 URL)
3. Default: `/var/tmp/neuron-compile-cache`

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

## libneuronxla Monkey-Patching

The Hub cache integration works by replacing `libneuronxla.create_compile_cache` at runtime:

```python
# hub_cache.py — hub_neuronx_cache() context manager
patch_everywhere("create_compile_cache", hf_create_compile_cache, "libneuronxla")
try:
    yield  # all compilations during this block use CompileCacheHfProxy
finally:
    patch_everywhere("create_compile_cache", create_compile_cache, "libneuronxla")
```

`CompileCacheHfProxy` wraps a local `CompileCacheFs` (or `CompileCacheS3`) and adds Hub lookup/download as a fallback. It delegates all writes to the local cache — Hub upload only happens via explicit `synchronize()`.

The patching utility is in `optimum/neuron/utils/patching.py`.

## Hub Cache Sync

`synchronize_hub_cache()` uploads the local cache to a HF Hub repo:
1. Calls `cleanup_local_cache()` to remove failed entries and stale locks (avoid syncing poison)
2. Creates `CompileCacheHfProxy` pointing at the Hub repo
3. Calls `proxy.synchronize()` which uses `upload_folder()` with `parent_commit` for optimistic concurrency
4. On HTTP 412 conflict: exponential backoff + jitter, up to 5 retries (constants: `_SYNC_MAX_RETRIES`, `_SYNC_BASE_WAIT_SECS`, `_SYNC_MAX_WAIT_SECS`)

### Registry Structure
Cache entries are indexed under `0_REGISTRY/{optimum_version}/{arch_digest}/{hash}.json` with symlink aliases at `0_REGISTRY/{optimum_version}/{model_type}/{org}/{model}/{hash}.json`.

## CLI Commands

All defined in `optimum/commands/neuron/cache.py`:

| Command | Description |
|---------|-------------|
| `cache create` | Create a Hub cache repo |
| `cache set` | Set active Hub cache repo locally |
| `cache synchronize` | Upload local cache to Hub (cleans up first) |
| `cache lookup` | Search Hub cache for compiled models |
| `cache status` | Show local cache entry counts, sizes, compiler versions |
| `cache cleanup` | Remove failed/locked/empty entries from local cache |

`cleanup` flags: `--all` (include empty), `--old-versions`, `--wipe`, `--dry-run`, `--cache_dir`.

## libneuronxla APIs

Used from `libneuronxla.neuron_cc_cache`:
- `CacheUrl.get_cache_url(cache_dir=None)` — resolve cache URL
- `create_compile_cache(cache_url)` — factory for `CompileCacheFs` or `CompileCacheS3`
- `cache.get_hlos()` → `(hlos, locked, done, failed)` — categorized HLO paths
- `cache.clear_locks()` — remove all `.lock` files
- `cache.clean()` — `shutil.rmtree` the cache

## Testing

CPU-only unit tests (no Neuron hardware):
- `tests/decoder/test_cache_cleanup.py` — tests for `cleanup.py` (local cache status/cleanup)
- `tests/decoder/test_cache_sync_retry.py` — tests for Hub sync retry logic

Integration tests (require Neuron hardware + Hub access):
- `tests/decoder/test_cache.py` — end-to-end Hub cache sync: compile, sync, verify registry entries

CI:
- `.github/workflows/test_cpu_compilation.yml` — runs CPU-only cache unit tests
- `.github/workflows/test_inf2_llm.yml` — runs `test_cache.py` on inf2 hardware
