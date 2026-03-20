# Plan: Bump vLLM + Neuron SDK Versions

## Context

optimum-neuron pins `vllm == 0.11.0` and Neuron SDK at torch-neuronx 2.8.0. The goal is to get as close to upstream vLLM as possible, so that when vLLM drops its `transformers < 5` constraint ([#30466](https://github.com/vllm-project/vllm/issues/30466)), we only need a version bump. Neuron SDK 2.28.1 is out with torch-neuronx 2.9.0.

---

## Target Versions

| Component | Current | Target |
|-----------|---------|--------|
| vLLM | `0.11.0` | **`0.13.0`** |
| torch-neuronx | `2.8.0.2.10.16998` | `2.9.0.2.12.22436` |
| torch | `2.8.0.*` | `2.9.0.*` |
| neuronx-cc | `2.21.33363.0` | `2.23.6484.0` |
| neuronx_distributed | `0.15.22404` | `0.17.26814` |
| libneuronxla | `2.2.12677.0` | `2.2.15515.0` |
| Python | `>=3.10,<3.12` | `>=3.10,<3.13` (align with Neuron SDK) |

**Why 0.13.0 not 0.17.1**: The vLLM PyPI wheel hard-pins torch (e.g. `torch==2.10.0` for 0.17.1). Since Neuron SDK 2.28.1 ships torch-neuronx 2.9.0, only vLLM 0.12.0 or 0.13.0 (both pin `torch==2.9.0`) are compatible. 0.13.0 is preferred — it includes more bug fixes and the `init_cached_hf_modules` removal is clean.

**Path to 0.17.1+**: When Neuron SDK ships torch 2.10 (expected Neuron ~2.29+), we can bump vLLM to 0.17.x. The code changes for 0.13.0 are identical to those needed for 0.17.1 — all API breaks happened in 0.12.0.

### transformers v5 note

All vLLM ≥0.12.0 add `transformers >= 4.56.0, < 5`. When upstream drops the `< 5` bound, we bump the vLLM pin. No action now.

---

## Code Changes

### 1. `pyproject.toml` — version pins

```toml
# vllm extra
"vllm == 0.13.0"   # was 0.11.0

# neuronx extra
"neuronx-cc==2.23.6484.0"                    # was 2.21.33363.0
"torch-neuronx==2.9.0.2.12.22436"            # was 2.8.0.2.10.16998
"torch==2.9.0.*"                              # was 2.8.0.*
"torchvision==0.24.*"                         # was 0.23.* (match torch 2.9)
"neuronx_distributed==0.17.26814"             # was 0.15.22404
"libneuronxla==2.2.15515.0"                   # was 2.2.12677.0

# Python version
requires-python = ">=3.10,<3.13"              # was >=3.10,<3.12 (align with Neuron SDK)
```

Also update `torchcodec` if needed, and verify numpy constraint compatibility.

### 2. Import path fixes — all introduced in v0.12.0

| Old import (0.11.0) | New import (0.13.0) | File |
|---|---|---|
| `from vllm.worker.worker_base import WorkerBase` | `from vllm.v1.worker.worker_base import WorkerBase` | `worker.py:26` |
| `from vllm.utils import FlexibleArgumentParser` | `from vllm.utils.argparse_utils import FlexibleArgumentParser` | `platform.py:18` |
| `from vllm.utils import is_pin_memory_available, make_tensor_with_pad` | `from vllm.utils.platform_utils import is_pin_memory_available` + `from vllm.utils.torch_utils import make_tensor_with_pad` | `runner.py:25` |

The `vllm.v1.sample.logits_processor.LogitsProcessors` import still works (now a package, `__init__` exports the class).

### 3. Remove `init_cached_hf_modules` — `worker.py:56-60`

Removed from vLLM in v0.13.0. Delete:
```python
if self.model_config.trust_remote_code:
    from vllm.utils import init_cached_hf_modules
    init_cached_hf_modules()
```

### 4. `WorkerBase` interface updates — `worker.py`

The v1 `WorkerBase` has new/changed methods:

| Method | Action |
|--------|--------|
| `execute_model()` return type → `ModelRunnerOutput \| None` | Update type hint |
| `sample_tokens(grammar_output)` (new) | Add: `raise NotImplementedError` |
| `initialize_from_config(kv_cache_config)` param type may change | Verify signature at 0.13.0 and update |
| `compile_or_warm_up_model()` → returns `float` | Return `0.0` |
| `check_health()` (new) | Add as no-op |
| `reset_mm_cache()` (new) | Add as no-op |
| `get_model() -> nn.Module` (new) | Return model or raise |
| `apply_model(fn)` (new) | Apply fn to model |
| `get_model_inspection() -> str` (new) | Return model info string |
| `get_cache_block_size_bytes()` (new) | Return 0 |
| `vocab_size` property (new) | Return from model config |

**Note**: Not all of these may exist in 0.13.0 — some were added later. Verify against the actual 0.13.0 `WorkerBase`. The table above is from 0.17.1; 0.13.0 will be a subset.

### 5. Data structures — no changes expected

Verified compatible at 0.17.1 (0.13.0 will be at least as compatible):
- `SamplingMetadata`: all current fields present
- `ModelRunnerOutput`: `req_ids`, `req_id_to_index`, `sampled_token_ids`, `logprobs`, `prompt_logprobs_dict`, `pooler_output` all present
- `SchedulerOutput`: `scheduled_new_reqs`, `scheduled_cached_reqs`, `finished_req_ids` present
- `NewRequestData`: `prompt_token_ids`, `sampling_params` present
- `CachedRequestData`: `req_ids` present

### 6. Sampler — likely no changes

`Sampler.sample()` signature unchanged through 0.17.1. Our `NeuronSampler` override should work.

---

## Files to Modify

| File | Changes |
|------|---------|
| `pyproject.toml` | vllm pin, all neuronx versions, python version, torchvision |
| `optimum/neuron/vllm/worker.py` | `WorkerBase` import path; remove `init_cached_hf_modules`; add new WorkerBase methods; update signatures |
| `optimum/neuron/vllm/platform.py` | `FlexibleArgumentParser` import path |
| `optimum/neuron/vllm/runner.py` | `is_pin_memory_available` + `make_tensor_with_pad` import paths |
| `optimum/neuron/vllm/sampler.py` | Verify compatibility (likely no changes) |
| `optimum/neuron/vllm/model_loader.py` | Verify `LogitsProcessor` import (likely no changes) |
| `docker/vllm/Dockerfile` | May need base image update for new SDK |
| `.github/workflows/test_inf2_vllm.yml` | May need SDK version update |

---

## Verification

Run validation in the same order CI uses, rather than in an ad hoc order. CI splits some suites across jobs, but the local validation order should still mirror the workflow sequence below.

### 1. Install and import sanity

1. `uv pip install -e ".[neuronx,vllm]"` — dependency resolution succeeds
2. `python -c "from optimum.neuron.vllm.plugin import register; print('OK')"` — all imports resolve
3. `make style_check`

### 2. CPU-only CI order

Match `.github/workflows/test_cpu_only.yml`:

1. `pytest -sv tests/decoder/test_cli.py`
2. `pytest -sv tests/decoder/test_decoder_config.py`
3. `pytest -sv tests/decoder/test_fused_logits_warper.py`
4. `pytest tests/decoder/test_cache_cleanup.py tests/decoder/test_cache_sync_retry.py -v --tb=short`
5. `pytest tests/exporters/test_cpu_compilation.py -v --tb=short`
6. `pytest -sv tests/vllm/test_server_manager.py`
7. `pytest -sv tests/vllm/test_serve_ports.py`

### 3. INF2 LLM CI order

Match `.github/workflows/test_inf2_llm.yml` in the same per-job order CI uses:

1. `pytest -sv tests/decoder/test_modules.py tests/decoder/test_attention.py`
2. `pytest -sv tests/decoder/test_manual_softmax.py`
3. `pytest -sv tests/decoder/test_flash_attention_large_d.py`
4. `pytest -sv tests/decoder/test_device_memory.py`
5. `pytest -sv tests/decoder/test_decoder_export.py`
6. `pytest -sv tests/decoder/test_decoder_hub.py`
7. `pytest -sv tests/decoder/test_decoder_generation.py`
8. `pytest -sv tests/decoder/test_chunked_prefill.py`
9. `pytest -sv tests/decoder/test_decoder_embedding.py`
10. `pytest -sv tests/decoder/test_decoder_pipelines.py`
11. `pytest -sv tests/decoder/test_cache.py`

### 4. INF2 vLLM CI order

Match `.github/workflows/test_inf2_vllm.yml` in the same per-job order CI uses:

1. `pytest -sv tests/vllm/engine/test_vllm_engine_embedding.py`
2. `pytest -sv tests/vllm/engine/test_vllm_engine_generate.py`
3. `pytest -sv tests/vllm/engine/test_vllm_engine_chunked_prefill.py`
4. `pytest -sv tests/vllm/service`
5. `make optimum-neuron-vllm`
6. `pytest -sv tests/vllm/docker`

### 5. INF2 exporters CI order

Match `.github/workflows/test_inf2_export.yml`:

1. `RANDOM_TASK_PER_MODEL=TRUE pytest tests/exporters --ignore=tests/exporters/test_cpu_compilation.py --durations=0 --timeout=300`

### 6. INF2 transformers CI order

Match `.github/workflows/test_inf2_transformers.yml` in the same order CI uses:

1. `pytest tests/inference/transformers/test_cache.py`
2. `pytest tests/inference/transformers/test_export_cli.py`
3. `pytest -m "not slow" tests/inference/transformers/test_modeling.py`
4. `pytest tests/pipelines`

### 7. INF2 diffusers CI order

Match `.github/workflows/test_inf2_diffusers.yml` in the same order CI uses:

1. `pytest tests/inference/diffusers/test_stable_diffusion.py`
2. `pytest tests/inference/diffusers/test_pixart.py`
3. `pytest tests/inference/diffusers/test_flux.py --forked -q`
4. `pytest tests/inference/diffusers/test_cache.py`
5. `pytest tests/inference/diffusers/test_export_cli.py`

### 8. Trainium training CI order

Match `.github/workflows/test_trainium_training.yml`:

1. `pytest -m "is_trainium_test" tests/training/ --collect-only`
2. `EARLY_EXIT=1 pytest -m "is_trainium_test" --ignore tests/training/test_overfit.py tests/training/ -v`

### 9. Final smoke check

1. `optimum-cli neuron serve` with a small model

If time is limited, do not reorder the suites. Truncate from the end of this list so failures are observed in the same progression as CI.

---

## Risk: Neuron SDK 2.28 breaking changes

Neuron 2.28 introduces a transition to native PyTorch support (TorchNeuron) starting with PyTorch 2.10. For PyTorch 2.9, the existing torch-neuronx/XLA approach is still supported. Since we're targeting torch-neuronx 2.9.0, we stay on the XLA path — no architectural migration needed yet.

However, bumping neuronx_distributed from 0.15 to 0.17 and neuronx-cc from 2.21 to 2.23 may introduce breaking changes in the inference/compilation path that are **outside the vLLM integration layer**. These would surface in `tests/decoder/` and `tests/exporters/` rather than `tests/vllm/`.

---

## References

- [vllm-ascend](https://github.com/vllm-project/vllm-ascend) — closest analog platform plugin
- [vLLM transformers v5 issue](https://github.com/vllm-project/vllm/issues/30466)
- [Neuron SDK releases](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/torch/torch-neuronx/index.html)
- [Neuron PyTorch 2.9 intro](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/appnotes/torch-neuronx/introducing-pytorch-2-9.html)
