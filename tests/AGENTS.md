# Test Infrastructure Agent Guide

Read this before working on any test code under `tests/`.

## Directory Structure

```
tests/
  conftest.py                  # Root conftest — registers fixture plugins via pytest_plugins
  fixtures/llm/
    export_models.py           # Model export fixtures + CLI (see detailed docs in the module docstring)
    vllm_service.py            # vLLM OpenAI-compatible service launcher fixture
    vllm_docker_service.py     # vLLM Docker service launcher fixture
  decoder/                     # LLM decoder tests (generation, export, pipelines, modules, etc.)
    conftest.py                # Reorders @subprocess_test tests to run before session fixtures
  vllm/                        # vLLM integration tests
    engine/                    # Direct engine API tests
    service/                   # OpenAI-compatible serving tests
    docker/                    # Docker container tests
```

## Fixture Registration

Fixtures in `tests/fixtures/llm/` are registered as pytest plugins in `tests/conftest.py`:

```python
pytest_plugins = [
    "fixtures.llm.vllm_docker_service",
    "fixtures.llm.vllm_service",
    "fixtures.llm.export_models",
]
```

This makes `neuron_llm_config`, `any_generate_model`, and `speculation` available to all tests.

## The Export Models Fixture System

Full documentation is in the `tests/fixtures/llm/export_models.py` module docstring. Key points:

### Model Configurations

Each model is compiled for two shapes: `(batch_size=4, seq_len=1024)` and `(batch_size=1, seq_len=8192)`.
Config names follow the pattern `<model>-<batch>x<seq>`, e.g. `llama-4x1024`.

### Using Fixtures in Tests

**`neuron_llm_config`** — for tests that need a specific model config:
```python
@pytest.mark.parametrize("neuron_llm_config", ["llama-4x1024"], indirect=True)
def test_something(neuron_llm_config: dict[str, Any]):
    model = NeuronModelForCausalLM.from_pretrained(neuron_llm_config["neuron_model_path"])
```

**`any_generate_model`** — for tests that should run across all generation models:
```python
def test_greedy_expectations(any_generate_model):
    model_path = any_generate_model["neuron_model_path"]
    config_name = any_generate_model["name"]  # e.g. "llama-4x1024"
```

Both yield a dict with keys: `name`, `model_id`, `task`, `export_kwargs`, `neuron_model_id`, `neuron_model_path`.

### Hub Caching

Compiled models are pushed to private HF Hub repos. The repo name encodes all invalidation keys:
`<org>/optimum-neuron-testing-<version>-<sdk>-<instance>-<code_hash>-<config_name>`.

The code hash changes when `pyproject.toml` or anything under `optimum/neuron/models/inference/` changes.
Old repos must be pruned manually: `python tools/prune_test_models.py`.

## Always Pre-Export Models Before Running Tests

**CI always runs `python tests/fixtures/llm/export_models.py` as a separate step before any pytest invocation** (see `.github/workflows/test_inf2_llm.yml` and `test_inf2_vllm.yml`). You must do the same locally:

```bash
# Export all models (or use a pattern like 'llama*')
python tests/fixtures/llm/export_models.py

# Then run tests
pytest -sv tests/decoder/test_decoder_generation.py
```

If you skip the pre-export step, fixtures will auto-export on first use. This causes:
- **Long hangs**: compilation takes 10-30+ minutes per model, making it hard to tell if a test is stuck or just compiling.
- **NeuronCore conflicts**: the compilation process may conflict with subprocess-isolated tests that also need device access.

### Expected Test Durations

Based on CI logs (inf2.8xlarge, models pre-exported), entire test groups complete within:

| Test group | Duration |
|---|---|
| LLM utils / hub / CLI / embedding | < 1 min each |
| LLM export tests | ~4 min |
| LLM generation tests | ~7 min |
| LLM pipeline tests | ~5 min |
| LLM module tests (NKI kernels) | ~19 min |
| LLM cache tests | ~5 min |
| vLLM engine generation | ~20 min |
| vLLM service tests | ~15 min |

An individual test should complete within **2 minutes**. If a test hangs longer than that, the most likely cause is a missing pre-export triggering compilation inside the fixture. Pre-export, then re-run.

## Never Wipe the Neuron Compiler Cache

The Neuron compiler cache is **content-addressed**: each compiled NEFF is keyed by the SHA hash of the HLO graph that produced it. The hash space is large enough to make collisions practically impossible.

**There is no such thing as a "stale compiler cache entry."** If the HLO graph changes (because you changed model code), the hash changes, and a new entry is created. The old entry is simply never matched again — it does no harm.

Wiping the compiler cache (e.g. `rm -rf /var/tmp/neuron-compile-cache`) only forces expensive recompilation with zero benefit. **Never suggest or perform cache deletion as a debugging step.**

## Subprocess Test Ordering

`tests/decoder/conftest.py` contains a `pytest_collection_modifyitems` hook that moves `@subprocess_test`-decorated tests to run **before** all other tests. This prevents session-scoped fixtures (which load models onto NeuronCores) from blocking subprocess tests that need device access in a child process.
