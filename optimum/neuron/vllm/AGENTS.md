# Optimum Neuron vLLM Guide

This guide covers the Optimum Neuron vLLM backend. For project-wide workflows see [AGENTS.md](../../../AGENTS.md).

## Architecture Overview
- Runner: `OptimumNeuronModelRunner` manages execution and batch state.
- Batch management: `OptimumNeuronCachedBatch` maps requests to KV cache slots.
- Model wrapper: `OptimumNeuronModelForCausalLM` adapts Neuron models to vLLM.
- Only NxD backend with `continuous_batching: true` is supported.

Key files:
- [optimum/neuron/vllm/runner.py](runner.py)
- [optimum/neuron/vllm/model_loader.py](model_loader.py)

## Request Lifecycle
- Prompt (context encoding) and decode (token generation) are executed on different graphs.
- Finished requests are removed from the cached batch to free KV cache slots.

## Temperature Conversion
vLLM uses `temperature=0.0` for greedy decoding, but Neuron requires `top_k=1`:
- Implemented in [optimum/neuron/vllm/runner.py](runner.py).

## Model Export Detection
If the model isn’t pre-compiled, vLLM searches the hub cache for a compatible config:
- `select_hub_cached_entries()` in [optimum/neuron/vllm/model_loader.py](model_loader.py)
- Falls back to on-the-fly export when no match is found.

## CLI
- Launch server via `optimum-cli neuron serve` (see [optimum/commands/neuron/serve.py](../../commands/neuron/serve.py)).

## Data-Parallel Serving
When `--data-parallel-size N` is passed (N > 1), `optimum-cli neuron serve` spawns N independent vLLM servers behind an aiohttp round-robin reverse proxy.

Key files:
- [optimum/neuron/vllm/server_manager.py](server_manager.py) — spawns and manages vLLM server subprocesses with Neuron core pinning.
- [optimum/neuron/vllm/reverse_proxy.py](reverse_proxy.py) — round-robin proxy with aggregated `/health`.

### Process Lifecycle (important)
vLLM servers are spawned **without** `start_new_session=True` — they inherit the parent's process group. On shutdown, `optimum-cli` sends SIGTERM to each server via `proc.terminate()`, and vLLM handles its own EngineCore child cleanup gracefully.

**Do NOT use `start_new_session=True` on vLLM server subprocesses.** vLLM already cleans up its own EngineCore workers on SIGTERM. Isolating servers in separate process groups makes them unreachable by the parent's signal handlers and test fixture teardown, causing Neuron core leaks that poison subsequent tests.
