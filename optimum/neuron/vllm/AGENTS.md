# Optimum Neuron vLLM Guide

This guide covers the Optimum Neuron vLLM backend. For project-wide workflows see [AGENTS.md](../../AGENTS.md).

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
