# vLLM benchmark report — Llama-3.1-8B-Instruct on inf2

Date: 2026-08-24

Benchmark of `meta-llama/Llama-3.1-8B-Instruct` served with the current vLLM
integration (vLLM 0.16) on an inf2.24xlarge, plus a smoke test of the
`optimum-neuron-vllm` Docker image. This is the first data point for this
hardware/configuration combination; earlier results in this directory were
produced on trn2 or inf2.48xlarge with larger tensor parallelism and batch
sizes.

## Environment

- Instance: inf2.24xlarge — 2 Inferentia2 devices / 4 NeuronCores (64 GB device memory)
- Configuration: TP=2, batch size 4, sequence length 4096 (all 4 NeuronCores)
- optimum-neuron 0.4.7.dev0
- vllm 0.16.0
- torch-neuronx 2.9.0.2.15.32035+de43f57c
- neuronx-cc 2.26.6360.0+6f180f47
- neuronx-distributed 0.19.28492+435aae2b
- aws-neuronx-runtime-lib 2.33.10.0 / aws-neuronx-collectives 2.33.10.0 / aws-neuronx-tools 2.31.13.0
- guidellm 0.1.0, lm-eval 0.4.12

## Performance results (guidellm)

Command: `./performance.sh meta-llama/Llama-3.1-8B-Instruct 4`
(concurrency 4 to match the compiled batch size), emulated data
1500±150 prompt tokens / 250±20 generated tokens.

| Input type | Requests/s | Request latency (s) | TTFT (ms) | ITL (ms) | Output tok/s |
|------------|-----------|--------------------|-----------|----------|--------------|
| synchronous | 0.075 | 13.40 | 1057.7 | 50.67 | 18.17 |
| throughput (4 users) | 0.188 | 15.30 | 1315.2 | 59.46 | 44.21 |

CSV: [single-instance/llama-3.1-8b-inf2/vllm-results.csv](single-instance/llama-3.1-8b-inf2/vllm-results.csv)

## Accuracy results (lm_eval)

Command: `./accuracy.sh meta-llama/Llama-3.1-8B-Instruct 4 gsm8k`

| Task | Filter | Metric | Value | Stderr |
|------|--------|--------|-------|--------|
| gsm8k | flexible-extract | exact_match | 0.7756 | ± 0.0115 |
| gsm8k | strict-match | exact_match | 0.7111 | ± 0.0125 |

In the expected range for this model. Details:
[single-instance/llama-3.1-8b-inf2/accuracy-results.md](single-instance/llama-3.1-8b-inf2/accuracy-results.md)

## Docker image

`make optimum-neuron-vllm` (Ubuntu 24.04 base, uv-managed Python 3.12 venv):

- Image size: 8.36 GB; build time ~2–3 min on this instance.
- Smoke test: container started with `SM_ON_MODEL=meta-llama/Llama-3.1-8B-Instruct`,
  `SM_ON_TENSOR_PARALLEL_SIZE=2`, `SM_ON_BATCH_SIZE=4`, `SM_ON_SEQUENCE_LENGTH=4096`
  and `/dev/neuron0` passed through (TP=2 uses one device's two NeuronCores);
  `/v1/models` listed the model and a greedy completion request returned sane
  text.

## Operational findings

These came up while running the benchmark and apply to any deployment of the
current stack:

1. **Host runtime must match the compiler.** Graphs compiled by neuronx-cc
   2.26 fail to load on aws-neuronx-runtime-lib 2.30 with
   `NRT_UNSUPPORTED_NEFF_VERSION` ("compiled by a newer version of Neuron
   compiler"). Upgrade the host packages to the versions pinned in
   `docker/vllm/Dockerfile` (runtime-lib/collectives 2.33.10, tools 2.31.13)
   before serving.
2. **Docker build: heredoc entrypoint broke on the legacy builder (fixed).**
   The `RUN cat <<'EOF' > entrypoint.sh` heredoc in `docker/vllm/Dockerfile`
   silently produced an *empty* `entrypoint.sh` with the legacy builder (no
   buildx), so the container died with `exec format error`. Fixed by checking
   the script into `docker/vllm/entrypoint.sh` and `COPY`ing it (works on both
   the legacy builder and BuildKit). The Makefile also needs `gawk` installed
   to resolve the image tag version.
3. **First serve recompiles.** The Hub compilation cache is keyed by compiler
   version, so cached entries from older stacks do not hit; the first serve of
   this configuration compiled for ~2.5 minutes (8B, TP=2) before the server
   came up. Subsequent serves reuse the cache.

## Comparison with existing results

No same-hardware baseline with the previous stack exists, so **no causal
performance claims** are made about the stack update. For orientation only:

- `single-instance/llama-3.1-8b-trn2` (2025-10-16, trn2, TP=4 / BS=32, older
  stack): synchronous ITL 22.1 ms, throughput 578.8 tok/s at 32 users. trn2
  chips, 4× the batch size and 8× the concurrency — not comparable to this run.
- `chunked-prefill/*` (inf2.48xlarge, TP=8 / BS=32, 32 users): 426–440 tok/s.
  Same chip family but 24 NeuronCores vs 4 and 8× the batch; per-core the
  larger-batch configurations amortize better (≈17.8 tok/s/core vs ≈11.1
  tok/s/core here), which reflects batching economics, not stack versions.

## Stack changes relevant to benchmarking

Relative to the previous vLLM 0.11-based release:

- vLLM 0.16.0: plugin ported to the new module layout; `--task` renamed to
  `--runner` in `optimum-cli neuron serve`; async scheduling is force-disabled
  on Neuron.
- EngineCore now starts with the spawn method (plus a picklable ModelConfig
  patch), fixing the Neuron runtime fork deadlock.
- Compiler/runtime bumps: neuronx-cc 2.26, torch-neuronx 2.9,
  neuronx-distributed 0.19, runtime-lib 2.33; Python ≥ 3.11 required.
- Numerics fixes: KV-cache reorder, gemma3 fp32 softmax, greedy tie-break.

When re-running the trn2 and data-parallel configurations, expect: host
runtime upgrades (finding 1), an image rebuild (the entrypoint fix in finding
2 is required for legacy-builder hosts), and a one-time recompilation per
configuration (finding 3). Async scheduling being disabled may shift
throughput numbers relative to previous runs — treat cross-stack comparisons
as qualitative.

## Runbook

### Reproduce this run (inf2.24xlarge)

```shell
# 0. Upgrade host runtime packages (see finding 1), then:
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[neuronx,vllm,tests]"
pip install guidellm==0.1.0 lm_eval[api]
export HF_TOKEN=$(cat ~/.cache/huggingface/token)

# 1. Serve (compiles once on first run, ~2.5 min)
benchmark/vllm/single-instance/serve.sh benchmark/vllm/single-instance/llama-3.1-8b-inf2 &

# 2. Performance + CSV
cd benchmark/vllm
./performance.sh meta-llama/Llama-3.1-8B-Instruct 4
python generate_csv.py --dir .

# 3. Accuracy (same server process)
./accuracy.sh meta-llama/Llama-3.1-8B-Instruct 4 gsm8k

# 4. Docker smoke test (stop the bare-metal server first)
make optimum-neuron-vllm   # requires gawk on the host to resolve the image tag
docker run -d --name vllm-smoke --device /dev/neuron0 \
  -e HF_TOKEN=$HF_TOKEN \
  -e SM_ON_MODEL=meta-llama/Llama-3.1-8B-Instruct \
  -e SM_ON_TENSOR_PARALLEL_SIZE=2 -e SM_ON_BATCH_SIZE=4 -e SM_ON_SEQUENCE_LENGTH=4096 \
  -p 8080:8080 optimum-neuron-vllm:latest
```

### Extend to the other configurations (needs suitable instances)

- **trn2 single-instance configs** (`qwen3-32B`, `qwen3-30B-A3B`, `qwen3-235B`,
  `llama4-Scout`, `llama4-Maverick`, `llama-3.1-8b-trn2`): same flow as above
  on a trn2 instance with enough NeuronCores for the config's TP; each config
  recompiles once on first serve.
- **data-parallel** (`llama3.1-8b` dp3/dp4, `llama3-70B-trn2`, `qwen3-30B-A3B`):
  rebuild `optimum-neuron-vllm:latest`, then
  `docker compose -f <config>/docker-compose*.yaml --env-file <config>/.env up`
  on an instance with the required devices; benchmark with `../performance.sh`.
- **chunked-prefill**: re-run on inf2.48xlarge at TP=8 / BS=32 following
  `chunked-prefill/README.md`; the four configurations differ only in prefill
  mode and sampling location.
