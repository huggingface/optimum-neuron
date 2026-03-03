# Chunked Prefill Benchmark Results

Benchmark comparing standard context encoding vs chunked prefill on Llama 3.1 8B Instruct.

## Setup

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-3.1-8B-Instruct` |
| Instance | `inf2.48xlarge` (12 Inferentia2 devices, 24 NeuronCores) |
| Batch size | 32 |
| Sequence length | 4096 |
| Tensor parallel | 8 |
| Chunk size | 1024 (for chunked configs) |
| Benchmark tool | [guidellm](https://github.com/neuralmagic/guidellm) |
| Concurrent users | 32 |

## Results

| Config | Prefill | Sampling | Sync ITL (ms) | Throughput ITL (ms) | Throughput (tok/s) | CSV |
|--------|---------|----------|---------------|--------------------|--------------------|-----|
| A | Standard CE | On-device (ODS) | 24.5 | 59.1 | 426.2 | `std-ods.csv` |
| B | Standard CE | CPU | 48.2 | 123.8 | 177.0 | `std-cpu-sampling.csv` |
| C | Chunked | CPU | 47.3 | 119.2 | 189.9 | `chunked-cpu-sampling.csv` |
| D | Chunked | Hybrid ODS | 24.6 | 55.9 | 440.4 | `hybrid-ods.csv` |

## Analysis

**On-device sampling (ODS) dominates performance.** Comparing configs A and B, ODS reduces
synchronous ITL from 48.2ms to 24.5ms — a ~24ms/step saving that comes from avoiding the
device-to-host round trip for token sampling. This translates to a 2.4x throughput improvement
(426.2 vs 177.0 tok/s).

**Chunked prefill is faster than standard CE at equal sampling.** Comparing configs B and C
(both CPU sampling), chunked prefill achieves +7.3% throughput (189.9 vs 177.0 tok/s) with
lower ITL (119.2 vs 123.8ms). The smaller prefill graph (1024 tokens vs 4096) compiles faster,
uses less device memory, and processes prompts more efficiently.

**Hybrid ODS is the optimal configuration.** Config D uses on-device sampling for the decode
graph and CPU sampling for the prefill graph. This combines the ODS speed advantage with
chunked prefill's efficiency, exceeding the production baseline (A) by +3.3% throughput
(440.4 vs 426.2 tok/s) and lower ITL (55.9 vs 59.1ms). This is the default configuration
when `sequence_length > 1024`.

## CSV Columns

All CSVs share the same schema with two rows: `synchronous` (single-user latency) and
`throughput` (32 concurrent users):

- `model_id` — HuggingFace model identifier
- `Date` — benchmark timestamp
- `Input type` — `synchronous` or `throughput`
- `Requests per Second` — sustained request rate
- `Request Latency (s)` — end-to-end request latency
- `Time-to-first-token (ms)` — time to first generated token
- `Inter Token Latency (ms)` — average time between consecutive tokens
- `Output Token Throughput (t/s)` — total tokens generated per second
