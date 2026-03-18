# Qwen3-30B-A3B-Instruct-2507 Benchmark Results

**Date:** 2026-03-18
**Instance:** inf2.48xlarge (12 Inferentia2 devices, 24 NeuronCores)
**Model:** Qwen/Qwen3-30B-A3B-Instruct-2507 (MoE, 30B total / 3B active)
**Config:** TP=8, SL=4096, bfloat16
**Workload:** ~1500 input tokens, ~250 output tokens (emulated)
**Tool:** guidellm

## Throughput mode (saturated)

| DP | BS | Users | TTFT (ms) | ITL (ms) | Throughput (tok/s) |
|----|-----|-------|-----------|----------|--------------------|
| 1  | 4   | 8     | 52,445    | 175      | 8.1                |
| 2  | 4   | 16    | 46,051    | 158      | 17.0               |
| 3  | 4   | 24    | 48,298    | 162      | 26.0               |
| 1  | 32  | 32    | 17,139    | 168      | 82.5               |
| 2  | 32  | 64    | 9,396     | 138      | 106.9              |
| 3  | 32  | 96    | 6,917     | 128      | 124.2              |
| 1  | 64  | 64    | 13,671    | 371      | failed             |

## Synchronous mode (single-user latency)

| DP | BS | TTFT (ms) | ITL (ms) | Latency (s) |
|----|-----|-----------|----------|-------------|
| 1  | 4   | 14,005    | 22       | 19.4        |
| 2  | 4   | 14,295    | 22       | 19.6        |
| 3  | 4   | 14,581    | 22       | 19.8        |
| 1  | 32  | 1,537     | 113      | 28.6        |
| 2  | 32  | 1,994     | 113      | 28.9        |
| 3  | 32  | 2,416     | 113      | 29.8        |

## Key findings

### BS=32 is the sweet spot

- **10x throughput** over BS=4 at DP1 (8.1 -> 82.5 tok/s)
- BS=64 exceeds device memory capacity — ITL degrades to 371ms and throughput requests time out
- Device memory utilization at BS=32: 74.7% (no headroom for BS=64)

### Data parallelism scales sub-linearly

At BS=32: 82.5 -> 106.9 -> 124.2 tok/s (1.0x -> 1.30x -> 1.51x for DP1 -> DP2 -> DP3).
Sub-linear scaling is expected: DP replicas share host CPU and memory bandwidth.

### ITL improves with more DP replicas

At BS=32 throughput mode: 168 -> 138 -> 128 ms (DP1 -> DP2 -> DP3).
More replicas means less queuing per replica, reducing effective inter-token latency.

### Latency vs throughput tradeoff

- BS=4 gives best single-user ITL (22ms) but poor throughput (8 tok/s)
- BS=32 sacrifices ITL (113ms) for much higher throughput (82-124 tok/s)
- TTFT is dramatically better at BS=32 in throughput mode (17s vs 52s) due to less queuing

### Optimal configuration

| Goal              | DP  | BS  | Expected throughput |
|-------------------|-----|-----|---------------------|
| Max throughput    | 3   | 32  | 124 tok/s           |
| Balanced          | 2   | 32  | 107 tok/s           |
| Low latency       | 1   | 4   | 8 tok/s (22ms ITL)  |

## Raw data

See [vllm-results.csv](vllm-results.csv) for the full dataset.
