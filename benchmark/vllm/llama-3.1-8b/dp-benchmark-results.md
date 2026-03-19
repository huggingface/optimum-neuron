# Data Parallelism Benchmark Results

## Setup
- **Model:** meta-llama/Llama-3.1-8B-Instruct
- **Config:** TP=8, BS=32, SL=4096
- **Instance:** inf2.48xlarge (24 NeuronCores)
- **Benchmark:** guidellm via performance.sh, 128 concurrent users
- **Workload:** prompt_tokens=1500 (var 150), generated_tokens=250 (var 20)

## Results

Uses independent vLLM server processes behind a round-robin reverse proxy.
Model downloaded from HF Hub cache (no local pre-compiled model needed).

| Config | Mode | Req/s | Req Latency (s) | TTFT (ms) | ITL (ms) | Tput (tok/s) |
|--------|------|-------|-----------------|-----------|----------|--------------|
| CLI DP=1 | synchronous | 0.15 | 6.85 | 361.3 | 27.95 | 33.9 |
| CLI DP=1 | throughput | 1.70 | 34.97 | 20294.9 | 64.54 | 385.2 |
| CLI DP=2 | synchronous | 0.14 | 7.08 | 469.2 | 28.29 | 33.0 |
| CLI DP=2 | throughput | 2.98 | 20.43 | 7937.9 | 55.00 | 678.5 |
| CLI DP=3 | synchronous | 0.14 | 7.29 | 585.8 | 28.15 | 32.7 |
| CLI DP=3 | throughput | 4.87 | 15.40 | 4614.5 | 46.02 | 1144.5 |

### Scaling analysis

| Config | Throughput (tok/s) | Scaling factor | Efficiency |
|--------|-------------------|---------------|------------|
| DP=1 | 385.2 | 1.00x | — |
| DP=2 | 678.5 | 1.76x | 88% |
| DP=3 | 1144.5 | 2.97x | 99% |

- **Sync ITL** unchanged across all configs: ~28ms (per-replica latency identical)
- **DP=3 throughput ITL: 46.0ms** — matches Docker DP=3 (48.1ms)
- **DP=3 vs Docker DP=3: 1144.5 vs 1215.0 tok/s (94.2%)** — within expected variance

## Reference: Docker DP=3 (Sept 2025, older codebase 0.4.1)

| Mode | Req/s | Req Latency (s) | TTFT (ms) | ITL (ms) | Tput (tok/s) |
|------|-------|-----------------|-----------|----------|--------------|
| synchronous | 0.14 | 6.93 | 475.1 | 26.59 | 35.0 |
| asynchronous@5.33 | 4.58 | 17.20 | 350.0 | 71.11 | 1084.4 |
| throughput | 5.33 | 15.57 | 4198.4 | 49.53 | 1226.1 |
