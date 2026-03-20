# Qwen3-Embedding-8B Benchmark on inf2.48xlarge

## Goal
Benchmark the Qwen3-Embedding-8B model served via vLLM on an inf2.48xlarge instance
(12 Inferentia2 devices, 24 NeuronCores) with data parallelism scaling.

## Model
- **Model ID:** `Qwen/Qwen3-Embedding-8B`
- **Architecture:** qwen3, 36 layers, hidden_size=4096, 32 attn heads, 8 KV heads
- **Output:** 4096-dim embeddings
- **Tensor parallelism:** TP=8 (shards across 4 devices)

## Exported Configurations

### SL=8192 (long context)
```bash
CUSTOM_CACHE_REPO=none optimum-cli export neuron \
  --model Qwen/Qwen3-Embedding-8B \
  --task feature-extraction \
  --batch_size 8 --sequence_length 8192 \
  --tensor_parallel_size 8 --torch_dtype bfloat16 \
  /home/ubuntu/models/qwen3-embedding-8b-tp8-bs8-sl8192/
```
- BS=32 OOM'd at SL=8192 (exceeds 16GB per-core HBM limit)
- BS=8 compiled successfully (cache hit from 0.6B model tests)
- Hub cache for this model is **poisoned** — must set `CUSTOM_CACHE_REPO=none`

### SL=1024 (short context)
```bash
CUSTOM_CACHE_REPO=none optimum-cli export neuron \
  --model Qwen/Qwen3-Embedding-8B \
  --task feature-extraction \
  --batch_size 32 --sequence_length 1024 \
  --tensor_parallel_size 8 --torch_dtype bfloat16 \
  /home/ubuntu/models/qwen3-embedding-8b-tp8-bs32-sl1024/
```
- BS=32 fits at SL=1024
- Compilation took ~106s (no cache hit, new HLO)

## Serving

```bash
# DP=1 (8 cores)
optimum-cli neuron serve -m /home/ubuntu/models/qwen3-embedding-8b-tp8-bs32-sl1024 \
  --task embed --data-parallel-size 1

# DP=3 (24 cores, all used)
optimum-cli neuron serve -m /home/ubuntu/models/qwen3-embedding-8b-tp8-bs32-sl1024 \
  --task embed --data-parallel-size 3

# Auto-detect (omit DP and TP → auto-selects DP=3)
optimum-cli neuron serve -m /home/ubuntu/models/qwen3-embedding-8b-tp8-bs32-sl1024 \
  --task embed
```

### Notes
- serve.sh updated to pass `TASK` env var
- serve.sh updated to always pass `DATA_PARALLEL_SIZE` (not just when >1)
- DP=3 had intermittent process group errors on first attempt; retry usually works

## Benchmark Tool
Custom `embedding_perf.py` (guidellm only supports text generation).
Sends concurrent requests to `/v1/embeddings`, measures throughput and latency.

```bash
cd benchmark/vllm/qwen3-embedding-8b
python3 ../embedding_perf.py \
  --target http://localhost:8080/v1 \
  --model "/home/ubuntu/models/qwen3-embedding-8b-tp8-bs32-sl1024" \
  --concurrent 32 --total 500 --prompt-tokens 512 \
  --output result.json
```

## Results

### SL=8192, BS=8, ~1500 tok prompts

| Config | Concurrent | req/s | tok/s | p50 (ms) | p90 (ms) | mean (ms) |
|--------|-----------|-------|-------|----------|----------|-----------|
| DP=1   | 8         | 0.65  | 985   | 11,986   | 12,000   | 12,007    |
| DP=1   | 32        | 0.81  | 1,230 | 17,997   | 24,006   | 36,563    |
| DP=3   | 8         | 0.47  | 711   | 11,998   | 29,807   | 16,778    |
| DP=3   | 32        | 0.91  | 1,365 | 17,844   | 83,668   | 33,835    |

### SL=1024, BS=32, ~512 tok prompts

| Config | Concurrent | req/s | tok/s | p50 (ms) | p90 (ms) | mean (ms) |
|--------|-----------|-------|-------|----------|----------|-----------|
| DP=1   | 32        | 1.40  | 721   | 7,829    | 15,869   | 21,989    |
| DP=3   | 32        | 1.85  | 949   | 7,228    | 43,216   | 16,854    |

## Key Observations

1. **Per-batch latency:** ~12s (SL=8192), ~7-8s (SL=1024) — dominated by the 8B model's forward pass through 36 layers
2. **DP scaling is modest:** DP=3 vs DP=1 gives ~1.1-1.3x throughput, far from the ideal 3x. The bottleneck is the compute-heavy forward pass, not the scheduling.
3. **SL=1024 is ~1.7x faster than SL=8192** per request, but less than the expected 8x because self-attention is quadratic in SL but linear layers dominate at lower SL.
4. **Hub cache poisoned:** `Qwen/Qwen3-Embedding-8B` at SL=8192 has a failed compilation entry in `aws-neuron/optimum-neuron-cache`. Must bypass with `CUSTOM_CACHE_REPO=none`.

## TODO (to continue later)
- [ ] Test with higher concurrency (64, 128) to see if throughput saturates differently
- [ ] Try TP=4 (allows DP=6) — more replicas might help if per-replica compute is the bottleneck
- [ ] Investigate why DP scaling is sublinear — check if round-robin proxy or vLLM scheduler is batching efficiently
- [ ] Test with Qwen3-Embedding-0.6B or 4B for comparison (much faster per-request)
- [ ] Consider TP=2 for smaller models (0.6B) to maximize DP replicas (DP=12)
