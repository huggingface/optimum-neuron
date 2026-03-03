# Llama 3.1 8B Memory Capacity Analysis

**Hardware:** inf2.48xlarge | **TP:** 8 | **Dtype:** bf16 | **Mode:** Hybrid ODS + Chunked Prefill

## Hardware Specifications

| Parameter | Value |
|---|---|
| Instance | inf2.48xlarge |
| Inferentia2 devices | 12 (24 NeuronCores) |
| HBM per NeuronCore | 16 GB (16,384 MB) |
| TP degree | 8 NeuronCores |
| Total HBM (TP group) | 128 GB |

## Model Architecture (Llama 3.1 8B)

| Parameter | Value |
|---|---|
| Hidden size | 4,096 |
| Intermediate size | 14,336 |
| Num layers | 32 |
| Num attention heads | 32 (4 per TP rank) |
| Num KV heads | 8 (1 per TP rank) |
| Head dim | 128 |
| Vocab size | 128,256 |
| Max position embeddings | 131,072 (128K) |

## KV Cache Formula

```
KV_per_rank_MB = batch_size × sequence_length / 64
```

Derived from:

```
BS × (num_kv_heads / TP) × SL × head_dim × num_layers × 2(K+V) × 2(bf16) bytes
= BS × 1 × SL × 128 × 32 × 4
= BS × SL × 16,384 bytes
= BS × SL / 64 MB
```

Verification: BS=32, SL=4,096 → 32 × 4096 / 64 = 2,048 MB per rank.
Matches logged value: `Allocated 8*2048.00 = 16384.00 MB for KV cache`.

## Measured HBM Usage (sysfs peak, BS=32, SL=4096)

| Category | Rank 0 (MB) | Rank 1-7 (MB) | Notes |
|---|---|---|---|
| tensors (weights + KV) | 5,815 | 3,996 | Weights + KV cache |
| — model weights | 3,767 | 1,948 | Rank 0 has unsharded embed/lm_head |
| — KV cache | 2,048 | 2,048 | BS × SL / 64 |
| model_shared_scratchpad | 1,024 | 1,024 | Activation memory (reused across layers) |
| collectives | 472 | 126 | AllReduce/AllGather buffers |
| dma_rings | 392 | 220 | DMA transfer ring buffers |
| model_code (NEFF) | 319 | 228 | Compiled graph binaries |
| other | 8 | 7 | constants + runtime + notifications |
| **Total** | **8,029** | **5,600** | |
| **Headroom** | **8,355** | **10,784** | Available out of 16,384 MB |

Rank 0 is the **binding constraint** (~1.8 GB extra for unsharded embedding/lm_head).

### Fixed Overhead (non-KV-cache)

| | Rank 0 | Rank 1-7 |
|---|---|---|
| Fixed overhead | 5,981 MB | 3,552 MB |
| Available for KV | 10,403 MB | 12,832 MB |

**Theoretical max BS × SL = 10,403 × 64 = 665,792** (rank 0 limited).

## Attention Scratchpad Scaling

The critical difference between standard context encoding (CE) and chunked prefill is
the attention score matrix size during prefill:

### Standard CE: O(SL²)

```
attn_scratchpad = num_heads_per_rank × SL² × 2 bytes = 4 × SL² × 2 bytes
```

| SL | Attention Scratchpad |
|---|---|
| 4K | 128 MB |
| 8K | 512 MB |
| 16K | 2,048 MB (2 GB) |
| 32K | 8,192 MB (8 GB) |
| 64K | 32,768 MB — **exceeds 16 GB per core** |

### Chunked Prefill (chunk_size=1024): O(SL)

```
attn_scratchpad = num_heads_per_rank × chunk_size × SL × 2 bytes = 4 × 1024 × SL × 2 bytes
```

| SL | Attention Scratchpad |
|---|---|
| 4K | 32 MB |
| 8K | 64 MB |
| 16K | 128 MB |
| 32K | 256 MB |
| 64K | 512 MB |
| 128K | 1,024 MB |

## Max BS × SL Configurations

### Chunked Prefill (chunk_size=1024)

| BS | Max SL (exact) | Max SL (pow2) | KV/rank (MB) | Attn SP delta (MB) | Est total (MB) | HBM % |
|---|---|---|---|---|---|---|
| 1 | 131,072 | 128K | 2,048 | 992 | 9,021 | 55% |
| 2 | 131,072 | 128K | 4,096 | 992 | 11,069 | 68% |
| 4 | 131,072 | 128K | 8,192 | 992 | 15,165 | 93% |
| 8 | 78,278 | 64K | 8,192 | 480 | 14,653 | 89% |
| 16 | 40,439 | 32K | 8,192 | 224 | 14,397 | 88% |
| 32 | 20,544 | 16K | 8,192 | 96 | 14,269 | 87% |
| 64 | 10,353 | 8K | 8,192 | 32 | 14,205 | 87% |
| 128 | 5,197 | 4K | 8,192 | 0 | 14,173 | 87% |
| 256 | 2,600 | 2K | 8,192 | 0 | 14,173 | 87% |

### Standard Context Encoding

| BS | Max SL (exact) | Max SL (pow2) | KV/rank (MB) | CE Attn delta (MB) | Est total (MB) | HBM % |
|---|---|---|---|---|---|---|
| 1 | 32,768 | 32K | 512 | 8,064 | 14,557 | 89% |
| 2 | 32,768 | 32K | 1,024 | 8,064 | 15,069 | 92% |
| 4 | 52,527 | 32K | 2,048 | 8,064 | 16,093 | 98% |
| 8 | 30,046 | 16K | 2,048 | 1,920 | 9,949 | 61% |
| 16 | 29,198 | 16K | 4,096 | 1,920 | 11,997 | 73% |
| 32 | 16,771 | 16K | 8,192 | 1,920 | 16,093 | 98% |
| 64 | 9,798 | 8K | 8,192 | 384 | 14,557 | 89% |
| 128 | 5,162 | 4K | 8,192 | 0 | 14,173 | 87% |
| 256 | 2,600 | 2K | 8,192 | 0 | 14,173 | 87% |

### Chunked vs Standard — Max Batch Size at Key Sequence Lengths

| SL | Chunked max BS | Standard max BS | CP advantage |
|---|---|---|---|
| 2K | 256 | 256 | 1x |
| 4K | 128 | 128 | 1x |
| 8K | 64 | 64 | 1x |
| 16K | 32 | 32 | 1x |
| **32K** | **16** | **4** | **4x** |
| **64K** | **8** | **0 (OOM)** | **Only chunked** |
| **128K** | **4** | **0 (OOM)** | **Only chunked** |

## Recommended Configurations

| Use Case | BS | SL | KV/rank | HBM % | Notes |
|---|---|---|---|---|---|
| High throughput (chatbot) | 32 | 4K | 2,048 MB | 49% | Max concurrent users, moderate context |
| High throughput (extended) | 16 | 8K | 2,048 MB | 49% | Good concurrency + longer context |
| Balanced | 8 | 16K | 2,048 MB | 50% | Medium concurrency, 16K context |
| Long context (RAG) | 4 | 32K | 2,048 MB | 50% | Document QA workloads |
| Very long context | 2 | 64K | 2,048 MB | 52% | Summarization, book-length inputs |
| Max context | 1 | 128K | 2,048 MB | 55% | Single-user full 128K context |
| Max throughput push | 64 | 8K | 8,192 MB | 87% | High concurrency, aggressive |
| Aggressive balanced | 32 | 16K | 8,192 MB | 87% | High BS + long context |

## Key Takeaways

1. **KV cache is the dominant variable cost.** At 2 bytes per element with GQA (1 KV head per rank),
   the formula simplifies to `BS × SL / 64 MB` per rank.

2. **Rank 0 is the binding constraint** due to ~1.8 GB of extra unsharded weights (embedding + lm_head),
   leaving 10.4 GB for KV cache vs 12.8 GB on other ranks.

3. **Chunked prefill is a prerequisite for long-context deployment.** Standard CE physically cannot
   handle SL > 32K on a 16 GB NeuronCore due to the O(SL²) attention score matrix. Chunked prefill
   reduces this to O(SL), enabling the model's full 128K context window with only 1 GB of scratchpad.

4. **At SL ≤ 16K, chunked and standard CE support the same max batch size.** The quadratic
   scratchpad overhead is small enough to fit. The chunked prefill advantage emerges at SL ≥ 32K.

5. **Conservative configs (BS × SL ≈ 128K) use only ~50% of HBM.** There is significant room to
   increase either BS or SL. Aggressive configs (BS × SL ≈ 512K) push to ~87% utilization.

## Methodology

- Memory measurements from `/sys/devices/virtual/neuron_device/*/neuron_core*/stats/memory_usage/device_mem/` (peak values)
- Anchor configuration: BS=32, SL=4096, TP=8, hybrid ODS, Llama 3.1 8B Instruct (bf16)
- Fixed overhead estimated as measured total minus KV cache at anchor point
- Scratchpad growth modeled as attention score matrix delta from anchor point
- Max SL values solved iteratively to account for scratchpad growth with SL
