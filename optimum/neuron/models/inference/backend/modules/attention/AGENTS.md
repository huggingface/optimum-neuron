# Attention Module Guide

This directory contains the attention implementation for NxD inference models. For
broader context see [optimum/neuron/models/inference/AGENTS.md](../../../AGENTS.md).

## Key files

- `attention_base.py` — `NeuronAttentionBase`: base class for all decoder attention
  layers. Handles TP-aware QKV/O projections, RoPE, flash attention dispatch, and
  KV cache management.
- `flash_attention_nki.py` — Custom NKI kernel (`flash_fwd_large_d`) for models
  with `head_dim > 128` (e.g. Gemma3-27B with `head_dim=256`).
- `gqa.py` — Grouped Query Attention sharding strategies (`GroupQueryAttention_QKV`,
  `GroupQueryAttention_O`).
- `utils.py` — Shared attention helpers (RoPE application, `repeat_kv`, manual
  softmax for token generation).

## Available Attention Kernels

Three attention kernel implementations are available. Only two are currently wired
into the dispatch. See also the [NKI library](https://github.com/aws-neuron/nki-library)
for additional kernel examples and patterns.

### 1. `attention_isa_kernel` (BIR/ISA-level, NKI-wrapped) — ACTIVE

- **Source**: `neuronxcc.nki._private_kernels.attention.attention_isa_kernel`
- **Nature**: Hand-written at ISA level (Neuron assembly), wrapped in NKI.
  "ISA" = Instruction Set Architecture — this is assembly-level code, not NKI.
- **I/O layout**: `(B*H, d, seq)` — batch and heads merged into first dimension
- **Features**: causal mask, sliding_window, SPMD grid dispatch, attention sinks
- **Constraints**: `head_dim ≤ 128` (uses `par_dim(head_dim)` for QK matmul). Infers
  `head_dim` by comparing dimensions — requires `seq > head_dim`.
- **Performance**: Optimized for large seq_len. On LNC1, slower than compiler-native
  below ~4096.
- **Used by**: `UNSHARDED_KERNEL` and `SHARDED_KERNEL` strategies

### 2. `flash_fwd` (pure NKI, neuronxcc) — NOT WIRED IN

- **Source**: `neuronxcc.nki.kernels.attention.flash_fwd`
- **Nature**: Pure NKI implementation, higher-level than ISA
- **I/O layout**: `(bs, n_heads, d, seq)` — batch and heads separate
- **Features**: `causal_mask`, `logit_bias`, configurable
  `FlashConfig(seq_tile_size, training, ...)`
- **Constraints**: `seq_len % seq_tile_size == 0`. Default `seq_tile_size=2048`,
  configurable to 1024/512.
- **Missing**: No native `sliding_window` support
- **Not currently used** in optimum-neuron — potential alternative for small seq_len
  without sliding window

### 3. `flash_fwd_large_d` (custom NKI, optimum-neuron) — ACTIVE

- **Source**: `flash_attention_nki.py` in this directory
- **Nature**: Custom NKI with d-tiling for `head_dim > 128`
- **I/O layout**: `(B*H, d, seq)` — same as ISA kernel
- **Features**: causal mask, sliding_window (2-level tile filtering), online softmax
- **Constraints**: `head_dim` multiple of 128, ≤ 512. `seq_len` multiple of
  `LARGE_TILE_SZ`.
- **Used by**: `LARGE_D_UNSHARDED_KERNEL` and `LARGE_D_SHARDED_KERNEL` strategies

### SPMD grid requirements

`flash_fwd` requires a 2D SPMD grid when called:
```python
kernel[batch, heads](q, k, v, ...)  # NOT kernel(q, k, v, ...)
```
The `attention_isa_kernel` uses a different convention (no grid for unsharded,
1D `nc()` grid for sharded).

### Sliding window support matrix

| Kernel | sliding_window |
|---|---|
| `attention_isa_kernel` | Yes (native parameter) |
| `flash_fwd` | No (only `logit_bias`, a full `[B,H,S,S]` tensor) |
| `flash_fwd_large_d` | Yes (native parameter, 2-level tile filtering) |

## Flash Attention Dispatch

`NeuronAttentionBase.get_flash_attention_strategy()` selects one of five strategies
at prefill time based on `head_dim`, `seq_len`, and `logical_nc_config` (LNC1 vs LNC2):

| Strategy | Kernel | When |
|---|---|---|
| `NONE` | Compiler-native matmul/softmax | seq_len too small or custom `qk_scale` |
| `UNSHARDED_KERNEL` | `attention_isa_kernel` | LNC1, `head_dim ≤ 128`, `seq_len ≥ 4096` |
| `SHARDED_KERNEL` | `attention_isa_kernel` (SPMD) | LNC2, `head_dim ≤ 128`, `seq_len % 1024 == 0` |
| `LARGE_D_UNSHARDED_KERNEL` | `flash_fwd_large_d` | LNC1, `head_dim > 128`, `seq_len ≥ 4096` and `% 2048 == 0` |
| `LARGE_D_SHARDED_KERNEL` | `flash_fwd_large_d` (SPMD) | LNC2, `head_dim > 128`, `seq_len ≥ 2048` and `% 2048 == 0` |

All kernel calls forward `sliding_window=self.sliding_window_size` (default 0 = no
window). Subclasses set `self.sliding_window_size` in `__init__` for models that use
sliding window attention (e.g. Gemma3).

## `flash_fwd_large_d` — NKI kernel for head_dim > 128

### Why it exists

The ISA reference kernel (`attention_isa_kernel`) uses `par_dim(head_dim)` for the QK
matmul. The Neuron tensor engine caps `par_dim` at 128, so models with `head_dim=256`
(e.g. Gemma3-27B) cannot use the ISA kernel directly.

`flash_fwd_large_d` solves this with **d-tiling**: it splits `head_dim` into
128-element chunks and accumulates the QK result across tiles. The PV matmul is
unaffected because `head_dim` sits in the free dimension there (max 512).

### Constraints

| Parameter | Constraint | Reason |
|---|---|---|
| `head_dim` (`d`) | Must be a multiple of 128, ≤ 512 | D-tiling tile size is fixed at 128; free dim limit is 512 |
| `seq_len` | Must be a multiple of `LARGE_TILE_SZ` | K/V tiles must be evenly divisible |
| `LARGE_TILE_SZ` | 2048 when `d=128`, 1024 when `d>128` | SBUF budget: extra `d_n_tiles` reduce the allowed tile size |

### SBUF budget

The main buffers that scale with tile size:

```
qk_res_buf:        (par_dim(128), LARGE_TILE_SZ)   -- QK results
p_local:           (par_dim(128), LARGE_TILE_SZ)   -- softmax numerator
p_local_transposed:(par_dim(128), LARGE_TILE_SZ)   -- transposed p for PV
k_tiles:           (d_n_tiles, par_dim(128), LARGE_TILE_SZ)
o_buffer:          (attn_core_tile_size, par_dim(128), d)
```

`attn_core_tile_size = max(4, 64 // d_n_tiles)` keeps `o_buffer` near ~4 MB.
`LARGE_TILE_SZ` halves from 2048→1024 when `d_n_tiles` doubles from 1→2.

### Sliding window

Two-level filtering avoids unnecessary computation:

1. **Large-tile level** (outer loop): skip entire K large-tiles that are fully
   outside the window (`forward_mask` in `flash_fwd_large_d`).
2. **Position level** (inner core): `affine_select` masks individual positions
   with two chained predicates — causal (`q_pos >= k_pos`) then window
   (`k_pos >= q_pos - window + 1`). `affine_select` only supports a single predicate
   so they are applied sequentially.

### Numerical stability

`m_buffer` and `l_buffer` are initialized to `-1e38` (sentinel for −∞). The unified
online-softmax update path in `_flash_attention_core_large_d` handles both the "first
tile" and "subsequent tiles" cases: when `m = −1e38`, the rescale factor
`α = exp(−1e38 − max) ≈ 0`, so the stale `o_buffer` contribution is zeroed out
automatically without a separate `initialize` flag.

### Adding a new model with head_dim > 128

1. Set `self.sliding_window_size` in your attention subclass `__init__` if the model
   uses sliding window attention (0 otherwise — the base class default).
2. No other changes are needed: `get_flash_attention_strategy()` automatically picks
   `LARGE_D_*` for `head_dim > 128`.
3. Ensure your export `seq_len` is a multiple of 2048 (LNC2) or 4096 (LNC1) so the
   kernel is actually engaged; otherwise it falls back to `NONE`.
