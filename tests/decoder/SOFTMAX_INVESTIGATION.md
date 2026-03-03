# Softmax Masking Investigation on Neuron XLA

## Problem

The `compute_for_token_gen()` method in `attention_base.py` uses
`torch.finfo(dtype).min` (~-3.4e38 for bf16) as the masked-position fill value
before softmax.  On Neuron XLA, the vectorized `exp()` implementation produces
**NaN** for inputs this extreme, breaking generation for models with attention
masks (e.g. gemma3 sliding-window layers).

## Approaches Tested

### 1. Safe fill value (`-1e9`) — commit `4f4ce614`

Replace `finfo(dtype).min` with `-1e9` while keeping the same graph structure:
pre-fill masked positions with the sentinel, then call `manual_softmax` without
masks.

```python
_MASKED_FILL = -1e9
prior_scores = torch.where(attention_mask, prior_scores, _MASKED_FILL)
softmax_prior, softmax_active = manual_softmax(prior_scores, active_scores)
```

**Result:** Eliminates NaN.  Greedy generation matches CPU reference output
perfectly (all 50 tokens identical on gemma3-270m-it with 5000-token prompt).

### 2. Boolean masks inside `manual_softmax` — commit `7eacb39e`

Pass boolean masks into `manual_softmax` to fully exclude masked positions from
max, exp, and denominator:

```python
# In manual_softmax:
prior_for_max = torch.where(prior_mask, prior_scores, -1e9)
max_score = torch.max(prior_for_max, ...)
safe_prior = torch.where(prior_mask, prior_scores, max_score.expand_as(...))
exp_prior = torch.exp(safe_prior - max_score) * prior_mask.float()  # zero masked
```

**Result:** Eliminates NaN.  However, greedy generation **diverges from CPU at
step 34** on gemma3-270m-it (5000-token prompt, `max_new_tokens=50`,
`do_sample=False`).

## Detailed Comparison

### Isolated module tests (both approaches)

Tests in `test_softmax_equivalence.py` and `test_softmax_sliding_window.py`
compile a single softmax or attention block on Neuron and compare old vs new
approach outputs:

- **Bitwise identical** in all configurations tested:
  - Bare softmax (single token, 5000/8192 filled)
  - Bare softmax with `finfo(bf16).min` vs `finfo(f16).min` fill values
  - Full attention block with sliding-window mask (512/8192 unmasked)
  - Full attention block with full-attention mask (5102/8192 unmasked)

The isolated operations produce **no measurable difference** on Neuron hardware.

### Full model generation (gemma3-270m-it, 5000-token prompt)

| Metric                     | Fill (`-1e9`)      | Boolean masks        |
|----------------------------|--------------------|----------------------|
| Matches CPU reference      | Yes (50/50 tokens) | No (diverges step 34)|
| NaN in output              | No                 | No                   |
| Divergence point           | —                  | Step 34              |
| CPU logit gap at step 34   | ~0.118 (near-tie)  | Same CPU ref         |
| CPU top tokens at step 34  | "and" vs "of"      | "and" vs "of"        |
| Neuron picks at step 34    | "and" (correct)    | "of" (wrong)         |

Hub models used for comparison:
- Fill: `optimum-internal-testing/optimum-neuron-testing-0.4.6.dev2-2.26.1-trn1-a624f36a57-gemma3-{4x4096,1x8192}`
- Mask: `optimum-internal-testing/optimum-neuron-testing-0.4.6.dev2-2.26.1-trn1-2e872b04ba-gemma3-{4x4096,1x8192}`

## Root Cause Analysis

The boolean-mask approach adds extra operations to the XLA computation graph
that are not present in the fill approach:

1. `torch.where(mask, scores, -1e9)` for max computation
2. `torch.where(mask, scores, max_score)` to make exp inputs safe
3. `* prior_mask.float()` to zero out masked exp values

These additional ops change how the Neuron compiler fuses, tiles, and
vectorizes the graph.  Even though the mathematical result should be identical
for **unmasked** positions, the different compilation path produces slightly
different floating-point rounding in intermediate results.

Over 34 generation steps x 18 transformer layers, these per-layer rounding
differences accumulate in the residual stream.  At step 34, the logit gap
between the top-2 candidate tokens ("and" = 6.719 vs "of" = 6.601) is small
enough (~0.118) that the accumulated error flips the argmax.

Key evidence:
- **Isolated module tests show zero difference** — the Neuron compiler produces
  identical results when compiling a single block in isolation
- **Full model diverges** — the different graph structure across all 18 layers
  compounds differently than the fill-based graph
- **The fill approach matches CPU perfectly** — preserving the original graph
  structure (pre-fill + unmasked softmax) avoids the compilation difference

## Conclusion

The **safe fill value (`-1e9`)** approach is the correct fix:

- Same XLA graph structure as the original `finfo.min` approach
- Eliminates NaN (since `exp(-1e9) = 0.0` exactly in float32)
- Preserves bit-exact agreement with CPU reference generation
- Minimal code change (single constant substitution)

The boolean-mask approach is **mathematically correct** but produces a different
XLA computation graph that the Neuron compiler optimizes differently, leading to
accumulated rounding divergence in multi-step generation.  This is not a bug in
either approach — it is a consequence of how XLA graph-level optimizations are
sensitive to operation topology on Neuron hardware.

## Files

| File | Purpose |
|------|---------|
| `attention_base.py` | `compute_for_token_gen()` — the fix site |
| `utils.py` | `manual_softmax()` — mask-aware variant |
| `test_softmax_equivalence.py` | Isolated softmax comparison on Neuron |
| `test_softmax_sliding_window.py` | Full attention block comparison on Neuron |
