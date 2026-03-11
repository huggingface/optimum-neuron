"""
Benchmark attention kernel paths at small seq_len on Neuron.

Compares:
  1. Compiler-native (XLA matmul/softmax) — current NONE fallback at seq_len < 4096
  2. attention_isa_kernel (ISA) — forced below the normal 4096 threshold
  3. flash_fwd (NKI from neuronxcc) — requires 2D SPMD grid [batch, heads]
  4. flash_attn_neuron (NKI, transformers-native layout) — from torch-neuronx evaluation

Uses Phi-3.5-mini dimensions: batch=4, heads=32, head_dim=96, seq=1024.
Target: inf2 (LNC1).

Usage:
    python benchmark/attention/benchmark_attention_kernels.py [--seq_len 1024] [--batch 4]
"""

import argparse
import math
import os
import tempfile
import time

import torch
import torch_neuronx
from torch_neuronx.xla_impl.ops import nki_jit


# --- Kernel imports ---
try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel

from neuronxcc.nki.kernels.attention import FlashConfig, flash_fwd


# Import flash_attn_neuron from local copy in benchmark/attention/
try:
    from nki_flash_attn import flash_attn_neuron

    _has_flash_attn_neuron = True
except ImportError:
    _has_flash_attn_neuron = False


# ---- Kernel wrappers ----


def make_compiler_native_model(batch, num_heads, seq_len, head_dim, dtype):
    """Trace a compiler-native attention model (matmul + softmax + matmul)."""

    class CompilerNativeAttn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = 1.0 / math.sqrt(head_dim)

        def forward(self, Q, K, V):
            # Q, K, V: (B, H, S, D)
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            # Causal mask
            mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), dtype=scores.dtype, device=scores.device),
                diagonal=1,
            )
            scores = scores + mask
            scores = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(dtype)
            return torch.matmul(scores, V)

    model = CompilerNativeAttn()
    q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype)
    k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype)
    v = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype)

    print("  Tracing compiler-native attention...")
    t0 = time.time()
    traced = torch_neuronx.trace(model, (q, k, v))
    compile_time = time.time() - t0
    print(f"  Compile time: {compile_time:.1f}s")

    return traced, (q, k, v)


def make_isa_kernel_model(batch, num_heads, seq_len, head_dim, dtype):
    """Trace a model that calls the ISA attention kernel."""

    _isa_call = nki_jit()(attention_isa_kernel)
    scale = 1.0 / math.sqrt(head_dim)

    class ISAKernelAttn(torch.nn.Module):
        def forward(self, Q, K, V):
            # ISA kernel expects (B*H, d, seq) for Q/K, (B*H, seq, d) for V
            out = torch.zeros_like(Q)
            _isa_call(
                Q,
                K,
                V,
                1.0,  # scale already applied to Q
                out,
                kernel_name="CausalAttentionMMSoftmaxMMWithoutSwap",
                sliding_window=0,
            )
            return out

    model = ISAKernelAttn()

    bh = batch * num_heads
    q = torch.randn(bh, head_dim, seq_len, dtype=dtype) * scale
    k = torch.randn(bh, head_dim, seq_len, dtype=dtype)
    v = torch.randn(bh, seq_len, head_dim, dtype=dtype)

    print("  Tracing ISA kernel attention...")
    t0 = time.time()
    traced = torch_neuronx.trace(model, (q, k, v))
    compile_time = time.time() - t0
    print(f"  Compile time: {compile_time:.1f}s")

    return traced, (q, k, v)


def make_flash_fwd_model(batch, num_heads, seq_len, head_dim, dtype):
    """Trace a model that calls flash_fwd from neuronxcc.

    flash_fwd requires a 2D SPMD grid [batch, heads].
    Reference: torch-neuronx/torch_neuronx/python_ops/nki_kernels/scaled_dot_product_attention.py
    """

    config = FlashConfig(seq_tile_size=min(seq_len, 2048), training=False, should_transpose_v=False)
    scale = 1.0 / math.sqrt(head_dim)

    class FlashFwdAttn(torch.nn.Module):
        def forward(self, Q, K, V):
            # flash_fwd(q, k, v, seed, logit_bias=None, softmax_scale=None,
            #           use_causal_mask=True, mixed_precision=True, dropout_p=0.0, config=None)
            # q: (bs, n_heads, d, seq_q)
            # k: (bs, n_heads, d, seq_k)
            # v: (bs, n_heads, seq_v, d) when should_transpose_v=False
            # Requires 2D SPMD grid: kernel[batch, heads](...)
            o = flash_fwd[batch, num_heads](
                Q,
                K,
                V,
                None,  # seed (must be None for inference)
                softmax_scale=scale,
                use_causal_mask=True,
                mixed_precision=True,
                config=config,
            )
            return o

    model = FlashFwdAttn()

    q = torch.randn(batch, num_heads, head_dim, seq_len, dtype=dtype)
    k = torch.randn(batch, num_heads, head_dim, seq_len, dtype=dtype)
    v = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype)

    print("  Tracing flash_fwd NKI kernel attention...")
    t0 = time.time()
    traced = torch_neuronx.trace(model, (q, k, v))
    compile_time = time.time() - t0
    print(f"  Compile time: {compile_time:.1f}s")

    return traced, (q, k, v)


def make_flash_attn_neuron_model(batch, num_heads, seq_len, head_dim, dtype):
    """Trace a model that calls flash_attn_neuron (transformers-native layout).

    From torch-neuronx/examples/huggingface/transformers/nki_flash_attn.py.
    Accepts standard transformers layout: Q,K,V as (bs, n_heads, seq, d).
    Requires 2D SPMD grid [batch, heads].
    Constraints: d <= 128, seq_q % 128 == 0, seq_k % 512 == 0.
    """

    scale = 1.0 / math.sqrt(head_dim)

    class FlashAttnNeuron(torch.nn.Module):
        def forward(self, Q, K, V):
            # flash_attn_neuron(q, k, v, attn_mask=None, softmax_scale=None, use_causal_mask=False)
            # q: (bs, n_heads, seq_q, d) — standard transformers layout
            # k: (bs, n_heads, seq_k, d)
            # v: (bs, n_heads, seq_v, d)
            # Returns: (bs, seq_q, n_heads, d)
            o = flash_attn_neuron[batch, num_heads](
                Q,
                K,
                V,
                softmax_scale=scale,
                use_causal_mask=True,
            )
            return o

    model = FlashAttnNeuron()

    q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype)
    k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype)
    v = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype)

    print("  Tracing flash_attn_neuron kernel...")
    t0 = time.time()
    traced = torch_neuronx.trace(model, (q, k, v))
    compile_time = time.time() - t0
    print(f"  Compile time: {compile_time:.1f}s")

    return traced, (q, k, v)


# ---- Benchmarking ----


def benchmark_model(name, traced_model, inputs, warmup=5, iterations=50):
    """Run warmup + timed iterations, return avg latency in ms."""
    print(f"\n  Benchmarking {name}...")

    # Warmup
    for _ in range(warmup):
        traced_model(*inputs)

    # Timed
    t0 = time.time()
    for _ in range(iterations):
        traced_model(*inputs)
    total = time.time() - t0

    avg_ms = (total / iterations) * 1000
    print(f"  {name}: {avg_ms:.2f} ms/iter (avg over {iterations} iterations)")
    return avg_ms


def get_neff_size(traced_model):
    """Get the NEFF binary size as a proxy for instruction count."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        torch.jit.save(traced_model, path)
        size = os.path.getsize(path)
        os.unlink(path)
        return size
    except Exception as e:
        print(f"  Warning: could not measure NEFF size: {e}")
        return None


ALL_KERNELS = ["native", "isa", "flash_fwd", "flash_attn_neuron"]


def main():
    parser = argparse.ArgumentParser(description="Benchmark attention kernels on Neuron")
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--num_heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--head_dim", type=int, default=96, help="Head dimension")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=50, help="Timed iterations")
    parser.add_argument(
        "--kernels",
        nargs="+",
        default=ALL_KERNELS,
        choices=ALL_KERNELS,
        help="Which kernels to benchmark",
    )
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    n_kernels = len(args.kernels)

    print("=== Attention Kernel Benchmark ===")
    print(f"  batch={args.batch}, heads={args.num_heads}, head_dim={args.head_dim}, seq_len={args.seq_len}")
    print(f"  dtype={args.dtype}")
    print()

    results = {}
    step = 0

    # 1. Compiler-native
    if "native" in args.kernels:
        step += 1
        print(f"[{step}/{n_kernels}] Compiler-native (XLA matmul/softmax)")
        traced, inputs = make_compiler_native_model(args.batch, args.num_heads, args.seq_len, args.head_dim, dtype)
        neff_size = get_neff_size(traced)
        if neff_size:
            print(f"  NEFF size: {neff_size / 1024 / 1024:.1f} MB")
        latency = benchmark_model("compiler-native", traced, inputs, args.warmup, args.iterations)
        results["compiler-native"] = {"latency_ms": latency, "neff_mb": neff_size / 1024 / 1024 if neff_size else None}
        del traced
        print()

    # 2. ISA kernel
    if "isa" in args.kernels:
        step += 1
        print(f"[{step}/{n_kernels}] attention_isa_kernel (ISA)")
        traced, inputs = make_isa_kernel_model(args.batch, args.num_heads, args.seq_len, args.head_dim, dtype)
        neff_size = get_neff_size(traced)
        if neff_size:
            print(f"  NEFF size: {neff_size / 1024 / 1024:.1f} MB")
        latency = benchmark_model("isa-kernel", traced, inputs, args.warmup, args.iterations)
        results["isa-kernel"] = {"latency_ms": latency, "neff_mb": neff_size / 1024 / 1024 if neff_size else None}
        del traced
        print()

    # 3. flash_fwd NKI kernel
    if "flash_fwd" in args.kernels:
        step += 1
        print(f"[{step}/{n_kernels}] flash_fwd (NKI from neuronxcc)")
        tile_size = min(args.seq_len, 2048)
        if args.seq_len % tile_size != 0:
            print(f"  SKIP: seq_len={args.seq_len} not divisible by tile_size={tile_size}")
        else:
            traced, inputs = make_flash_fwd_model(args.batch, args.num_heads, args.seq_len, args.head_dim, dtype)
            neff_size = get_neff_size(traced)
            if neff_size:
                print(f"  NEFF size: {neff_size / 1024 / 1024:.1f} MB")
            latency = benchmark_model("flash-fwd", traced, inputs, args.warmup, args.iterations)
            results["flash-fwd"] = {"latency_ms": latency, "neff_mb": neff_size / 1024 / 1024 if neff_size else None}
            del traced
        print()

    # 4. flash_attn_neuron (transformers-native layout)
    if "flash_attn_neuron" in args.kernels:
        step += 1
        print(f"[{step}/{n_kernels}] flash_attn_neuron (NKI, transformers-native)")
        if not _has_flash_attn_neuron:
            print("  SKIP: flash_attn_neuron not available (torch-neuronx examples not found)")
        elif args.head_dim > 128:
            print(f"  SKIP: head_dim={args.head_dim} > 128")
        elif args.seq_len % 128 != 0:
            print(f"  SKIP: seq_len={args.seq_len} not divisible by 128")
        elif args.seq_len % 512 != 0:
            print(f"  SKIP: seq_len={args.seq_len} not divisible by 512")
        else:
            traced, inputs = make_flash_attn_neuron_model(
                args.batch, args.num_heads, args.seq_len, args.head_dim, dtype
            )
            neff_size = get_neff_size(traced)
            if neff_size:
                print(f"  NEFF size: {neff_size / 1024 / 1024:.1f} MB")
            latency = benchmark_model("flash-attn-neuron", traced, inputs, args.warmup, args.iterations)
            results["flash-attn-neuron"] = {
                "latency_ms": latency,
                "neff_mb": neff_size / 1024 / 1024 if neff_size else None,
            }
            del traced
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Kernel':<22} {'Latency (ms)':<15} {'NEFF (MB)':<12}")
    print("-" * 49)
    for name, data in results.items():
        neff_str = f"{data['neff_mb']:.1f}" if data["neff_mb"] else "N/A"
        print(f"{name:<22} {data['latency_ms']:<15.2f} {neff_str:<12}")


if __name__ == "__main__":
    main()
