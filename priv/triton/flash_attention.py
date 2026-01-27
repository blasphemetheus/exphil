#!/usr/bin/env python3
"""
Flash Attention benchmark for ExPhil.

Compares standard attention vs PyTorch's built-in scaled_dot_product_attention
which uses Flash Attention under the hood (PyTorch 2.0+).

Usage:
    python priv/triton/flash_attention.py              # Run benchmark
    python priv/triton/flash_attention.py --test       # Run correctness test
"""

import argparse
import math
import torch


def attention_standard(q, k, v):
    """
    Standard attention (O(N²) memory).

    Args:
        q, k, v: [batch, heads, seq_len, head_dim]
    """
    head_dim = q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)

    # O(N²) attention matrix
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)

    return out


def attention_flash(q, k, v):
    """
    Flash Attention via PyTorch's scaled_dot_product_attention.

    This automatically uses the most efficient implementation:
    - Flash Attention (if available)
    - Memory-efficient attention
    - Math attention (fallback)
    """
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)


def benchmark_attention(batch=32, heads=4, seq_len=60, head_dim=64, iterations=100, warmup=10):
    """Benchmark standard vs Flash Attention."""
    print("=" * 60)
    print("Flash Attention Benchmark")
    print("=" * 60)
    print(f"\nConfig: batch={batch}, heads={heads}, seq_len={seq_len}, head_dim={head_dim}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cpu":
        print("\nWARNING: Running on CPU. Results not meaningful.")

    # Create test data
    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)

    results = {}

    # Warmup
    print(f"\nWarming up ({warmup} iterations)...")
    for _ in range(warmup):
        _ = attention_standard(q, k, v)
        _ = attention_flash(q, k, v)
    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark standard attention
    print(f"\nBenchmarking standard attention ({iterations} iterations)...")
    if device == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    for _ in range(iterations):
        _ = attention_standard(q, k, v)

    if device == "cuda":
        end.record()
        torch.cuda.synchronize()
        standard_time = start.elapsed_time(end) / iterations
    else:
        import time
        t0 = time.time()
        for _ in range(iterations):
            _ = attention_standard(q, k, v)
        standard_time = (time.time() - t0) / iterations * 1000

    print(f"  Standard: {standard_time:.3f} ms")
    results['standard'] = standard_time

    # Benchmark Flash Attention (PyTorch SDPA)
    print(f"\nBenchmarking Flash Attention / SDPA ({iterations} iterations)...")
    if device == "cuda":
        torch.cuda.synchronize()
        start.record()

    for _ in range(iterations):
        _ = attention_flash(q, k, v)

    if device == "cuda":
        end.record()
        torch.cuda.synchronize()
        flash_time = start.elapsed_time(end) / iterations
    else:
        t0 = time.time()
        for _ in range(iterations):
            _ = attention_flash(q, k, v)
        flash_time = (time.time() - t0) / iterations * 1000

    print(f"  Flash/SDPA: {flash_time:.3f} ms")
    results['flash'] = flash_time

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    speedup = standard_time / flash_time
    target = 16.67

    print(f"\n60 FPS threshold: {target:.2f} ms\n")
    print(f"  Standard attention: {standard_time:6.3f} ms  (1.00x)  60 FPS: {'✓' if standard_time < target else '✗'}")
    print(f"  Flash/SDPA:         {flash_time:6.3f} ms  ({speedup:.2f}x)  60 FPS: {'✓' if flash_time < target else '✗'}")

    # Memory comparison
    print(f"\nMemory comparison:")
    n2_memory = batch * heads * seq_len * seq_len * 4  # float32
    print(f"  Standard: ~{n2_memory / 1024:.1f} KB (stores N² attention matrix)")
    print(f"  Flash:    O(N) working memory (no N² matrix)")

    # Check which backend SDPA is using
    if device == "cuda":
        print(f"\nSDPA backend info:")
        print(f"  Flash Attention available: {torch.backends.cuda.flash_sdp_enabled()}")
        print(f"  Memory-efficient available: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
        print(f"  Math fallback enabled: {torch.backends.cuda.math_sdp_enabled()}")

    return results


def test_correctness():
    """Test that Flash Attention produces correct results."""
    print("=" * 60)
    print("Flash Attention Correctness Test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    configs = [
        (1, 1, 16, 32),
        (2, 4, 32, 64),
        (4, 4, 64, 64),
        (32, 4, 60, 64),  # ExPhil default
    ]

    all_passed = True
    for batch, heads, seq_len, head_dim in configs:
        print(f"Testing batch={batch}, heads={heads}, seq={seq_len}, dim={head_dim}...", end=" ")

        q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
        k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
        v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)

        ref = attention_standard(q, k, v)
        out = attention_flash(q, k, v)

        max_diff = (ref - out).abs().max().item()
        passed = max_diff < 1e-3
        status = "PASS" if passed else "FAIL"
        all_passed = all_passed and passed

        print(f"max_diff={max_diff:.6f} [{status}]")

    print("\n" + "=" * 60)
    print("All tests PASSED!" if all_passed else "Some tests FAILED!")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flash Attention benchmark")
    parser.add_argument("--test", action="store_true", help="Run correctness test")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seq", type=int, default=60)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    if args.test:
        test_correctness()
    else:
        benchmark_attention(args.batch, args.heads, args.seq, args.dim, args.iterations)
