#!/usr/bin/env python3
"""
Flash Attention implementation in Triton for ExPhil.

Simplified implementation based on the official Triton tutorial.
Computes exact attention without materializing the O(N²) attention matrix.

Usage:
    python priv/triton/flash_attention.py              # Run benchmark
    python priv/triton/flash_attention.py --test       # Run correctness test

References:
    - FlashAttention-2: https://arxiv.org/abs/2307.08691
    - Triton tutorial: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
"""

import argparse
import math
import torch
import triton
import triton.language as tl


# =============================================================================
# Triton Kernel (Simplified)
# =============================================================================

@triton.jit
def _flash_attn_fwd(
    Q, K, V, Out,
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """
    Flash Attention forward kernel.

    Grid: (cdiv(N_CTX, BLOCK_M), Z * H)
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Initialize pointers to Q, K, V
    off_q = off_z * stride_qz + off_h * stride_qh
    off_k = off_z * stride_kz + off_h * stride_kh
    off_v = off_z * stride_vz + off_h * stride_vh

    q_ptrs = Q + off_q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + off_k + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + off_v + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    # Initialize accumulator
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Load Q (stays in SRAM)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # Loop over K, V blocks
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # Load K, V
        k = tl.load(k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n[:, None]) < N_CTX, other=0.0)
        v = tl.load(v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n[:, None]) < N_CTX, other=0.0)

        # Compute QK^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= sm_scale

        # Mask out-of-bounds
        qk = tl.where(
            (offs_m[:, None] < N_CTX) & ((start_n + offs_n[None, :]) < N_CTX),
            qk, float("-inf")
        )

        # Online softmax
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_ij = tl.sum(p, 1)
        l_new = alpha * l_i + l_ij

        # Update accumulator
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)

        # Update stats
        l_i = l_new
        m_i = m_new

    # Final normalization
    acc = acc / l_i[:, None]

    # Store output
    off_o = off_z * stride_oz + off_h * stride_oh
    o_ptrs = Out + off_o + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


# =============================================================================
# Python Wrapper
# =============================================================================

def flash_attention_triton(q, k, v):
    """
    Flash Attention using Triton kernel.

    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        v: Value tensor [batch, heads, seq_len, head_dim]

    Returns:
        Output tensor [batch, heads, seq_len, head_dim]
    """
    # Input validation
    assert q.dim() == 4, f"Expected 4D tensor, got {q.dim()}D"
    batch, heads, seq_len, head_dim = q.shape
    assert k.shape == q.shape and v.shape == q.shape

    # Head dim must be power of 2 and <= 128 for efficiency
    assert head_dim in [16, 32, 64, 128], f"head_dim must be 16, 32, 64, or 128, got {head_dim}"

    # Ensure contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # Allocate output
    out = torch.empty_like(q)

    # Kernel config
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_DMODEL = head_dim

    # Scale factor
    sm_scale = 1.0 / math.sqrt(head_dim)

    # Grid
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * heads)

    # Launch
    _flash_attn_fwd[grid](
        q, k, v, out,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        batch, heads, seq_len,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
    )

    return out


def attention_pytorch_reference(q, k, v):
    """
    Standard attention using PyTorch (O(N²) memory).
    """
    head_dim = q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)

    return out


def attention_pytorch_sdpa(q, k, v):
    """
    PyTorch's built-in scaled_dot_product_attention (uses Flash Attention when available).
    """
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)


# =============================================================================
# Benchmarks
# =============================================================================

def benchmark_attention(batch=32, heads=4, seq_len=60, head_dim=64, iterations=100, warmup=10):
    """Benchmark Flash Attention vs standard attention."""
    print("=" * 60)
    print("Flash Attention Benchmark")
    print("=" * 60)
    print(f"\nConfig: batch={batch}, heads={heads}, seq_len={seq_len}, head_dim={head_dim}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cpu":
        print("\nWARNING: Running on CPU. Install CUDA for meaningful benchmarks.")
        return None, None, None

    # Create test data
    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)

    results = {}

    # Warmup all implementations
    print(f"\nWarming up ({warmup} iterations)...")
    for _ in range(warmup):
        _ = attention_pytorch_reference(q, k, v)
        try:
            _ = flash_attention_triton(q, k, v)
        except Exception as e:
            print(f"  Triton warmup error: {e}")
        try:
            _ = attention_pytorch_sdpa(q, k, v)
        except Exception:
            pass
    torch.cuda.synchronize()

    # Benchmark PyTorch reference
    print(f"\nBenchmarking PyTorch reference ({iterations} iterations)...")
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iterations):
        _ = attention_pytorch_reference(q, k, v)
    end.record()
    torch.cuda.synchronize()

    pytorch_time = start.elapsed_time(end) / iterations
    print(f"  PyTorch reference: {pytorch_time:.3f} ms")
    results['pytorch'] = pytorch_time

    # Benchmark PyTorch SDPA (built-in Flash Attention)
    try:
        print(f"\nBenchmarking PyTorch SDPA ({iterations} iterations)...")
        torch.cuda.synchronize()
        start.record()
        for _ in range(iterations):
            _ = attention_pytorch_sdpa(q, k, v)
        end.record()
        torch.cuda.synchronize()

        sdpa_time = start.elapsed_time(end) / iterations
        print(f"  PyTorch SDPA: {sdpa_time:.3f} ms")
        results['sdpa'] = sdpa_time
    except Exception as e:
        print(f"  PyTorch SDPA not available: {e}")
        sdpa_time = None

    # Benchmark Triton Flash Attention
    try:
        print(f"\nBenchmarking Triton Flash Attention ({iterations} iterations)...")
        torch.cuda.synchronize()
        start.record()
        for _ in range(iterations):
            _ = flash_attention_triton(q, k, v)
        end.record()
        torch.cuda.synchronize()

        triton_time = start.elapsed_time(end) / iterations
        print(f"  Triton: {triton_time:.3f} ms")
        results['triton'] = triton_time
    except Exception as e:
        print(f"  Triton error: {e}")
        triton_time = None

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    target = 16.67
    print(f"\n60 FPS threshold: {target:.2f} ms\n")

    for name, time in results.items():
        if time:
            meets = "✓ YES" if time < target else "✗ NO"
            speedup = pytorch_time / time if name != 'pytorch' else 1.0
            print(f"  {name:20s}: {time:6.3f} ms  ({speedup:.2f}x)  60 FPS: {meets}")

    return results


def test_correctness():
    """Test that Flash Attention produces correct results."""
    print("=" * 60)
    print("Flash Attention Correctness Test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Test configurations (head_dim must be power of 2)
    configs = [
        (1, 1, 16, 32),   # Tiny
        (2, 4, 32, 64),   # Small
        (4, 4, 64, 64),   # Medium
        (32, 4, 60, 64),  # ExPhil default
    ]

    all_passed = True
    for batch, heads, seq_len, head_dim in configs:
        print(f"Testing batch={batch}, heads={heads}, seq={seq_len}, dim={head_dim}...", end=" ")

        try:
            q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
            k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
            v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)

            # Reference
            ref = attention_pytorch_reference(q, k, v)

            # Flash attention
            out = flash_attention_triton(q, k, v)

            # Compare
            max_diff = (ref - out).abs().max().item()
            mean_diff = (ref - out).abs().mean().item()

            # Tolerance (Flash Attention has slight numerical differences)
            passed = max_diff < 0.01
            status = "PASS" if passed else "FAIL"
            all_passed = all_passed and passed

            print(f"max_diff={max_diff:.6f} [{status}]")

        except Exception as e:
            print(f"ERROR: {e}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)

    return all_passed


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flash Attention benchmark")
    parser.add_argument("--test", action="store_true", help="Run correctness test")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--seq", type=int, default=60, help="Sequence length")
    parser.add_argument("--dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations")
    args = parser.parse_args()

    if args.test:
        test_correctness()
    else:
        benchmark_attention(
            batch=args.batch,
            heads=args.heads,
            seq_len=args.seq,
            head_dim=args.dim,
            iterations=args.iterations
        )
