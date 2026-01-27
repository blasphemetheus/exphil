#!/usr/bin/env python3
"""
Flash Attention implementation in Triton for ExPhil.

This implements Flash Attention 2 (Dao et al., 2023) which computes exact
attention without materializing the O(N²) attention matrix.

Key insight: Softmax can be computed incrementally using the "online softmax"
trick, allowing block-wise computation that fits in SRAM.

Usage:
    python priv/triton/flash_attention.py              # Run benchmark
    python priv/triton/flash_attention.py --test       # Run correctness test
    python priv/triton/flash_attention.py --profile    # Profile kernel

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
# Triton Kernel
# =============================================================================

@triton.jit
def flash_attention_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    seq_len, head_dim,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Flash Attention forward pass.

    Grid: (num_blocks_m, batch * num_heads)

    Each block computes a BLOCK_M x head_dim tile of the output.
    """
    # Program IDs
    block_m = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // (stride_qh // stride_qb) if stride_qh > stride_qb else batch_head
    head = batch_head % (stride_qh // stride_qb) if stride_qh > stride_qb else 0

    # Compute batch and head indices
    # Assuming layout: [batch, heads, seq, dim] with strides
    batch = batch_head // tl.cdiv(stride_qb, stride_qh) if stride_qb > stride_qh else batch_head // 1

    # Simpler: assume batch_head encodes both
    # We'll handle this via strides

    # Offsets for this block
    offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Base pointers for this batch/head
    q_base = Q + batch_head * stride_qh
    k_base = K + batch_head * stride_kh
    v_base = V + batch_head * stride_vh
    o_base = Out + batch_head * stride_oh

    # Initialize output accumulator and softmax stats
    # m_i: running max, l_i: running sum of exp
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

    # Load Q block (stays in SRAM for entire computation)
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q_mask = offs_m[:, None] < seq_len
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Iterate over K, V blocks
    num_blocks_n = tl.cdiv(seq_len, BLOCK_N)
    for block_n in range(num_blocks_n):
        start_n = block_n * BLOCK_N
        offs_n_curr = start_n + tl.arange(0, BLOCK_N)

        # Load K block
        k_ptrs = k_base + offs_n_curr[:, None] * stride_kn + offs_k[None, :] * stride_kk
        k_mask = offs_n_curr[:, None] < seq_len
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # Compute QK^T for this block
        # q: [BLOCK_M, BLOCK_K], k: [BLOCK_N, BLOCK_K]
        # qk: [BLOCK_M, BLOCK_N]
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, tl.trans(k), qk)
        qk *= scale

        # Mask out invalid positions
        qk_mask = (offs_m[:, None] < seq_len) & (offs_n_curr[None, :] < seq_len)
        qk = tl.where(qk_mask, qk, float('-inf'))

        # Online softmax update
        # m_ij: max of this block
        m_ij = tl.max(qk, axis=1)
        # New running max
        m_new = tl.maximum(m_i, m_ij)

        # Correction factors
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)

        # Update l_i (sum of exponentials)
        l_i = alpha * l_i + beta * tl.sum(tl.exp(qk - m_ij[:, None]), axis=1)

        # Compute attention weights for this block
        p = tl.exp(qk - m_new[:, None])

        # Load V block
        v_ptrs = v_base + offs_n_curr[:, None] * stride_vn + offs_k[None, :] * stride_vk
        v_mask = offs_n_curr[:, None] < seq_len
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # Update accumulator: scale old acc and add new contribution
        acc = alpha[:, None] * acc + tl.dot(p.to(v.dtype), v)

        # Update running max
        m_i = m_new

    # Final normalization
    acc = acc / l_i[:, None]

    # Store output
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    o_mask = offs_m[:, None] < seq_len
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=o_mask)


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
    batch, heads, seq_len, head_dim = q.shape

    # Allocate output
    out = torch.empty_like(q)

    # Kernel config
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = head_dim  # Must match head_dim

    # Scale factor
    scale = 1.0 / math.sqrt(head_dim)

    # Grid: one block per (seq_chunk, batch*head)
    num_blocks_m = triton.cdiv(seq_len, BLOCK_M)
    grid = (num_blocks_m, batch * heads)

    # Get strides
    stride_qb, stride_qh, stride_qm, stride_qk = q.stride()
    stride_kb, stride_kh, stride_kn, stride_kk = k.stride()
    stride_vb, stride_vh, stride_vn, stride_vk = v.stride()
    stride_ob, stride_oh, stride_om, stride_ok = out.stride()

    # Launch kernel
    flash_attention_kernel[grid](
        q, k, v, out,
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_kn, stride_kk,
        stride_vb, stride_vh, stride_vn, stride_vk,
        stride_ob, stride_oh, stride_om, stride_ok,
        seq_len, head_dim,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return out


def attention_pytorch_reference(q, k, v):
    """
    Standard attention using PyTorch (O(N²) memory).

    Args:
        q, k, v: [batch, heads, seq_len, head_dim]

    Returns:
        Output tensor [batch, heads, seq_len, head_dim]
    """
    head_dim = q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)

    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, N, N]

    # Softmax
    attn = torch.softmax(scores, dim=-1)

    # Apply to values
    out = torch.matmul(attn, v)

    return out


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
        return

    # Create test data
    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)

    # Warmup
    print(f"\nWarming up ({warmup} iterations)...")
    for _ in range(warmup):
        _ = attention_pytorch_reference(q, k, v)
        _ = flash_attention_triton(q, k, v)
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
    print(f"  PyTorch: {pytorch_time:.3f} ms")

    # Benchmark Triton Flash Attention
    print(f"\nBenchmarking Triton Flash Attention ({iterations} iterations)...")
    torch.cuda.synchronize()

    start.record()
    for _ in range(iterations):
        _ = flash_attention_triton(q, k, v)
    end.record()
    torch.cuda.synchronize()

    triton_time = start.elapsed_time(end) / iterations
    print(f"  Triton: {triton_time:.3f} ms")

    # Speedup
    speedup = pytorch_time / triton_time
    print(f"\nSpeedup: {speedup:.2f}x")

    # 60 FPS check
    target = 16.67
    print(f"\n60 FPS threshold: {target:.2f} ms")
    print(f"PyTorch meets 60 FPS: {'YES' if pytorch_time < target else 'NO'}")
    print(f"Triton meets 60 FPS: {'YES' if triton_time < target else 'NO'}")

    # Memory comparison
    print(f"\nMemory comparison (theoretical):")
    n2_memory = batch * heads * seq_len * seq_len * 4  # float32
    linear_memory = batch * heads * seq_len * head_dim * 4 * 3  # Q, K, V only
    print(f"  Standard attention: {n2_memory / 1024:.1f} KB (O(N²) for scores)")
    print(f"  Flash attention: ~{linear_memory / 1024:.1f} KB (O(N) working memory)")

    return pytorch_time, triton_time


def test_correctness():
    """Test that Flash Attention produces correct results."""
    print("=" * 60)
    print("Flash Attention Correctness Test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test configurations
    configs = [
        (1, 1, 16, 32),   # Tiny
        (2, 4, 32, 64),   # Small
        (4, 4, 64, 64),   # Medium
        (32, 4, 60, 64),  # ExPhil default
    ]

    all_passed = True
    for batch, heads, seq_len, head_dim in configs:
        print(f"\nTesting batch={batch}, heads={heads}, seq={seq_len}, dim={head_dim}...")

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

        # Tolerance (Flash Attention has slight numerical differences due to reordering)
        passed = max_diff < 1e-2
        status = "PASS" if passed else "FAIL"
        all_passed = all_passed and passed

        print(f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f} [{status}]")

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
    parser.add_argument("--profile", action="store_true", help="Profile kernel")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--seq", type=int, default=60, help="Sequence length")
    parser.add_argument("--dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations")
    args = parser.parse_args()

    if args.test:
        test_correctness()
    elif args.profile:
        print("Profiling not yet implemented. Use nsys or ncu.")
    else:
        benchmark_attention(
            batch=args.batch,
            heads=args.heads,
            seq_len=args.seq,
            head_dim=args.dim,
            iterations=args.iterations
        )
