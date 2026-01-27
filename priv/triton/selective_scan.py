#!/usr/bin/env python3
"""
Triton Selective Scan Kernel for Mamba SSM

This implements the core SSM recurrence:
    h[t] = A[t] * h[t-1] + B[t] * x[t]
    y[t] = C[t] * h[t]

Using parallel associative scan with Triton's built-in primitives.

Usage:
    python selective_scan.py              # Run benchmarks
    python selective_scan.py --test       # Run correctness tests
    python selective_scan.py --profile    # Profile kernel

Requirements:
    pip install triton torch

Integration with Elixir:
    See docs/CUSTOM_KERNELS.md for Port-based integration
"""

import torch
import triton
import triton.language as tl
import argparse
import time


# =============================================================================
# Triton Kernel: Selective Scan Forward
# =============================================================================

@triton.jit
def selective_scan_fwd_kernel(
    # Pointers to tensors
    x_ptr,          # Input: [batch, seq_len, hidden]
    a_ptr,          # Discretized A: [batch, seq_len, hidden, state]
    b_ptr,          # Discretized B*x: [batch, seq_len, hidden, state]
    c_ptr,          # C matrix: [batch, seq_len, state]
    out_ptr,        # Output: [batch, seq_len, hidden]
    # Dimensions
    batch: tl.constexpr,
    seq_len: tl.constexpr,
    hidden: tl.constexpr,
    state: tl.constexpr,
    # Block sizes
    BLOCK_SEQ: tl.constexpr,
    BLOCK_STATE: tl.constexpr,
):
    """
    Parallel selective scan using associative scan.

    Each program instance handles one (batch, hidden_dim) pair.
    Within each instance, we scan across seq_len for all state dimensions.
    """
    # Program ID identifies which (batch, hidden) we're processing
    pid_batch = tl.program_id(0)
    pid_hidden = tl.program_id(1)

    # Base offsets
    batch_offset = pid_batch * seq_len * hidden * state
    hidden_offset = pid_hidden * state

    # Process each state dimension
    # (In a more optimized version, we'd vectorize across state too)
    for s in range(state):
        # Load A and Bx for this (batch, hidden, state) across all seq positions
        seq_offs = tl.arange(0, BLOCK_SEQ)
        mask = seq_offs < seq_len

        # Compute flat indices
        # Layout: [batch, seq, hidden, state] -> batch*seq*hidden*state + seq*hidden*state + hidden*state + state
        a_idx = batch_offset + seq_offs * hidden * state + hidden_offset + s
        b_idx = batch_offset + seq_offs * hidden * state + hidden_offset + s

        a = tl.load(a_ptr + a_idx, mask=mask, other=1.0)
        bx = tl.load(b_ptr + b_idx, mask=mask, other=0.0)

        # Associative scan operation: (a1, b1) ⊗ (a2, b2) = (a1*a2, a1*b2 + b1)
        # This computes h[t] = A[t]*h[t-1] + Bx[t] in parallel

        # Triton's associative_scan with custom combine function
        # For now, use a simple sequential scan (Triton's associative_scan
        # requires specific setup)
        h = tl.zeros([BLOCK_SEQ], dtype=tl.float32)
        h_prev = 0.0
        for t in range(seq_len):
            a_t = tl.load(a_ptr + batch_offset + t * hidden * state + hidden_offset + s)
            bx_t = tl.load(b_ptr + batch_offset + t * hidden * state + hidden_offset + s)
            h_t = a_t * h_prev + bx_t
            h_prev = h_t
            # Store h[t] for this state dimension
            # We'll accumulate y = sum_s(C[s] * h[s]) after

        # Load C and compute output contribution
        # C is [batch, seq, state]
        c_base = pid_batch * seq_len * state

        for t in range(seq_len):
            c_t = tl.load(c_ptr + c_base + t * state + s)
            # Recompute h[t] (or we could store it)
            # For efficiency, we should store h and do a separate output pass


# =============================================================================
# Simpler Kernel: Fused Scan (Sequential but Fused)
# =============================================================================

@triton.jit
def selective_scan_fused_kernel(
    # Inputs
    x_ptr,          # [batch, seq_len, hidden]
    dt_ptr,         # [batch, seq_len, hidden]
    A_ptr,          # [hidden, state] - diagonal A matrix (shared)
    B_ptr,          # [batch, seq_len, state]
    C_ptr,          # [batch, seq_len, state]
    # Output
    out_ptr,        # [batch, seq_len, hidden]
    # Dimensions
    seq_len,
    hidden,
    state,
    # Strides
    stride_x_batch, stride_x_seq, stride_x_hidden,
    stride_dt_batch, stride_dt_seq, stride_dt_hidden,
    stride_B_batch, stride_B_seq, stride_B_state,
    stride_C_batch, stride_C_seq, stride_C_state,
    stride_out_batch, stride_out_seq, stride_out_hidden,
    # Block size
    BLOCK_STATE: tl.constexpr,
):
    """
    Fused selective scan kernel.

    Each program handles one (batch, hidden_dim) pair.
    Scans sequentially but fuses all operations (discretization, scan, output).
    """
    pid_batch = tl.program_id(0)
    pid_hidden = tl.program_id(1)

    # State indices
    state_offs = tl.arange(0, BLOCK_STATE)
    state_mask = state_offs < state

    # Initialize hidden state: h = [BLOCK_STATE] zeros
    h = tl.zeros([BLOCK_STATE], dtype=tl.float32)

    # Load A diagonal for this hidden dim (shared across batch/seq)
    # A is [hidden, state]
    A_diag = tl.load(A_ptr + pid_hidden * state + state_offs, mask=state_mask, other=-1.0)

    # Scan through sequence
    for t in range(seq_len):
        # Load inputs for this timestep
        x_idx = pid_batch * stride_x_batch + t * stride_x_seq + pid_hidden * stride_x_hidden
        x_t = tl.load(x_ptr + x_idx)
        dt_t = tl.load(dt_ptr + x_idx)

        # Clamp dt
        dt_t = tl.minimum(tl.maximum(dt_t, 0.001), 0.1)

        # Load B and C for this timestep: [state] vectors
        bc_base = pid_batch * stride_B_batch + t * stride_B_seq
        B_t = tl.load(B_ptr + bc_base + state_offs, mask=state_mask, other=0.0)
        C_t = tl.load(C_ptr + bc_base + state_offs, mask=state_mask, other=0.0)

        # Discretize
        A_bar = tl.exp(dt_t * A_diag)  # [BLOCK_STATE]
        B_bar = dt_t * B_t             # [BLOCK_STATE]

        # SSM recurrence: h = A_bar * h + B_bar * x
        h = A_bar * h + B_bar * x_t

        # Output: y = C * h (masked sum)
        y_t = tl.sum(tl.where(state_mask, C_t * h, 0.0))

        # Store output
        out_idx = pid_batch * stride_out_batch + t * stride_out_seq + pid_hidden * stride_out_hidden
        tl.store(out_ptr + out_idx, y_t)


# =============================================================================
# PyTorch Wrapper
# =============================================================================

def selective_scan_triton(x, dt, A, B, C, dt_min=0.001, dt_max=0.1):
    """
    Triton-accelerated selective scan.

    Args:
        x: Input tensor [batch, seq_len, hidden]
        dt: Delta/timestep [batch, seq_len, hidden]
        A: State transition [hidden, state] (diagonal, typically negative)
        B: Input projection [batch, seq_len, state]
        C: Output projection [batch, seq_len, state]

    Returns:
        y: Output tensor [batch, seq_len, hidden]
    """
    batch, seq_len, hidden = x.shape
    state = A.shape[1]

    # Ensure contiguous
    x = x.contiguous()
    dt = dt.contiguous()
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    # Allocate output
    out = torch.empty_like(x)

    # Block size must be power of 2 and >= state
    BLOCK_STATE = triton.next_power_of_2(state)

    # Launch kernel: one program per (batch, hidden) pair
    grid = (batch, hidden)

    selective_scan_fused_kernel[grid](
        x, dt, A, B, C, out,
        seq_len, hidden, state,
        # Strides for x
        x.stride(0), x.stride(1), x.stride(2),
        # Strides for dt
        dt.stride(0), dt.stride(1), dt.stride(2),
        # Strides for B
        B.stride(0), B.stride(1), B.stride(2),
        # Strides for C
        C.stride(0), C.stride(1), C.stride(2),
        # Strides for out
        out.stride(0), out.stride(1), out.stride(2),
        # Block size
        BLOCK_STATE=BLOCK_STATE,
    )

    return out


# =============================================================================
# Reference Implementation (PyTorch, for correctness testing)
# =============================================================================

def selective_scan_reference(x, dt, A, B, C, dt_min=0.001, dt_max=0.1):
    """
    Reference implementation in pure PyTorch (sequential).
    """
    batch, seq_len, hidden = x.shape
    state = A.shape[1]
    device = x.device

    # Clamp dt
    dt = dt.clamp(dt_min, dt_max)

    # Initialize
    h = torch.zeros(batch, hidden, state, device=device)
    outputs = []

    for t in range(seq_len):
        # Discretize
        dt_t = dt[:, t, :]  # [batch, hidden]
        A_bar = torch.exp(dt_t.unsqueeze(-1) * A)  # [batch, hidden, state]
        B_bar = dt_t.unsqueeze(-1) * B[:, t, :].unsqueeze(1)  # [batch, hidden, state]

        # Recurrence
        x_t = x[:, t, :]  # [batch, hidden]
        h = A_bar * h + B_bar * x_t.unsqueeze(-1)

        # Output
        C_t = C[:, t, :]  # [batch, state]
        y_t = (h * C_t.unsqueeze(1)).sum(-1)  # [batch, hidden]
        outputs.append(y_t)

    return torch.stack(outputs, dim=1)


# =============================================================================
# Benchmarking
# =============================================================================

def benchmark():
    """Benchmark Triton vs PyTorch reference."""
    print("=" * 60)
    print("Selective Scan Benchmark")
    print("=" * 60)

    # Configuration matching ExPhil defaults
    batch = 32
    seq_len = 60
    hidden = 512  # inner_size = hidden_size * expand_factor
    state = 16

    print(f"\nConfig: batch={batch}, seq={seq_len}, hidden={hidden}, state={state}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cpu":
        print("WARNING: No CUDA available, Triton kernel won't run")
        return

    # Create inputs
    x = torch.randn(batch, seq_len, hidden, device=device)
    dt = torch.rand(batch, seq_len, hidden, device=device) * 0.1
    A = -torch.arange(1, state + 1, device=device).float().unsqueeze(0).expand(hidden, -1)
    B = torch.randn(batch, seq_len, state, device=device)
    C = torch.randn(batch, seq_len, state, device=device)

    # Warmup
    print("\nWarming up...")
    for _ in range(3):
        _ = selective_scan_reference(x, dt, A, B, C)
        if device == "cuda":
            torch.cuda.synchronize()

    # Benchmark reference
    print("Benchmarking PyTorch reference...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        _ = selective_scan_reference(x, dt, A, B, C)
        torch.cuda.synchronize()
    ref_time = (time.perf_counter() - start) / 10 * 1000
    print(f"  Reference: {ref_time:.2f} ms")

    # Benchmark Triton
    print("Benchmarking Triton kernel...")
    try:
        # Warmup
        for _ in range(3):
            _ = selective_scan_triton(x, dt, A, B, C)
            torch.cuda.synchronize()

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            _ = selective_scan_triton(x, dt, A, B, C)
            torch.cuda.synchronize()
        triton_time = (time.perf_counter() - start) / 10 * 1000
        print(f"  Triton: {triton_time:.2f} ms")
        print(f"  Speedup: {ref_time / triton_time:.2f}x")
    except Exception as e:
        print(f"  Triton failed: {e}")

    # 60 FPS check
    print(f"\n60 FPS threshold: 16.67 ms")
    print(f"Reference meets 60 FPS: {'YES' if ref_time < 16.67 else 'NO'}")


def test_correctness():
    """Test Triton kernel correctness against reference."""
    print("=" * 60)
    print("Correctness Test")
    print("=" * 60)

    device = "cuda"
    batch, seq_len, hidden, state = 2, 8, 4, 2

    x = torch.randn(batch, seq_len, hidden, device=device)
    dt = torch.rand(batch, seq_len, hidden, device=device) * 0.1
    A = -torch.arange(1, state + 1, device=device).float().unsqueeze(0).expand(hidden, -1)
    B = torch.randn(batch, seq_len, state, device=device)
    C = torch.randn(batch, seq_len, state, device=device)

    ref_out = selective_scan_reference(x, dt, A, B, C)

    try:
        triton_out = selective_scan_triton(x, dt, A, B, C)

        diff = (ref_out - triton_out).abs().max().item()
        print(f"Max absolute difference: {diff:.6f}")

        if diff < 1e-4:
            print("✓ PASSED")
        else:
            print("✗ FAILED - outputs differ")
            print(f"Reference:\n{ref_out[0, :4, :2]}")
            print(f"Triton:\n{triton_out[0, :4, :2]}")
    except Exception as e:
        print(f"Triton kernel failed: {e}")


# =============================================================================
# Elixir Integration Helper
# =============================================================================

def serve_elixir():
    """
    Simple msgpack-based server for Elixir Port integration.

    Protocol:
        Elixir sends: {op, tensors...} as msgpack
        Python returns: {ok, result} or {error, message}
    """
    import sys
    import struct

    try:
        import msgpack
    except ImportError:
        print("Install msgpack: pip install msgpack", file=sys.stderr)
        sys.exit(1)

    def read_message():
        # Read 4-byte length prefix
        header = sys.stdin.buffer.read(4)
        if len(header) < 4:
            return None
        length = struct.unpack(">I", header)[0]
        data = sys.stdin.buffer.read(length)
        return msgpack.unpackb(data, raw=False)

    def write_message(msg):
        data = msgpack.packb(msg, use_bin_type=True)
        sys.stdout.buffer.write(struct.pack(">I", len(data)))
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()

    print("Triton selective scan server started", file=sys.stderr)

    while True:
        msg = read_message()
        if msg is None:
            break

        op = msg.get("op")

        if op == "scan":
            # Unpack tensors from binary
            x = torch.frombuffer(msg["x"], dtype=torch.float32).reshape(msg["x_shape"]).cuda()
            dt = torch.frombuffer(msg["dt"], dtype=torch.float32).reshape(msg["dt_shape"]).cuda()
            A = torch.frombuffer(msg["A"], dtype=torch.float32).reshape(msg["A_shape"]).cuda()
            B = torch.frombuffer(msg["B"], dtype=torch.float32).reshape(msg["B_shape"]).cuda()
            C = torch.frombuffer(msg["C"], dtype=torch.float32).reshape(msg["C_shape"]).cuda()

            out = selective_scan_triton(x, dt, A, B, C)

            write_message({
                "status": "ok",
                "result": out.cpu().numpy().tobytes(),
                "shape": list(out.shape)
            })

        elif op == "ping":
            write_message({"status": "ok", "message": "pong"})

        else:
            write_message({"status": "error", "message": f"Unknown op: {op}"})


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton Selective Scan")
    parser.add_argument("--test", action="store_true", help="Run correctness tests")
    parser.add_argument("--profile", action="store_true", help="Profile kernel")
    parser.add_argument("--serve", action="store_true", help="Start Elixir integration server")
    args = parser.parse_args()

    if args.test:
        test_correctness()
    elif args.serve:
        serve_elixir()
    elif args.profile:
        # Use Triton's autotuner for profiling
        print("Profiling not yet implemented")
    else:
        benchmark()
