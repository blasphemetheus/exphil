"""
Triton kernel for linear scan: h[t] = a[t] * h[t-1] + b[t]

This kernel implements the same fused_linear_scan as the CUDA C reference,
but written in Triton's Python DSL. Each program instance handles one
(batch, hidden) pair and loops over timesteps sequentially.

Used for benchmarking Triton AOT vs CUDA C, Rust-CUDA, Futhark, etc.
"""

import triton
import triton.language as tl


@triton.jit
def linear_scan_kernel(
    a_ptr,       # [batch, seq_len, hidden] - decay coefficients
    b_ptr,       # [batch, seq_len, hidden] - additive terms
    h0_ptr,      # [batch, hidden] - initial hidden state
    output_ptr,  # [batch, seq_len, hidden] - output
    batch,       # int
    seq_len,     # int
    hidden,      # int
):
    """Linear scan: h[t] = a[t] * h[t-1] + b[t], one program per (batch, hidden)."""
    # Program ID maps to (batch_idx, hidden_idx)
    pid = tl.program_id(0)
    batch_idx = pid // hidden
    hidden_idx = pid % hidden

    if batch_idx >= batch:
        return

    # Load initial state
    h0_offset = batch_idx * hidden + hidden_idx
    h_state = tl.load(h0_ptr + h0_offset)

    # Sequential scan over timesteps
    for t in range(seq_len):
        idx = batch_idx * seq_len * hidden + t * hidden + hidden_idx
        a_t = tl.load(a_ptr + idx)
        b_t = tl.load(b_ptr + idx)

        # Linear recurrence
        h_state = a_t * h_state + b_t

        # Store output
        tl.store(output_ptr + idx, h_state)
