#!/usr/bin/env python3
"""
FlashAttention Server for Elixir Port Integration

This server receives tensor data from Elixir, runs FlashAttention-2
on GPU using PyTorch, and returns the results.

Protocol (msgpack over length-prefixed frames):
  Request:  {op: "forward", q: bytes, k: bytes, v: bytes, ...shapes..., causal: bool}
  Response: {status: "ok", result: bytes, shape: [b, s, d]} or {status: "error", message: str}

Requirements:
  pip install flash-attn --no-build-isolation
  # Requires CUDA 11.8+ and PyTorch 2.0+

Usage:
  # Started automatically by ExPhil.Bridge.FlashAttentionPort
  python3 priv/python/flash_attention_server.py
"""

import sys
import struct
import torch
import numpy as np

try:
    import msgpack
except ImportError:
    print("ERROR: msgpack not installed. Run: pip install msgpack", file=sys.stderr)
    sys.exit(1)

# Try to import flash_attn
FLASH_ATTN_AVAILABLE = False
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    print("WARNING: flash_attn not installed. Install with: pip install flash-attn --no-build-isolation", file=sys.stderr)


def flash_attention_forward(q, k, v, causal=True, softmax_scale=None):
    """
    Run FlashAttention-2 forward pass.

    Args:
        q: Query tensor [batch, seq_len, num_heads, head_dim], f16/bf16
        k: Key tensor [batch, seq_len, num_heads, head_dim], f16/bf16
        v: Value tensor [batch, seq_len, num_heads, head_dim], f16/bf16
        causal: Whether to apply causal masking
        softmax_scale: Scaling factor (default: 1/sqrt(head_dim))

    Returns:
        output: [batch, seq_len, num_heads, head_dim], same dtype as input
    """
    if not FLASH_ATTN_AVAILABLE:
        raise RuntimeError("flash_attn not installed")

    # flash_attn expects [batch, seqlen, nheads, headdim]
    # Our input is [batch, seq_len, dim] - need to reshape
    # Actually, we'll handle the reshape in Elixir to keep this simple

    return flash_attn_func(q, k, v, causal=causal, softmax_scale=softmax_scale)


def standard_attention_forward(q, k, v, causal=True):
    """
    Standard PyTorch attention as fallback.

    Args:
        q, k, v: [batch, seq_len, dim] tensors
        causal: Whether to apply causal masking

    Returns:
        output: [batch, seq_len, dim]
    """
    batch, seq_len, dim = q.shape
    scale = dim ** -0.5

    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Apply causal mask if needed
    if causal:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))

    # Softmax and weighted sum
    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, v)

    return output


def get_key(request, key, default=None):
    """Get a value from request, handling both string and bytes keys."""
    # Try string key first, then bytes key
    if key in request:
        return request[key]
    bytes_key = key.encode() if isinstance(key, str) else key
    if bytes_key in request:
        return request[bytes_key]
    return default


def handle_request(request):
    """Process a request and return response."""
    op = get_key(request, "op")

    # Handle both string and bytes op values
    if op in ("ping", b"ping"):
        return {"status": "ok", "message": "pong", "flash_attn": FLASH_ATTN_AVAILABLE}

    elif op in ("info", b"info"):
        return {
            "status": "ok",
            "flash_attn_available": FLASH_ATTN_AVAILABLE,
            "cuda_available": torch.cuda.is_available(),
            "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu",
            "pytorch_version": torch.__version__
        }

    elif op == "forward" or op == b"forward":
        try:
            # Extract shapes
            batch = get_key(request, "batch")
            seq_len = get_key(request, "seq_len")
            dim = get_key(request, "dim")
            causal = get_key(request, "causal", True)
            use_flash = get_key(request, "use_flash", True) and FLASH_ATTN_AVAILABLE

            # Reconstruct tensors from binary
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if use_flash else torch.float32

            q = torch.frombuffer(bytearray(get_key(request, "q")), dtype=torch.float32).reshape(batch, seq_len, dim)
            k = torch.frombuffer(bytearray(get_key(request, "k")), dtype=torch.float32).reshape(batch, seq_len, dim)
            v = torch.frombuffer(bytearray(get_key(request, "v")), dtype=torch.float32).reshape(batch, seq_len, dim)

            # Move to device
            q = q.to(device=device)
            k = k.to(device=device)
            v = v.to(device=device)

            if use_flash:
                # FlashAttention requires [batch, seq, nheads, headdim] and fp16/bf16
                # For simplicity, we treat dim as nheads=1, headdim=dim
                # A more complete implementation would have num_heads parameter
                num_heads = get_key(request, "num_heads", 1)
                head_dim = dim // num_heads

                q = q.reshape(batch, seq_len, num_heads, head_dim).to(dtype)
                k = k.reshape(batch, seq_len, num_heads, head_dim).to(dtype)
                v = v.reshape(batch, seq_len, num_heads, head_dim).to(dtype)

                output = flash_attention_forward(q, k, v, causal=causal)
                output = output.reshape(batch, seq_len, dim).float()
            else:
                # Fallback to standard attention
                output = standard_attention_forward(q, k, v, causal=causal)

            # Convert back to binary
            result_bin = output.cpu().numpy().astype(np.float32).tobytes()

            return {
                "status": "ok",
                "result": result_bin,
                "shape": [batch, seq_len, dim],
                "used_flash": use_flash
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    elif op in ("benchmark", b"benchmark"):
        # Quick benchmark to compare flash vs standard
        try:
            batch = get_key(request, "batch", 4)
            seq_len = get_key(request, "seq_len", 128)
            dim = get_key(request, "dim", 256)
            num_iters = get_key(request, "num_iters", 10)

            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Create test tensors
            q = torch.randn(batch, seq_len, dim, device=device, dtype=torch.float32)
            k = torch.randn(batch, seq_len, dim, device=device, dtype=torch.float32)
            v = torch.randn(batch, seq_len, dim, device=device, dtype=torch.float32)

            # Benchmark standard attention
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            import time

            start = time.perf_counter()
            for _ in range(num_iters):
                _ = standard_attention_forward(q, k, v, causal=True)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            standard_time = (time.perf_counter() - start) / num_iters * 1000

            # Benchmark flash attention if available
            flash_time = None
            if FLASH_ATTN_AVAILABLE:
                num_heads = 4
                head_dim = dim // num_heads
                q_flash = q.reshape(batch, seq_len, num_heads, head_dim).half()
                k_flash = k.reshape(batch, seq_len, num_heads, head_dim).half()
                v_flash = v.reshape(batch, seq_len, num_heads, head_dim).half()

                # Warmup
                _ = flash_attention_forward(q_flash, k_flash, v_flash, causal=True)
                torch.cuda.synchronize()

                start = time.perf_counter()
                for _ in range(num_iters):
                    _ = flash_attention_forward(q_flash, k_flash, v_flash, causal=True)
                torch.cuda.synchronize()
                flash_time = (time.perf_counter() - start) / num_iters * 1000

            return {
                "status": "ok",
                "standard_ms": standard_time,
                "flash_ms": flash_time,
                "speedup": standard_time / flash_time if flash_time else None,
                "config": {"batch": batch, "seq_len": seq_len, "dim": dim}
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    else:
        return {"status": "error", "message": f"Unknown operation: {op}"}


def main():
    """Main server loop."""
    print(f"FlashAttention server starting (flash_attn={FLASH_ATTN_AVAILABLE})", file=sys.stderr)

    while True:
        try:
            # Read length-prefixed message
            length_bytes = sys.stdin.buffer.read(4)
            if not length_bytes or len(length_bytes) < 4:
                break

            length = struct.unpack(">I", length_bytes)[0]
            data = sys.stdin.buffer.read(length)

            if len(data) < length:
                break

            # Decode and process
            # raw=True keeps binary data as bytes instead of trying to decode as UTF-8
            request = msgpack.unpackb(data, raw=True, strict_map_key=False)
            response = handle_request(request)

            # Encode and send response
            response_bytes = msgpack.packb(response, use_bin_type=True)
            sys.stdout.buffer.write(struct.pack(">I", len(response_bytes)))
            sys.stdout.buffer.write(response_bytes)
            sys.stdout.buffer.flush()

        except Exception as e:
            # Send error response
            error_response = {"status": "error", "message": str(e)}
            response_bytes = msgpack.packb(error_response, use_bin_type=True)
            sys.stdout.buffer.write(struct.pack(">I", len(response_bytes)))
            sys.stdout.buffer.write(response_bytes)
            sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
