#!/usr/bin/env python3
"""
PyTorch Selective Scan Server for Elixir Port Integration

This server receives tensor data from Elixir, runs the selective scan
on GPU using PyTorch, and returns the results.

Protocol (msgpack over length-prefixed frames):
  Request:  {op: "scan", x: bytes, dt: bytes, ...shapes...}
  Response: {status: "ok", result: bytes, shape: [b, l, h]} or {status: "error", message: str}

Usage:
  # Started automatically by ExPhil.Bridge.PyTorchPort
  python3 priv/python/pytorch_scan_server.py
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


def selective_scan_pytorch(x, dt, A, B, C, dt_min=0.001, dt_max=0.1):
    """
    Fast selective scan using PyTorch on GPU.

    This simple sequential implementation achieves ~5ms on RTX 4090,
    which is 10x faster than Nx/XLA's parallel scan (~55ms).

    Args:
        x: Input [batch, seq_len, hidden], f32
        dt: Delta [batch, seq_len, hidden], f32
        A: State transition [hidden, state], f32
        B: Input projection [batch, seq_len, state], f32
        C: Output projection [batch, seq_len, state], f32

    Returns:
        y: Output [batch, seq_len, hidden], f32
    """
    batch, seq_len, hidden = x.shape
    state = A.shape[1]
    device = x.device

    # Clamp dt
    dt = dt.clamp(dt_min, dt_max)

    # Initialize hidden state
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


class ScanServer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"PyTorch scan server started on {self.device}", file=sys.stderr)

        # Warmup
        if self.device == "cuda":
            self._warmup()

    def _warmup(self):
        """Warmup GPU with a small scan."""
        x = torch.randn(1, 10, 64, device=self.device)
        dt = torch.rand(1, 10, 64, device=self.device) * 0.1
        A = -torch.arange(1, 17, device=self.device).float().unsqueeze(0).expand(64, -1)
        B = torch.randn(1, 10, 16, device=self.device)
        C = torch.randn(1, 10, 16, device=self.device)
        _ = selective_scan_pytorch(x, dt, A, B, C)
        torch.cuda.synchronize()
        print("GPU warmup complete", file=sys.stderr)

    def read_message(self):
        """Read length-prefixed msgpack message from stdin."""
        header = sys.stdin.buffer.read(4)
        if len(header) < 4:
            return None
        length = struct.unpack(">I", header)[0]
        data = sys.stdin.buffer.read(length)
        return msgpack.unpackb(data, raw=False)

    def write_message(self, msg):
        """Write length-prefixed msgpack message to stdout."""
        data = msgpack.packb(msg, use_bin_type=True)
        sys.stdout.buffer.write(struct.pack(">I", len(data)))
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()

    def handle_scan(self, msg):
        """Handle a scan request."""
        try:
            # Extract shapes
            batch = msg["batch"]
            seq_len = msg["seq_len"]
            hidden = msg["hidden"]
            state = msg["state"]

            # Convert bytes to tensors
            x = torch.frombuffer(bytearray(msg["x"]), dtype=torch.float32).reshape(batch, seq_len, hidden).to(self.device)
            dt = torch.frombuffer(bytearray(msg["dt"]), dtype=torch.float32).reshape(batch, seq_len, hidden).to(self.device)
            A = torch.frombuffer(bytearray(msg["A"]), dtype=torch.float32).reshape(hidden, state).to(self.device)
            B = torch.frombuffer(bytearray(msg["B"]), dtype=torch.float32).reshape(batch, seq_len, state).to(self.device)
            C = torch.frombuffer(bytearray(msg["C"]), dtype=torch.float32).reshape(batch, seq_len, state).to(self.device)

            # Run scan
            result = selective_scan_pytorch(x, dt, A, B, C)

            # Sync and convert back
            if self.device == "cuda":
                torch.cuda.synchronize()

            result_bytes = result.cpu().numpy().tobytes()

            return {
                "status": "ok",
                "result": result_bytes,
                "shape": list(result.shape)
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    def handle_ping(self, msg):
        """Handle ping request."""
        return {
            "status": "ok",
            "message": "pong",
            "device": self.device,
            "cuda_available": torch.cuda.is_available()
        }

    def handle_info(self, msg):
        """Handle info request."""
        info = {
            "status": "ok",
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "pytorch_version": torch.__version__,
        }
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        return info

    def run(self):
        """Main server loop."""
        while True:
            msg = self.read_message()
            if msg is None:
                break

            op = msg.get("op", "unknown")

            if op == "scan":
                response = self.handle_scan(msg)
            elif op == "ping":
                response = self.handle_ping(msg)
            elif op == "info":
                response = self.handle_info(msg)
            else:
                response = {"status": "error", "message": f"Unknown op: {op}"}

            self.write_message(response)


if __name__ == "__main__":
    server = ScanServer()
    server.run()
