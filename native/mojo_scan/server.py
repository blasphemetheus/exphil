#!/usr/bin/env python3
"""
Mojo Linear Scan Server for Elixir Port Integration

Wraps the Mojo kernel via Python interop. Falls back to NumPy if Mojo
is not available (for testing the protocol without Mojo installed).

Protocol: identical to pytorch_scan_server.py
  Request:  4-byte big-endian length + msgpack
  Response: 4-byte big-endian length + msgpack

Usage:
  # Started by ExPhil.Bridge.MojoPort GenServer
  python3 native/mojo_scan/server.py
"""

import sys
import struct
import time
import numpy as np

try:
    import msgpack
except ImportError:
    print("ERROR: msgpack not installed. Run: pip install msgpack", file=sys.stderr)
    sys.exit(1)

# Try to import Mojo kernel
MOJO_AVAILABLE = False
try:
    # Mojo Python interop: import compiled .mojo as Python module
    sys.path.insert(0, ".")
    from linear_scan import run_linear_scan
    MOJO_AVAILABLE = True
    print("Mojo kernel loaded successfully", file=sys.stderr)
except ImportError:
    print("Mojo kernel not available, using NumPy fallback", file=sys.stderr)


def linear_scan_numpy(a_vals, b_vals, h0):
    """NumPy reference implementation (CPU, sequential)."""
    batch, seq_len, hidden = a_vals.shape
    output = np.empty_like(a_vals)

    for bi in range(batch):
        h_state = h0[bi].copy()
        for t in range(seq_len):
            h_state = a_vals[bi, t] * h_state + b_vals[bi, t]
            output[bi, t] = h_state

    return output


def linear_scan_numpy_vectorized(a_vals, b_vals, h0):
    """NumPy vectorized over batch and hidden (sequential over time)."""
    batch, seq_len, hidden = a_vals.shape
    output = np.empty_like(a_vals)
    h_state = h0.copy()

    for t in range(seq_len):
        h_state = a_vals[:, t, :] * h_state + b_vals[:, t, :]
        output[:, t, :] = h_state

    return output


class ScanServer:
    def __init__(self):
        self.use_mojo = MOJO_AVAILABLE
        mode = "mojo" if self.use_mojo else "numpy"
        print(f"Mojo scan server started (mode: {mode})", file=sys.stderr)

    def read_message(self):
        """Read length-prefixed msgpack message from stdin."""
        header = sys.stdin.buffer.read(4)
        if len(header) < 4:
            return None
        length = struct.unpack(">I", header)[0]
        data = sys.stdin.buffer.read(length)
        return msgpack.unpackb(data, raw=True, strict_map_key=False)

    def write_message(self, msg):
        """Write length-prefixed msgpack message to stdout."""
        data = msgpack.packb(msg, use_bin_type=True)
        sys.stdout.buffer.write(struct.pack(">I", len(data)))
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()

    def get_val(self, msg, key, default=None, raw=False):
        """Get value from msg, handling both str and bytes keys.

        Args:
            raw: If True, return bytes as-is (for binary tensor data).
                 If False, decode bytes to str for string fields.
        """
        for k in [key.encode(), key]:
            if k in msg:
                val = msg[k]
                if not raw and isinstance(val, bytes) and isinstance(default, (str, type(None))):
                    return val.decode()
                return val
        return default

    def handle_ping(self, msg):
        return {
            "status": "ok",
            "message": "pong",
            "device": "mojo" if self.use_mojo else "numpy",
            "mojo_available": self.use_mojo
        }

    def handle_info(self, msg):
        return {
            "status": "ok",
            "device": "mojo" if self.use_mojo else "numpy",
            "mojo_available": self.use_mojo,
            "numpy_version": np.__version__
        }

    def handle_scan(self, msg):
        try:
            batch = self.get_val(msg, "batch")
            seq_len = self.get_val(msg, "seq_len")
            hidden = self.get_val(msg, "hidden")
            mode = self.get_val(msg, "mode", "auto")

            a_bin = self.get_val(msg, "a", raw=True)
            b_bin = self.get_val(msg, "b", raw=True)
            h0_bin = self.get_val(msg, "h0", raw=True)

            # Convert to numpy arrays
            a_vals = np.frombuffer(bytearray(a_bin), dtype=np.float32).reshape(batch, seq_len, hidden)
            b_vals = np.frombuffer(bytearray(b_bin), dtype=np.float32).reshape(batch, seq_len, hidden)
            h0 = np.frombuffer(bytearray(h0_bin), dtype=np.float32).reshape(batch, hidden)

            if mode == "mojo" and self.use_mojo:
                # Call Mojo kernel via Python interop
                output = np.empty_like(a_vals)
                run_linear_scan(
                    a_vals.ctypes.data, b_vals.ctypes.data,
                    h0.ctypes.data, output.ctypes.data,
                    batch, seq_len, hidden, True
                )
                result = output
            elif mode == "numpy_vec":
                result = linear_scan_numpy_vectorized(a_vals, b_vals, h0)
            else:
                result = linear_scan_numpy_vectorized(a_vals, b_vals, h0)

            return {
                "status": "ok",
                "result": result.tobytes(),
                "shape": [batch, seq_len, hidden]
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def handle_benchmark(self, msg):
        try:
            batch = self.get_val(msg, "batch")
            seq_len = self.get_val(msg, "seq_len")
            hidden = self.get_val(msg, "hidden")
            warmup_iters = self.get_val(msg, "warmup", 5)
            timed_iters = self.get_val(msg, "iterations", 30)

            a_vals = np.random.uniform(0.5, 0.99, (batch, seq_len, hidden)).astype(np.float32)
            b_vals = np.random.randn(batch, seq_len, hidden).astype(np.float32)
            h0 = np.zeros((batch, hidden), dtype=np.float32)

            # Warmup
            for _ in range(warmup_iters):
                linear_scan_numpy_vectorized(a_vals, b_vals, h0)

            # Timed (numpy vectorized)
            numpy_times = []
            for _ in range(timed_iters):
                t0 = time.perf_counter_ns()
                linear_scan_numpy_vectorized(a_vals, b_vals, h0)
                numpy_times.append((time.perf_counter_ns() - t0) / 1e3)

            numpy_times.sort()

            result = {
                "status": "ok",
                "numpy_median_us": numpy_times[len(numpy_times) // 2],
                "numpy_min_us": numpy_times[0],
                "numpy_max_us": numpy_times[-1],
            }

            if self.use_mojo:
                output = np.empty_like(a_vals)
                for _ in range(warmup_iters):
                    run_linear_scan(
                        a_vals.ctypes.data, b_vals.ctypes.data,
                        h0.ctypes.data, output.ctypes.data,
                        batch, seq_len, hidden, True
                    )

                mojo_times = []
                for _ in range(timed_iters):
                    t0 = time.perf_counter_ns()
                    run_linear_scan(
                        a_vals.ctypes.data, b_vals.ctypes.data,
                        h0.ctypes.data, output.ctypes.data,
                        batch, seq_len, hidden, True
                    )
                    mojo_times.append((time.perf_counter_ns() - t0) / 1e3)

                mojo_times.sort()
                result["mojo_median_us"] = mojo_times[len(mojo_times) // 2]
                result["mojo_min_us"] = mojo_times[0]
                result["mojo_max_us"] = mojo_times[-1]

            return result

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def run(self):
        while True:
            msg = self.read_message()
            if msg is None:
                break

            op = msg.get(b"op", msg.get("op", b"unknown"))
            if isinstance(op, bytes):
                op = op.decode()

            if op == "scan":
                response = self.handle_scan(msg)
            elif op == "ping":
                response = self.handle_ping(msg)
            elif op == "info":
                response = self.handle_info(msg)
            elif op == "benchmark":
                response = self.handle_benchmark(msg)
            else:
                response = {"status": "error", "message": f"Unknown op: {op}"}

            self.write_message(response)


if __name__ == "__main__":
    server = ScanServer()
    server.run()
