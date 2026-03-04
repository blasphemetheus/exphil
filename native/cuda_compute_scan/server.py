#!/usr/bin/env python3
"""
NVIDIA CCCL / CuPy Linear Scan Server for Elixir Port Integration

Uses CuPy for GPU-accelerated linear scan with custom CUDA kernels.
Two GPU modes:
  1. Sequential scan (same algorithm as CUDA C, one thread per batch*hidden)
  2. Parallel prefix scan with associative operator (a1,b1)*(a2,b2) = (a1*a2, a1*b2+b1)

Falls back to NumPy on CPU if CuPy is not available.

Protocol: 4-byte big-endian length + msgpack (same as mojo_scan/server.py)

Usage:
  # Started by ExPhil.Bridge.CudaComputePort GenServer
  python3 native/cuda_compute_scan/server.py
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

# Try CuPy for GPU
CUPY_AVAILABLE = False
cp = None
try:
    import cupy
    cp = cupy
    # Verify GPU is accessible
    cp.cuda.Device(0).compute_capability
    CUPY_AVAILABLE = True
    print(f"CuPy {cp.__version__} loaded (GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()})", file=sys.stderr)
except Exception as e:
    print(f"CuPy not available: {e}, using NumPy fallback", file=sys.stderr)

# CUDA kernel: sequential scan (one thread per batch*hidden, loops over time)
SEQUENTIAL_SCAN_KERNEL = r"""
extern "C" __global__
void sequential_scan_kernel(
    const float* a, const float* b, const float* h0,
    float* output, int batch, int seq_len, int hidden
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * hidden;
    if (idx >= total) return;

    int bi = idx / hidden;
    int hi = idx % hidden;

    float h_state = h0[bi * hidden + hi];
    for (int t = 0; t < seq_len; t++) {
        int offset = bi * seq_len * hidden + t * hidden + hi;
        h_state = a[offset] * h_state + b[offset];
        output[offset] = h_state;
    }
}
"""

# CUDA kernel: parallel prefix scan using Blelloch algorithm
# Operates on pairs (a_coeff, b_additive) with operator:
#   (a1, b1) * (a2, b2) = (a1*a2, a2*b1 + b2)
# This computes the full scan in O(log T) parallel steps.
PARALLEL_SCAN_KERNEL = r"""
extern "C" __global__
void parallel_scan_kernel(
    const float* a_in, const float* b_in, const float* h0,
    float* output, int batch, int seq_len, int hidden
) {
    // Each block handles one (batch, hidden) pair
    int bi = blockIdx.x;
    int hi = blockIdx.y;
    if (bi >= batch || hi >= hidden) return;

    extern __shared__ float shared[];
    // shared layout: [2*seq_len] for a_coeffs, [2*seq_len] for b_additive
    float* sa = shared;
    float* sb = shared + 2 * seq_len;

    int tid = threadIdx.x;

    // Load input pairs into shared memory
    // Each pair is (a_coeff, b_additive)
    for (int t = tid; t < seq_len; t += blockDim.x) {
        int offset = bi * seq_len * hidden + t * hidden + hi;
        sa[t] = a_in[offset];
        sb[t] = b_in[offset];
    }
    __syncthreads();

    // Up-sweep (reduce) phase - Blelloch scan
    for (int stride = 1; stride < seq_len; stride *= 2) {
        for (int t = tid; t < seq_len; t += blockDim.x) {
            if (t >= stride) {
                int left = t - stride;
                // (sa[left], sb[left]) * (sa[t], sb[t])
                // = (sa[left]*sa[t], sa[t]*sb[left] + sb[t])
                float new_a = sa[left] * sa[t];
                float new_b = sa[t] * sb[left] + sb[t];
                // Need double buffering to avoid race
                sa[t + seq_len] = new_a;
                sb[t + seq_len] = new_b;
            } else {
                sa[t + seq_len] = sa[t];
                sb[t + seq_len] = sb[t];
            }
        }
        __syncthreads();
        // Swap buffers
        for (int t = tid; t < seq_len; t += blockDim.x) {
            sa[t] = sa[t + seq_len];
            sb[t] = sb[t + seq_len];
        }
        __syncthreads();
    }

    // After prefix scan, sa[t] and sb[t] hold the cumulative operator from 0..t
    // To get h[t], apply the scanned operator to h0:
    // h[t] = sa[t] * h0 + sb[t]
    float h0_val = h0[bi * hidden + hi];
    for (int t = tid; t < seq_len; t += blockDim.x) {
        int offset = bi * seq_len * hidden + t * hidden + hi;
        output[offset] = sa[t] * h0_val + sb[t];
    }
}
"""

# Compile kernels once
_sequential_kernel = None
_parallel_kernel = None

def get_sequential_kernel():
    global _sequential_kernel
    if _sequential_kernel is None and CUPY_AVAILABLE:
        _sequential_kernel = cp.RawKernel(SEQUENTIAL_SCAN_KERNEL, "sequential_scan_kernel")
    return _sequential_kernel

def get_parallel_kernel():
    global _parallel_kernel
    if _parallel_kernel is None and CUPY_AVAILABLE:
        _parallel_kernel = cp.RawKernel(PARALLEL_SCAN_KERNEL, "parallel_scan_kernel")
    return _parallel_kernel


def linear_scan_numpy(a_vals, b_vals, h0):
    """NumPy reference (CPU, vectorized over batch/hidden)."""
    batch, seq_len, hidden = a_vals.shape
    output = np.empty_like(a_vals)
    h_state = h0.copy()
    for t in range(seq_len):
        h_state = a_vals[:, t, :] * h_state + b_vals[:, t, :]
        output[:, t, :] = h_state
    return output


def linear_scan_cupy_sequential(a_vals, b_vals, h0):
    """CuPy sequential scan — same algorithm as CUDA C kernel."""
    kernel = get_sequential_kernel()
    batch, seq_len, hidden = a_vals.shape

    a_gpu = cp.asarray(a_vals)
    b_gpu = cp.asarray(b_vals)
    h0_gpu = cp.asarray(h0)
    output_gpu = cp.empty_like(a_gpu)

    total_threads = batch * hidden
    block_size = 256
    grid_size = (total_threads + block_size - 1) // block_size

    kernel((grid_size,), (block_size,),
           (a_gpu, b_gpu, h0_gpu, output_gpu,
            np.int32(batch), np.int32(seq_len), np.int32(hidden)))

    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(output_gpu)


def linear_scan_cupy_parallel(a_vals, b_vals, h0):
    """CuPy parallel prefix scan — O(log T) depth Blelloch algorithm."""
    kernel = get_parallel_kernel()
    batch, seq_len, hidden = a_vals.shape

    a_gpu = cp.asarray(a_vals)
    b_gpu = cp.asarray(b_vals)
    h0_gpu = cp.asarray(h0)
    output_gpu = cp.empty_like(a_gpu)

    # Each block handles one (batch, hidden) pair
    block_size = min(seq_len, 256)
    # Shared memory: 2 arrays of 2*seq_len floats (double buffer)
    shared_mem = 2 * 2 * seq_len * 4  # bytes

    kernel((batch, hidden), (block_size,),
           (a_gpu, b_gpu, h0_gpu, output_gpu,
            np.int32(batch), np.int32(seq_len), np.int32(hidden)),
           shared_mem=shared_mem)

    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(output_gpu)


class CudaComputeScanServer:
    def __init__(self):
        self.has_gpu = CUPY_AVAILABLE
        mode = "cupy_gpu" if self.has_gpu else "numpy_cpu"
        print(f"CCCL scan server started (mode: {mode})", file=sys.stderr)

    def read_message(self):
        header = sys.stdin.buffer.read(4)
        if len(header) < 4:
            return None
        length = struct.unpack(">I", header)[0]
        data = sys.stdin.buffer.read(length)
        return msgpack.unpackb(data, raw=True, strict_map_key=False)

    def write_message(self, msg):
        data = msgpack.packb(msg, use_bin_type=True)
        sys.stdout.buffer.write(struct.pack(">I", len(data)))
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()

    def get_val(self, msg, key, default=None, raw=False):
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
            "device": "cupy_gpu" if self.has_gpu else "numpy_cpu",
            "cupy_available": self.has_gpu
        }

    def handle_info(self, msg):
        info = {
            "status": "ok",
            "device": "cupy_gpu" if self.has_gpu else "numpy_cpu",
            "cupy_available": self.has_gpu,
            "numpy_version": np.__version__,
            "modes": ["sequential", "parallel", "numpy"]
        }
        if self.has_gpu:
            info["cupy_version"] = cp.__version__
            props = cp.cuda.runtime.getDeviceProperties(0)
            info["gpu_name"] = props["name"].decode()
            info["compute_capability"] = f"{props['major']}.{props['minor']}"
        return info

    def handle_scan(self, msg):
        try:
            batch = self.get_val(msg, "batch")
            seq_len = self.get_val(msg, "seq_len")
            hidden = self.get_val(msg, "hidden")
            mode = self.get_val(msg, "mode", "auto")

            a_bin = self.get_val(msg, "a", raw=True)
            b_bin = self.get_val(msg, "b", raw=True)
            h0_bin = self.get_val(msg, "h0", raw=True)

            a_vals = np.frombuffer(bytearray(a_bin), dtype=np.float32).reshape(batch, seq_len, hidden)
            b_vals = np.frombuffer(bytearray(b_bin), dtype=np.float32).reshape(batch, seq_len, hidden)
            h0 = np.frombuffer(bytearray(h0_bin), dtype=np.float32).reshape(batch, hidden)

            if mode == "parallel" and self.has_gpu:
                result = linear_scan_cupy_parallel(a_vals, b_vals, h0)
            elif mode == "sequential" and self.has_gpu:
                result = linear_scan_cupy_sequential(a_vals, b_vals, h0)
            elif mode == "auto" and self.has_gpu:
                result = linear_scan_cupy_sequential(a_vals, b_vals, h0)
            else:
                result = linear_scan_numpy(a_vals, b_vals, h0)

            return {
                "status": "ok",
                "result": result.tobytes(),
                "shape": [batch, seq_len, hidden],
                "mode_used": mode if mode != "auto" else ("cupy_sequential" if self.has_gpu else "numpy")
            }

        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
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

            result = {"status": "ok"}

            # NumPy CPU baseline
            for _ in range(warmup_iters):
                linear_scan_numpy(a_vals, b_vals, h0)

            numpy_times = []
            for _ in range(timed_iters):
                t0 = time.perf_counter_ns()
                linear_scan_numpy(a_vals, b_vals, h0)
                numpy_times.append((time.perf_counter_ns() - t0) / 1e3)

            numpy_times.sort()
            result["numpy_median_us"] = numpy_times[len(numpy_times) // 2]

            if self.has_gpu:
                # CuPy sequential
                for _ in range(warmup_iters):
                    linear_scan_cupy_sequential(a_vals, b_vals, h0)

                seq_times = []
                for _ in range(timed_iters):
                    t0 = time.perf_counter_ns()
                    linear_scan_cupy_sequential(a_vals, b_vals, h0)
                    seq_times.append((time.perf_counter_ns() - t0) / 1e3)

                seq_times.sort()
                result["sequential_median_us"] = seq_times[len(seq_times) // 2]

                # CuPy parallel prefix scan
                for _ in range(warmup_iters):
                    linear_scan_cupy_parallel(a_vals, b_vals, h0)

                par_times = []
                for _ in range(timed_iters):
                    t0 = time.perf_counter_ns()
                    linear_scan_cupy_parallel(a_vals, b_vals, h0)
                    par_times.append((time.perf_counter_ns() - t0) / 1e3)

                par_times.sort()
                result["parallel_median_us"] = par_times[len(par_times) // 2]

            return result

        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
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
    server = CudaComputeScanServer()
    server.run()
