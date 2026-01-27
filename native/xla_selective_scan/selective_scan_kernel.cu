// XLA Custom Call kernel for Mamba selective scan
//
// This kernel performs the selective scan operation entirely on GPU,
// avoiding any CPU-GPU data transfer when called from EXLA.
//
// Build: nvcc -shared -o libxla_selective_scan.so selective_scan_kernel.cu -Xcompiler -fPIC
//
// The XLA CustomCall interface expects:
//   void custom_call(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len)

#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

// Shape parameters packed in opaque data
struct ScanParams {
    int32_t batch;
    int32_t seq_len;
    int32_t hidden;
    int32_t state;
    float dt_min;
    float dt_max;
};

// =============================================================================
// CUDA Kernel
// =============================================================================

__global__ void selective_scan_kernel(
    const float* __restrict__ x,      // [batch, seq_len, hidden]
    const float* __restrict__ dt,     // [batch, seq_len, hidden]
    const float* __restrict__ A,      // [hidden, state]
    const float* __restrict__ B,      // [batch, seq_len, state]
    const float* __restrict__ C,      // [batch, seq_len, state]
    float* __restrict__ out,          // [batch, seq_len, hidden]
    int batch,
    int seq_len,
    int hidden,
    int state,
    float dt_min,
    float dt_max
) {
    // Each thread handles one (batch, hidden) pair
    int b = blockIdx.x;
    int h = threadIdx.x + blockIdx.y * blockDim.x;

    if (b >= batch || h >= hidden) return;

    // Initialize hidden state (max state size = 32)
    float h_state[32];
    for (int s = 0; s < state && s < 32; s++) {
        h_state[s] = 0.0f;
    }

    // Load A diagonal for this hidden dim
    float A_diag[32];
    for (int s = 0; s < state && s < 32; s++) {
        A_diag[s] = A[h * state + s];
    }

    // Scan through sequence
    for (int t = 0; t < seq_len; t++) {
        // Load inputs
        int x_idx = b * seq_len * hidden + t * hidden + h;
        float x_t = x[x_idx];
        float dt_t = dt[x_idx];

        // Clamp dt
        dt_t = fminf(fmaxf(dt_t, dt_min), dt_max);

        // Load B and C indices
        int bc_idx = b * seq_len * state + t * state;

        // Compute output
        float y_t = 0.0f;

        for (int s = 0; s < state && s < 32; s++) {
            // Discretize: A_bar = exp(dt * A)
            float A_bar = expf(dt_t * A_diag[s]);
            float B_bar = dt_t * B[bc_idx + s];
            float C_s = C[bc_idx + s];

            // Recurrence: h = A_bar * h + B_bar * x
            h_state[s] = A_bar * h_state[s] + B_bar * x_t;

            // Accumulate output: y = sum(C * h)
            y_t += C_s * h_state[s];
        }

        // Store output
        out[x_idx] = y_t;
    }
}

// =============================================================================
// XLA CustomCall Entry Point
// =============================================================================

extern "C" {

// XLA CustomCall interface
// buffers[0-4] are inputs (x, dt, A, B, C)
// buffers[5] is output
void SelectiveScan(cudaStream_t stream, void** buffers,
                   const char* opaque, size_t opaque_len) {
    // Parse parameters from opaque data
    ScanParams params;
    if (opaque_len >= sizeof(ScanParams)) {
        memcpy(&params, opaque, sizeof(ScanParams));
    } else {
        // Default parameters if opaque not provided
        params.batch = 32;
        params.seq_len = 60;
        params.hidden = 512;
        params.state = 16;
        params.dt_min = 0.001f;
        params.dt_max = 0.1f;
    }

    // Extract buffer pointers
    const float* x = static_cast<const float*>(buffers[0]);
    const float* dt = static_cast<const float*>(buffers[1]);
    const float* A = static_cast<const float*>(buffers[2]);
    const float* B = static_cast<const float*>(buffers[3]);
    const float* C = static_cast<const float*>(buffers[4]);
    float* out = static_cast<float*>(buffers[5]);

    // Configure kernel launch
    int threads_per_block = min(params.hidden, 256);
    int blocks_y = (params.hidden + threads_per_block - 1) / threads_per_block;

    dim3 grid(params.batch, blocks_y, 1);
    dim3 block(threads_per_block, 1, 1);

    // Launch kernel on the provided stream
    selective_scan_kernel<<<grid, block, 0, stream>>>(
        x, dt, A, B, C, out,
        params.batch, params.seq_len, params.hidden, params.state,
        params.dt_min, params.dt_max
    );
}

}  // extern "C"
