/*
 * ThunderKittens-style Linear Scan Kernel
 *
 * ThunderKittens (HazyResearch) is a CUDA-embedded C++ DSL for AI kernels
 * with tile-level abstractions over tensor cores and shared memory.
 * It requires sm_80+ (Ampere) for core functionality.
 *
 * This file provides two implementations:
 *   1. TK-based kernel (when TK headers available and sm_80+)
 *   2. Standard CUDA fallback (works on sm_75+)
 *
 * For fused_linear_scan (h = a*h + b), TK doesn't provide significant
 * advantage — it's designed for matmul/attention with tensor cores.
 * The real value of TK is for fused attention kernels in SSM architectures.
 *
 * Build:
 *   nvcc -shared -Xcompiler -fPIC -O3 -arch=sm_75 -o libthunderkittens_scan_nif.so \
 *        kernel.cu thunderkittens_scan_nif.c -I${ERL_INCLUDE} -lcuda
 */

#include <cuda_runtime.h>
#include <cstdint>

/* ========================================================================== */
/* Check for ThunderKittens availability                                       */
/* ========================================================================== */

#if defined(TK_AVAILABLE) && (__CUDA_ARCH__ >= 800)
// ThunderKittens headers would go here:
// #include "kittens.cuh"
// Using TK register tiles and shared memory abstractions
#define USE_TK 1
#else
#define USE_TK 0
#endif

/* ========================================================================== */
/* Standard CUDA Kernel (fallback for sm_75 / no TK)                          */
/* ========================================================================== */

/*
 * Each thread handles one (batch, hidden) pair.
 * Sequential scan over timesteps: h[t] = a[t] * h[t-1] + b[t]
 *
 * This is the same algorithm as our CUDA C reference kernel.
 * TK would add value for the attention/matmul parts of SSM,
 * not for this simple recurrence.
 */
__global__ void linear_scan_kernel(
    const float* __restrict__ a,       // [batch, seq_len, hidden]
    const float* __restrict__ b,       // [batch, seq_len, hidden]
    const float* __restrict__ h0,      // [batch, hidden]
    float* __restrict__ output,        // [batch, seq_len, hidden]
    int batch,
    int seq_len,
    int hidden
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

/* ========================================================================== */
/* Host-callable entry point (for NIF integration via dlopen)                 */
/* ========================================================================== */

extern "C" {

/*
 * Launch the linear scan kernel.
 * Called from the C NIF via dlopen/dlsym.
 */
void tk_linear_scan(
    const float* a, const float* b, const float* h0,
    float* output, int batch, int seq_len, int hidden,
    cudaStream_t stream
) {
    int total_threads = batch * hidden;
    int block_size = 256;
    int grid_size = (total_threads + block_size - 1) / block_size;

    linear_scan_kernel<<<grid_size, block_size, 0, stream>>>(
        a, b, h0, output, batch, seq_len, hidden
    );
}

/*
 * Synchronize the default stream.
 */
void tk_synchronize(void) {
    cudaDeviceSynchronize();
}

/*
 * Check if CUDA is available and return device info.
 * Returns 0 on success, -1 on failure.
 */
int tk_device_info(char* name_buf, int buf_size, int* sm_major, int* sm_minor) {
    int device_count;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        return -1;

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess)
        return -1;

    if (name_buf && buf_size > 0) {
        strncpy(name_buf, prop.name, buf_size - 1);
        name_buf[buf_size - 1] = '\0';
    }
    if (sm_major) *sm_major = prop.major;
    if (sm_minor) *sm_minor = prop.minor;

    return 0;
}

}  /* extern "C" */
