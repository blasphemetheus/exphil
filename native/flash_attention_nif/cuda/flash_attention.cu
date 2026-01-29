/**
 * FlashAttention-2 Forward Pass CUDA Kernel
 *
 * This is a simplified implementation based on flash-attention-minimal.
 * For production, consider using the full FlashAttention-2 from Dao-AILab.
 *
 * Reference:
 * - https://github.com/tspeterkim/flash-attention-minimal
 * - https://arxiv.org/abs/2307.08691 (FlashAttention-2 paper)
 *
 * Requires: CUDA 12.0+, Ampere+ GPU (sm_80+)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>

// Block sizes for tiled computation
constexpr int BLOCK_SIZE = 32;

// Check CUDA errors
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                    \
            return -1;                                                           \
        }                                                                        \
    } while (0)

/**
 * Flash Attention Forward Kernel (Simplified)
 *
 * Uses tiled computation to avoid materializing full attention matrix.
 * This version is for educational purposes - production should use
 * the full FlashAttention-2 implementation.
 *
 * Layout: [batch, seq_len, num_heads, head_dim]
 */
__global__ void flash_attention_kernel_f32(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const int causal,
    const float softmax_scale
) {
    // Block and thread indices
    const int batch_head_idx = blockIdx.x;  // Combined batch and head index
    const int query_idx = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    if (query_idx >= seq_len) return;

    // Calculate batch and head from combined index
    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;

    // Pointers to this batch/head's data
    const int head_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    const float* q_head = Q + head_offset;
    const float* k_head = K + head_offset;
    const float* v_head = V + head_offset;
    float* o_head = O + head_offset;

    // Shared memory for K and V tiles
    extern __shared__ float shared_mem[];
    float* s_k = shared_mem;                           // [BLOCK_SIZE, head_dim]
    float* s_v = shared_mem + BLOCK_SIZE * head_dim;   // [BLOCK_SIZE, head_dim]

    // Registers for this query's computation
    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    float out[64];  // Assume head_dim <= 64 for register storage

    for (int d = 0; d < head_dim; d++) {
        out[d] = 0.0f;
    }

    // Load query for this thread (each thread handles one query position)
    float q_local[64];
    for (int d = 0; d < head_dim; d++) {
        q_local[d] = q_head[query_idx * head_dim + d];
    }

    // Process K/V in tiles
    const int num_kv_tiles = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int tile_idx = 0; tile_idx < num_kv_tiles; tile_idx++) {
        const int kv_start = tile_idx * BLOCK_SIZE;
        const int kv_end = min(kv_start + BLOCK_SIZE, seq_len);
        const int tile_size = kv_end - kv_start;

        // Collaboratively load K and V tiles to shared memory
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            for (int j = threadIdx.y; j < tile_size; j += blockDim.y) {
                const int kv_idx = kv_start + j;
                s_k[j * head_dim + d] = k_head[kv_idx * head_dim + d];
                s_v[j * head_dim + d] = v_head[kv_idx * head_dim + d];
            }
        }
        __syncthreads();

        // Compute attention scores for this tile
        for (int j = 0; j < tile_size; j++) {
            const int kv_idx = kv_start + j;

            // Apply causal mask
            if (causal && kv_idx > query_idx) continue;

            // Compute dot product Q[i] . K[j]
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += q_local[d] * s_k[j * head_dim + d];
            }
            score *= softmax_scale;

            // Online softmax update
            float new_max = fmaxf(max_val, score);
            float old_scale = expf(max_val - new_max);
            float new_scale = expf(score - new_max);

            // Rescale running sum
            sum_exp = sum_exp * old_scale + new_scale;
            max_val = new_max;

            // Update output with rescaled old values + new contribution
            for (int d = 0; d < head_dim; d++) {
                out[d] = out[d] * old_scale + new_scale * s_v[j * head_dim + d];
            }
        }

        __syncthreads();
    }

    // Final normalization
    if (sum_exp > 0.0f) {
        for (int d = 0; d < head_dim; d++) {
            o_head[query_idx * head_dim + d] = out[d] / sum_exp;
        }
    }
}

// ============================================================================
// C Interface Functions
// ============================================================================

extern "C" {

/**
 * Check if CUDA is available and has compatible GPU
 */
int flash_attention_cuda_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess || device_count == 0) {
        return 0;
    }

    // Check for Ampere+ (compute capability >= 8.0)
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    if (prop.major < 8) {
        fprintf(stderr, "FlashAttention requires Ampere+ GPU (compute >= 8.0), "
                        "found %d.%d\n", prop.major, prop.minor);
        return 0;
    }

    return 1;
}

/**
 * Initialize CUDA for flash attention
 */
int flash_attention_cuda_init() {
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        return -1;
    }

    // Warm up GPU
    cudaFree(0);

    return 0;
}

/**
 * Flash attention forward pass (f32)
 *
 * Layout: [batch, seq_len, num_heads, head_dim] (row-major)
 */
int flash_attention_forward_f32(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    int batch,
    int seq_len,
    int num_heads,
    int head_dim,
    int causal,
    float softmax_scale
) {
    // Allocate device memory
    const size_t tensor_size = batch * seq_len * num_heads * head_dim * sizeof(float);

    float *d_q, *d_k, *d_v, *d_output;
    CUDA_CHECK(cudaMalloc(&d_q, tensor_size));
    CUDA_CHECK(cudaMalloc(&d_k, tensor_size));
    CUDA_CHECK(cudaMalloc(&d_v, tensor_size));
    CUDA_CHECK(cudaMalloc(&d_output, tensor_size));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_q, q, tensor_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, k, tensor_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, v, tensor_size, cudaMemcpyHostToDevice));

    // Calculate grid and block dimensions
    const int num_batch_heads = batch * num_heads;
    const int num_query_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 grid(num_batch_heads, num_query_blocks);
    dim3 block(min(32, head_dim), min(BLOCK_SIZE, seq_len));

    // Shared memory for K and V tiles
    const size_t shared_mem_size = 2 * BLOCK_SIZE * head_dim * sizeof(float);

    // Launch kernel
    flash_attention_kernel_f32<<<grid, block, shared_mem_size>>>(
        d_q, d_k, d_v, d_output,
        seq_len, num_heads, head_dim,
        causal, softmax_scale
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy output back to host
    CUDA_CHECK(cudaMemcpy(output, d_output, tensor_size, cudaMemcpyDeviceToHost));

    // Free device memory
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_output);

    return 0;
}

}  // extern "C"
