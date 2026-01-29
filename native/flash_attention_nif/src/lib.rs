//! FlashAttention-2 Forward Pass NIF
//!
//! Provides GPU-accelerated flash attention for inference. This is forward-only
//! (no gradients) and intended for real-time inference (e.g., Dolphin play).
//!
//! ## Features
//!
//! - `cuda`: Enable CUDA kernel support (requires CUDA toolkit + Ampere+ GPU)
//!
//! ## Fallback
//!
//! When CUDA is unavailable, falls back to a CPU implementation that is
//! mathematically equivalent but slower.

use rustler::{Atom, Binary, Env, NifResult, OwnedBinary};
use std::sync::atomic::{AtomicBool, Ordering};

mod atoms {
    rustler::atoms! {
        ok,
        error,
        not_available,
        cuda_error,
        shape_mismatch,
        invalid_dtype,
    }
}

// Track if CUDA is initialized and available
static CUDA_AVAILABLE: AtomicBool = AtomicBool::new(false);
static CUDA_CHECKED: AtomicBool = AtomicBool::new(false);

// ============================================================================
// CUDA FFI Declarations (when cuda feature enabled)
// ============================================================================

#[cfg(feature = "cuda")]
extern "C" {
    fn flash_attention_cuda_available() -> i32;
    fn flash_attention_cuda_init() -> i32;

    fn flash_attention_forward_f32(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        output: *mut f32,
        batch: i32,
        seq_len: i32,
        num_heads: i32,
        head_dim: i32,
        causal: i32,
        softmax_scale: f32,
    ) -> i32;

    fn flash_attention_forward_f16(
        q: *const u16,  // f16 as u16
        k: *const u16,
        v: *const u16,
        output: *mut u16,
        batch: i32,
        seq_len: i32,
        num_heads: i32,
        head_dim: i32,
        causal: i32,
        softmax_scale: f32,
    ) -> i32;
}

// ============================================================================
// CPU Fallback Implementation
// ============================================================================

/// Standard attention on CPU (for testing and fallback)
///
/// Computes: softmax(QK^T / sqrt(d)) * V
fn attention_forward_cpu_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    causal: bool,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let head_size = seq_len * head_dim;
    let batch_size = num_heads * head_size;

    // Process each batch and head independently
    for b in 0..batch {
        for h in 0..num_heads {
            let offset = b * batch_size + h * head_size;

            // For each query position
            for i in 0..seq_len {
                let q_offset = offset + i * head_dim;

                // Compute attention scores and find max for numerical stability
                let mut scores = vec![0.0f32; seq_len];
                let mut max_score = f32::NEG_INFINITY;

                for j in 0..seq_len {
                    // Apply causal mask
                    if causal && j > i {
                        scores[j] = f32::NEG_INFINITY;
                        continue;
                    }

                    let k_offset = offset + j * head_dim;

                    // Dot product Q[i] . K[j]
                    let mut score = 0.0f32;
                    for d in 0..head_dim {
                        score += q[q_offset + d] * k[k_offset + d];
                    }
                    score *= scale;
                    scores[j] = score;
                    max_score = max_score.max(score);
                }

                // Softmax with numerical stability
                let mut sum_exp = 0.0f32;
                for j in 0..seq_len {
                    if scores[j] > f32::NEG_INFINITY {
                        scores[j] = (scores[j] - max_score).exp();
                        sum_exp += scores[j];
                    } else {
                        scores[j] = 0.0;
                    }
                }

                // Normalize
                if sum_exp > 0.0 {
                    for j in 0..seq_len {
                        scores[j] /= sum_exp;
                    }
                }

                // Weighted sum of values
                let o_offset = offset + i * head_dim;
                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for j in 0..seq_len {
                        let v_offset = offset + j * head_dim;
                        sum += scores[j] * v[v_offset + d];
                    }
                    output[o_offset + d] = sum;
                }
            }
        }
    }
}

// ============================================================================
// NIF Functions
// ============================================================================

/// Check if CUDA flash attention is available
#[rustler::nif]
fn cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        if !CUDA_CHECKED.load(Ordering::SeqCst) {
            let available = unsafe { flash_attention_cuda_available() } != 0;
            if available {
                let init_result = unsafe { flash_attention_cuda_init() };
                CUDA_AVAILABLE.store(init_result == 0, Ordering::SeqCst);
            }
            CUDA_CHECKED.store(true, Ordering::SeqCst);
        }
        CUDA_AVAILABLE.load(Ordering::SeqCst)
    }

    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Get backend info
#[rustler::nif]
fn backend_info() -> String {
    #[cfg(feature = "cuda")]
    {
        if cuda_available() {
            "cuda".to_string()
        } else {
            "cpu (cuda feature enabled but unavailable)".to_string()
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        "cpu".to_string()
    }
}

/// Flash attention forward pass
///
/// ## Arguments
///
/// * `q_data` - Query tensor binary [batch, seq_len, num_heads, head_dim], f32
/// * `k_data` - Key tensor binary [batch, seq_len, num_heads, head_dim], f32
/// * `v_data` - Value tensor binary [batch, seq_len, num_heads, head_dim], f32
/// * `batch` - Batch size
/// * `seq_len` - Sequence length
/// * `num_heads` - Number of attention heads
/// * `head_dim` - Dimension per head
/// * `causal` - Whether to apply causal masking
///
/// ## Returns
///
/// Output tensor binary [batch, seq_len, num_heads, head_dim], f32
#[rustler::nif]
fn forward_f32<'a>(
    env: Env<'a>,
    q_data: Binary<'a>,
    k_data: Binary<'a>,
    v_data: Binary<'a>,
    batch: i32,
    seq_len: i32,
    num_heads: i32,
    head_dim: i32,
    causal: bool,
) -> NifResult<(Atom, Binary<'a>)> {
    let total_size = (batch * seq_len * num_heads * head_dim) as usize;
    let byte_size = total_size * 4; // f32 = 4 bytes

    // Validate input sizes
    if q_data.len() != byte_size || k_data.len() != byte_size || v_data.len() != byte_size {
        return Err(rustler::Error::Term(Box::new(format!(
            "Shape mismatch: expected {} bytes, got q={}, k={}, v={}",
            byte_size, q_data.len(), k_data.len(), v_data.len()
        ))));
    }

    // Reinterpret as f32 slices
    let q: &[f32] = unsafe {
        std::slice::from_raw_parts(q_data.as_ptr() as *const f32, total_size)
    };
    let k: &[f32] = unsafe {
        std::slice::from_raw_parts(k_data.as_ptr() as *const f32, total_size)
    };
    let v: &[f32] = unsafe {
        std::slice::from_raw_parts(v_data.as_ptr() as *const f32, total_size)
    };

    // Allocate output
    let mut output_binary = OwnedBinary::new(byte_size)
        .ok_or_else(|| rustler::Error::Term(Box::new("Failed to allocate output")))?;

    let output: &mut [f32] = unsafe {
        std::slice::from_raw_parts_mut(output_binary.as_mut_ptr() as *mut f32, total_size)
    };

    // Try CUDA first, fall back to CPU
    #[cfg(feature = "cuda")]
    {
        if CUDA_AVAILABLE.load(Ordering::SeqCst) {
            let scale = 1.0 / (head_dim as f32).sqrt();
            let result = unsafe {
                flash_attention_forward_f32(
                    q.as_ptr(),
                    k.as_ptr(),
                    v.as_ptr(),
                    output.as_mut_ptr(),
                    batch,
                    seq_len,
                    num_heads,
                    head_dim,
                    if causal { 1 } else { 0 },
                    scale,
                )
            };

            if result == 0 {
                return Ok((atoms::ok(), output_binary.release(env)));
            }
            // Fall through to CPU on CUDA error
        }
    }

    // CPU fallback
    attention_forward_cpu_f32(
        q, k, v, output,
        batch as usize,
        seq_len as usize,
        num_heads as usize,
        head_dim as usize,
        causal,
    );

    Ok((atoms::ok(), output_binary.release(env)))
}

/// Benchmark: Run forward pass multiple times and return average time in microseconds
#[rustler::nif(schedule = "DirtyCpu")]
fn benchmark_forward(
    batch: i32,
    seq_len: i32,
    num_heads: i32,
    head_dim: i32,
    iterations: i32,
) -> NifResult<(Atom, f64, String)> {
    let total_size = (batch * seq_len * num_heads * head_dim) as usize;

    // Allocate test tensors
    let q: Vec<f32> = (0..total_size).map(|i| (i as f32 / total_size as f32) * 0.1).collect();
    let k: Vec<f32> = (0..total_size).map(|i| ((i + 1000) as f32 / total_size as f32) * 0.1).collect();
    let v: Vec<f32> = (0..total_size).map(|i| ((i + 2000) as f32 / total_size as f32) * 0.1).collect();
    let mut output: Vec<f32> = vec![0.0; total_size];

    let backend;

    // Warmup
    #[cfg(feature = "cuda")]
    {
        if CUDA_AVAILABLE.load(Ordering::SeqCst) {
            backend = "cuda";
            let scale = 1.0 / (head_dim as f32).sqrt();
            unsafe {
                flash_attention_forward_f32(
                    q.as_ptr(), k.as_ptr(), v.as_ptr(), output.as_mut_ptr(),
                    batch, seq_len, num_heads, head_dim, 1, scale,
                );
            }
        } else {
            backend = "cpu";
            attention_forward_cpu_f32(
                &q, &k, &v, &mut output,
                batch as usize, seq_len as usize, num_heads as usize, head_dim as usize,
                true,
            );
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        backend = "cpu";
        attention_forward_cpu_f32(
            &q, &k, &v, &mut output,
            batch as usize, seq_len as usize, num_heads as usize, head_dim as usize,
            true,
        );
    }

    // Timed iterations
    let start = std::time::Instant::now();

    for _ in 0..iterations {
        #[cfg(feature = "cuda")]
        {
            if CUDA_AVAILABLE.load(Ordering::SeqCst) {
                let scale = 1.0 / (head_dim as f32).sqrt();
                unsafe {
                    flash_attention_forward_f32(
                        q.as_ptr(), k.as_ptr(), v.as_ptr(), output.as_mut_ptr(),
                        batch, seq_len, num_heads, head_dim, 1, scale,
                    );
                }
            } else {
                attention_forward_cpu_f32(
                    &q, &k, &v, &mut output,
                    batch as usize, seq_len as usize, num_heads as usize, head_dim as usize,
                    true,
                );
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            attention_forward_cpu_f32(
                &q, &k, &v, &mut output,
                batch as usize, seq_len as usize, num_heads as usize, head_dim as usize,
                true,
            );
        }
    }

    let elapsed = start.elapsed();
    let avg_us = elapsed.as_micros() as f64 / iterations as f64;

    Ok((atoms::ok(), avg_us, backend.to_string()))
}

// Register NIFs
rustler::init!("Elixir.ExPhil.Native.FlashAttention");
