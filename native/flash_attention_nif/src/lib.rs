//! FlashAttention-2 NIF with Forward and Backward Pass Support
//!
//! Provides GPU-accelerated flash attention for both inference and training.
//!
//! ## Features
//!
//! - `cuda`: Enable CUDA kernel support (requires CUDA toolkit + Ampere+ GPU)
//!
//! ## API
//!
//! - `forward_f32`: Forward pass (inference only, fastest)
//! - `forward_with_logsumexp`: Forward pass that saves logsumexp for backward
//! - `backward_f32`: Backward pass computing dQ, dK, dV gradients
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
///
/// Also saves logsumexp for backward pass when logsumexp output is provided.
fn attention_forward_with_logsumexp_cpu_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    logsumexp: &mut [f32],  // [batch, num_heads, seq_len]
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
            let lse_offset = b * num_heads * seq_len + h * seq_len;

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

                // Softmax with numerical stability, saving logsumexp
                let mut sum_exp = 0.0f32;
                for j in 0..seq_len {
                    if scores[j] > f32::NEG_INFINITY {
                        sum_exp += (scores[j] - max_score).exp();
                    }
                }

                // Save logsumexp: log(sum(exp(scores - max))) + max = log(sum(exp(scores)))
                let lse = max_score + sum_exp.ln();
                logsumexp[lse_offset + i] = lse;

                // Normalize softmax
                for j in 0..seq_len {
                    if scores[j] > f32::NEG_INFINITY {
                        scores[j] = (scores[j] - lse).exp();
                    } else {
                        scores[j] = 0.0;
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

/// Backward pass for attention on CPU
///
/// Computes gradients dQ, dK, dV given:
/// - d_out: gradient from downstream
/// - q, k, v: original inputs
/// - output: forward output
/// - logsumexp: saved from forward pass
///
/// Algorithm (FlashAttention-2 paper, Algorithm 4):
/// 1. Recompute attention scores S = QK^T / sqrt(d)
/// 2. Recompute attention weights P = softmax(S) using saved logsumexp
/// 3. Compute dV = P^T @ dO
/// 4. Compute dP = dO @ V^T
/// 5. Compute Di = rowsum(dO ⊙ O)
/// 6. Compute dS = P ⊙ (dP - Di)
/// 7. Compute dQ = dS @ K / sqrt(d)
/// 8. Compute dK = dS^T @ Q / sqrt(d)
fn attention_backward_cpu_f32(
    d_out: &[f32],     // [batch, seq_len, num_heads, head_dim]
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &[f32],
    logsumexp: &[f32], // [batch, num_heads, seq_len]
    dq: &mut [f32],
    dk: &mut [f32],
    dv: &mut [f32],
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
            let lse_offset = b * num_heads * seq_len + h * seq_len;

            // For each query position i
            for i in 0..seq_len {
                let q_offset = offset + i * head_dim;
                let o_offset = offset + i * head_dim;
                let do_offset = offset + i * head_dim;
                let lse_i = logsumexp[lse_offset + i];

                // Compute Di = sum(dO[i] * O[i]) for softmax gradient
                let mut di = 0.0f32;
                for d in 0..head_dim {
                    di += d_out[do_offset + d] * output[o_offset + d];
                }

                // For each key/value position j
                for j in 0..seq_len {
                    // Apply causal mask
                    if causal && j > i {
                        continue;
                    }

                    let k_offset = offset + j * head_dim;
                    let v_offset = offset + j * head_dim;

                    // Recompute attention score S[i,j] = Q[i] . K[j] / sqrt(d)
                    let mut score = 0.0f32;
                    for d in 0..head_dim {
                        score += q[q_offset + d] * k[k_offset + d];
                    }
                    score *= scale;

                    // Recompute attention weight P[i,j] = exp(S[i,j] - lse[i])
                    let p_ij = (score - lse_i).exp();

                    // dV[j] += P[i,j] * dO[i]
                    for d in 0..head_dim {
                        dv[v_offset + d] += p_ij * d_out[do_offset + d];
                    }

                    // dP[i,j] = dO[i] . V[j]
                    let mut dp_ij = 0.0f32;
                    for d in 0..head_dim {
                        dp_ij += d_out[do_offset + d] * v[v_offset + d];
                    }

                    // dS[i,j] = P[i,j] * (dP[i,j] - Di)
                    // This is the gradient through softmax
                    let ds_ij = p_ij * (dp_ij - di);

                    // dQ[i] += dS[i,j] * K[j] / sqrt(d)
                    for d in 0..head_dim {
                        dq[q_offset + d] += ds_ij * k[k_offset + d] * scale;
                    }

                    // dK[j] += dS[i,j] * Q[i] / sqrt(d)
                    for d in 0..head_dim {
                        dk[k_offset + d] += ds_ij * q[q_offset + d] * scale;
                    }
                }
            }
        }
    }
}

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

/// Flash attention forward pass with logsumexp saving (for backward pass)
///
/// Returns (output, logsumexp) where logsumexp is [batch, num_heads, seq_len]
#[rustler::nif]
fn forward_with_logsumexp<'a>(
    env: Env<'a>,
    q_data: Binary<'a>,
    k_data: Binary<'a>,
    v_data: Binary<'a>,
    batch: i32,
    seq_len: i32,
    num_heads: i32,
    head_dim: i32,
    causal: bool,
) -> NifResult<(Atom, Binary<'a>, Binary<'a>)> {
    let total_size = (batch * seq_len * num_heads * head_dim) as usize;
    let byte_size = total_size * 4; // f32 = 4 bytes
    let lse_size = (batch * num_heads * seq_len) as usize;
    let lse_byte_size = lse_size * 4;

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

    // Allocate outputs
    let mut output_binary = OwnedBinary::new(byte_size)
        .ok_or_else(|| rustler::Error::Term(Box::new("Failed to allocate output")))?;
    let mut lse_binary = OwnedBinary::new(lse_byte_size)
        .ok_or_else(|| rustler::Error::Term(Box::new("Failed to allocate logsumexp")))?;

    let output: &mut [f32] = unsafe {
        std::slice::from_raw_parts_mut(output_binary.as_mut_ptr() as *mut f32, total_size)
    };
    let logsumexp: &mut [f32] = unsafe {
        std::slice::from_raw_parts_mut(lse_binary.as_mut_ptr() as *mut f32, lse_size)
    };

    // CPU implementation (TODO: add CUDA path)
    attention_forward_with_logsumexp_cpu_f32(
        q, k, v, output, logsumexp,
        batch as usize,
        seq_len as usize,
        num_heads as usize,
        head_dim as usize,
        causal,
    );

    Ok((atoms::ok(), output_binary.release(env), lse_binary.release(env)))
}

/// Flash attention backward pass
///
/// Computes gradients dQ, dK, dV given:
/// - d_out: gradient from downstream [batch, seq_len, num_heads, head_dim]
/// - q, k, v: original inputs
/// - output: forward output
/// - logsumexp: saved from forward_with_logsumexp [batch, num_heads, seq_len]
///
/// Returns (dQ, dK, dV)
#[rustler::nif]
fn backward_f32<'a>(
    env: Env<'a>,
    d_out_data: Binary<'a>,
    q_data: Binary<'a>,
    k_data: Binary<'a>,
    v_data: Binary<'a>,
    output_data: Binary<'a>,
    logsumexp_data: Binary<'a>,
    batch: i32,
    seq_len: i32,
    num_heads: i32,
    head_dim: i32,
    causal: bool,
) -> NifResult<(Atom, Binary<'a>, Binary<'a>, Binary<'a>)> {
    let total_size = (batch * seq_len * num_heads * head_dim) as usize;
    let byte_size = total_size * 4;
    let lse_size = (batch * num_heads * seq_len) as usize;
    let lse_byte_size = lse_size * 4;

    // Validate input sizes
    if d_out_data.len() != byte_size {
        return Err(rustler::Error::Term(Box::new(format!(
            "d_out size mismatch: expected {}, got {}", byte_size, d_out_data.len()
        ))));
    }
    if q_data.len() != byte_size || k_data.len() != byte_size || v_data.len() != byte_size {
        return Err(rustler::Error::Term(Box::new(format!(
            "QKV size mismatch: expected {}", byte_size
        ))));
    }
    if output_data.len() != byte_size {
        return Err(rustler::Error::Term(Box::new(format!(
            "output size mismatch: expected {}, got {}", byte_size, output_data.len()
        ))));
    }
    if logsumexp_data.len() != lse_byte_size {
        return Err(rustler::Error::Term(Box::new(format!(
            "logsumexp size mismatch: expected {}, got {}", lse_byte_size, logsumexp_data.len()
        ))));
    }

    // Reinterpret as f32 slices
    let d_out: &[f32] = unsafe {
        std::slice::from_raw_parts(d_out_data.as_ptr() as *const f32, total_size)
    };
    let q: &[f32] = unsafe {
        std::slice::from_raw_parts(q_data.as_ptr() as *const f32, total_size)
    };
    let k: &[f32] = unsafe {
        std::slice::from_raw_parts(k_data.as_ptr() as *const f32, total_size)
    };
    let v: &[f32] = unsafe {
        std::slice::from_raw_parts(v_data.as_ptr() as *const f32, total_size)
    };
    let output: &[f32] = unsafe {
        std::slice::from_raw_parts(output_data.as_ptr() as *const f32, total_size)
    };
    let logsumexp: &[f32] = unsafe {
        std::slice::from_raw_parts(logsumexp_data.as_ptr() as *const f32, lse_size)
    };

    // Allocate gradient outputs (initialized to zero)
    let mut dq_binary = OwnedBinary::new(byte_size)
        .ok_or_else(|| rustler::Error::Term(Box::new("Failed to allocate dQ")))?;
    let mut dk_binary = OwnedBinary::new(byte_size)
        .ok_or_else(|| rustler::Error::Term(Box::new("Failed to allocate dK")))?;
    let mut dv_binary = OwnedBinary::new(byte_size)
        .ok_or_else(|| rustler::Error::Term(Box::new("Failed to allocate dV")))?;

    // Zero-initialize gradient outputs
    dq_binary.as_mut_slice().fill(0);
    dk_binary.as_mut_slice().fill(0);
    dv_binary.as_mut_slice().fill(0);

    let dq: &mut [f32] = unsafe {
        std::slice::from_raw_parts_mut(dq_binary.as_mut_ptr() as *mut f32, total_size)
    };
    let dk: &mut [f32] = unsafe {
        std::slice::from_raw_parts_mut(dk_binary.as_mut_ptr() as *mut f32, total_size)
    };
    let dv: &mut [f32] = unsafe {
        std::slice::from_raw_parts_mut(dv_binary.as_mut_ptr() as *mut f32, total_size)
    };

    // CPU implementation (TODO: add CUDA path)
    attention_backward_cpu_f32(
        d_out, q, k, v, output, logsumexp,
        dq, dk, dv,
        batch as usize,
        seq_len as usize,
        num_heads as usize,
        head_dim as usize,
        causal,
    );

    Ok((
        atoms::ok(),
        dq_binary.release(env),
        dk_binary.release(env),
        dv_binary.release(env),
    ))
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
rustler::init!(
    "Elixir.ExPhil.Native.FlashAttention",
    [
        cuda_available,
        backend_info,
        forward_f32,
        forward_with_logsumexp,
        backward_f32,
        benchmark_forward
    ]
);
