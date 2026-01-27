//! CUDA kernel implementation for selective scan
//!
//! Uses cudarc to interface with CUDA and compile/run PTX kernels.
//! Falls back to CPU implementation when CUDA is not available.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum KernelError {
    #[cfg(feature = "cuda")]
    #[error("CUDA error: {0}")]
    Cuda(#[from] cudarc::driver::DriverError),

    #[cfg(feature = "cuda")]
    #[error("NVRTC error: {0}")]
    Nvrtc(#[from] cudarc::nvrtc::CompileError),

    #[error("No CUDA device available")]
    NoDevice,

    #[error("Kernel launch failed: {0}")]
    Launch(String),

    #[error("CUDA not compiled in - rebuild with --features cuda")]
    NoCudaFeature,
}

// =============================================================================
// CUDA Implementation
// =============================================================================

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::*;
    use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
    use std::sync::Arc;

    /// CUDA kernel source code (compiled at runtime via NVRTC)
    const SELECTIVE_SCAN_KERNEL: &str = r#"
extern "C" __global__ void selective_scan_kernel(
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
"#;

    pub fn is_cuda_available() -> bool {
        CudaDevice::new(0).is_ok()
    }

    pub fn get_device_info() -> Result<String, KernelError> {
        let dev = CudaDevice::new(0)?;
        Ok(format!(
            "Device 0: {} (CUDA available)",
            dev.name()?
        ))
    }

    pub fn selective_scan_cuda(
        x: &[f32],
        dt: &[f32],
        a: &[f32],
        b: &[f32],
        c: &[f32],
        batch: usize,
        seq_len: usize,
        hidden: usize,
        state: usize,
    ) -> Result<Vec<f32>, KernelError> {
        // Initialize CUDA device
        let dev = Arc::new(CudaDevice::new(0)?);

        // Compile PTX from CUDA source at runtime
        let ptx = cudarc::nvrtc::compile_ptx(SELECTIVE_SCAN_KERNEL)?;
        dev.load_ptx(ptx, "selective_scan", &["selective_scan_kernel"])?;

        // Copy input tensors to GPU
        let x_dev = dev.htod_sync_copy(x)?;
        let dt_dev = dev.htod_sync_copy(dt)?;
        let a_dev = dev.htod_sync_copy(a)?;
        let b_dev = dev.htod_sync_copy(b)?;
        let c_dev = dev.htod_sync_copy(c)?;

        // Allocate output buffer on GPU
        let out_size = batch * seq_len * hidden;
        let mut out_dev: CudaSlice<f32> = dev.alloc_zeros(out_size)?;

        // Get compiled kernel function
        let func = dev.get_func("selective_scan", "selective_scan_kernel")
            .ok_or_else(|| KernelError::Launch("Failed to get kernel function".into()))?;

        // Configure kernel launch
        // Grid: (batch, ceil(hidden/threads_per_block), 1)
        // Block: (threads_per_block, 1, 1)
        let threads_per_block = hidden.min(256);
        let blocks_y = (hidden + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (batch as u32, blocks_y as u32, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let dt_min: f32 = 0.001;
        let dt_max: f32 = 0.1;

        // Launch kernel
        unsafe {
            func.launch(
                cfg,
                (
                    &x_dev,
                    &dt_dev,
                    &a_dev,
                    &b_dev,
                    &c_dev,
                    &mut out_dev,
                    batch as i32,
                    seq_len as i32,
                    hidden as i32,
                    state as i32,
                    dt_min,
                    dt_max,
                ),
            )?;
        }

        // Copy result back to CPU
        let out = dev.dtoh_sync_copy(&out_dev)?;

        Ok(out)
    }
}

// =============================================================================
// CPU Fallback Implementation
// =============================================================================

/// CPU implementation of selective scan (for testing/fallback)
pub fn selective_scan_cpu(
    x: &[f32],
    dt: &[f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    batch: usize,
    seq_len: usize,
    hidden: usize,
    state: usize,
) -> Vec<f32> {
    let dt_min: f32 = 0.001;
    let dt_max: f32 = 0.1;

    let mut out = vec![0.0f32; batch * seq_len * hidden];

    for batch_idx in 0..batch {
        for h in 0..hidden {
            // Hidden state for this (batch, hidden) pair
            let mut h_state = vec![0.0f32; state];

            // Load A diagonal
            let a_diag: Vec<f32> = (0..state).map(|s| a[h * state + s]).collect();

            for t in 0..seq_len {
                let x_idx = batch_idx * seq_len * hidden + t * hidden + h;
                let x_t = x[x_idx];
                let dt_t = dt[x_idx].clamp(dt_min, dt_max);

                let bc_idx = batch_idx * seq_len * state + t * state;

                let mut y_t = 0.0f32;

                for s in 0..state {
                    let a_bar = (dt_t * a_diag[s]).exp();
                    let b_bar = dt_t * b[bc_idx + s];
                    let c_s = c[bc_idx + s];

                    h_state[s] = a_bar * h_state[s] + b_bar * x_t;
                    y_t += c_s * h_state[s];
                }

                out[x_idx] = y_t;
            }
        }
    }

    out
}

// =============================================================================
// Public API
// =============================================================================

/// Check if CUDA is available
pub fn is_cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        cuda_impl::is_cuda_available()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Get CUDA device info
pub fn get_device_info() -> Result<String, KernelError> {
    #[cfg(feature = "cuda")]
    {
        cuda_impl::get_device_info()
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(KernelError::NoCudaFeature)
    }
}

/// Perform selective scan (CUDA if available, CPU fallback otherwise)
pub fn selective_scan_cuda(
    x: &[f32],
    dt: &[f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    batch: usize,
    seq_len: usize,
    hidden: usize,
    state: usize,
) -> Result<Vec<f32>, KernelError> {
    #[cfg(feature = "cuda")]
    {
        if cuda_impl::is_cuda_available() {
            cuda_impl::selective_scan_cuda(x, dt, a, b, c, batch, seq_len, hidden, state)
        } else {
            // Fall back to CPU
            Ok(selective_scan_cpu(x, dt, a, b, c, batch, seq_len, hidden, state))
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        // CPU-only build
        Ok(selective_scan_cpu(x, dt, a, b, c, batch, seq_len, hidden, state))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_scan_basic() {
        let batch = 2;
        let seq_len = 4;
        let hidden = 8;
        let state = 2;

        let x: Vec<f32> = (0..batch * seq_len * hidden).map(|i| i as f32 * 0.1).collect();
        let dt: Vec<f32> = vec![0.05; batch * seq_len * hidden];
        let a: Vec<f32> = (0..hidden * state).map(|i| -(1.0 + (i % state) as f32)).collect();
        let b: Vec<f32> = vec![1.0; batch * seq_len * state];
        let c: Vec<f32> = vec![1.0; batch * seq_len * state];

        let result = selective_scan_cpu(&x, &dt, &a, &b, &c, batch, seq_len, hidden, state);

        assert_eq!(result.len(), batch * seq_len * hidden);
        // Values should be non-zero after the scan
        assert!(result.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_cuda_availability() {
        println!("CUDA compiled in: {}", cfg!(feature = "cuda"));
        println!("CUDA available: {}", is_cuda_available());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_scan() {
        if !is_cuda_available() {
            println!("Skipping: no CUDA device");
            return;
        }

        let batch = 2;
        let seq_len = 4;
        let hidden = 8;
        let state = 2;

        let x: Vec<f32> = (0..batch * seq_len * hidden).map(|i| i as f32 * 0.1).collect();
        let dt: Vec<f32> = vec![0.05; batch * seq_len * hidden];
        let a: Vec<f32> = (0..hidden * state).map(|i| -(1.0 + (i % state) as f32)).collect();
        let b: Vec<f32> = vec![1.0; batch * seq_len * state];
        let c: Vec<f32> = vec![1.0; batch * seq_len * state];

        // Run both CPU and CUDA
        let cpu_result = selective_scan_cpu(&x, &dt, &a, &b, &c, batch, seq_len, hidden, state);
        let cuda_result = selective_scan_cuda(&x, &dt, &a, &b, &c, batch, seq_len, hidden, state)
            .expect("CUDA scan failed");

        // Verify they match (within floating point tolerance)
        assert_eq!(cpu_result.len(), cuda_result.len());
        for (i, (cpu, cuda)) in cpu_result.iter().zip(cuda_result.iter()).enumerate() {
            let diff = (cpu - cuda).abs();
            assert!(diff < 1e-4, "Mismatch at {}: CPU={}, CUDA={}, diff={}", i, cpu, cuda, diff);
        }

        println!("CPU and CUDA results match!");
    }
}
