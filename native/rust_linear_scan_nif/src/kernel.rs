//! CUDA kernel for linear scan: h[t] = a[t] * h[t-1] + b[t]
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

    #[cfg(feature = "cuda")]
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
    use once_cell::sync::OnceCell;
    use std::sync::Arc;

    /// Cached CUDA device and compiled module
    static CUDA_CONTEXT: OnceCell<Arc<CudaDevice>> = OnceCell::new();

    fn get_cuda_device() -> Result<Arc<CudaDevice>, KernelError> {
        let dev = CUDA_CONTEXT.get_or_try_init(|| {
            let dev = CudaDevice::new(0)?;

            let ptx = cudarc::nvrtc::compile_ptx(LINEAR_SCAN_KERNEL)?;
            dev.load_ptx(ptx, "linear_scan", &[
                "fused_linear_scan_kernel",
            ])?;

            Ok::<_, KernelError>(dev)
        })?;
        Ok(Arc::clone(dev))
    }

    /// CUDA kernel: one thread per (batch, hidden), sequential loop over timesteps
    const LINEAR_SCAN_KERNEL: &str = r#"
extern "C" __global__ void fused_linear_scan_kernel(
    const float* __restrict__ a_vals,   // [batch, seq_len, hidden]
    const float* __restrict__ b_vals,   // [batch, seq_len, hidden]
    const float* __restrict__ h0,       // [batch, hidden]
    float* __restrict__ output,         // [batch, seq_len, hidden]
    int batch,
    int seq_len,
    int hidden
) {
    // Each thread handles one (batch, hidden) pair
    int b = blockIdx.x;
    int h = threadIdx.x + blockIdx.y * blockDim.x;

    if (b >= batch || h >= hidden) return;

    // Load initial state
    float h_state = h0[b * hidden + h];

    // Sequential scan over timesteps
    for (int t = 0; t < seq_len; t++) {
        int idx = b * seq_len * hidden + t * hidden + h;
        float a_t = a_vals[idx];
        float b_t = b_vals[idx];

        // Linear recurrence: h = a * h + b
        h_state = a_t * h_state + b_t;

        // Store output
        output[idx] = h_state;
    }
}
"#;

    pub fn is_cuda_available() -> bool {
        CudaDevice::new(0).is_ok()
    }

    pub fn linear_scan_cuda(
        a: &[f32],
        b: &[f32],
        h0: &[f32],
        batch: usize,
        seq_len: usize,
        hidden: usize,
    ) -> Result<Vec<f32>, KernelError> {
        let dev = get_cuda_device()?;

        // Copy input tensors to GPU
        let a_dev = dev.htod_sync_copy(a)?;
        let b_dev = dev.htod_sync_copy(b)?;
        let h0_dev = dev.htod_sync_copy(h0)?;

        // Allocate output buffer on GPU
        let out_size = batch * seq_len * hidden;
        let mut out_dev: CudaSlice<f32> = dev.alloc_zeros(out_size)?;

        let func = dev.get_func("linear_scan", "fused_linear_scan_kernel")
            .ok_or_else(|| KernelError::Launch("Failed to get kernel function".into()))?;

        // Grid: (batch, ceil(hidden/threads_per_block), 1)
        // Block: (threads_per_block, 1, 1)
        let threads_per_block = hidden.min(256);
        let blocks_y = (hidden + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (batch as u32, blocks_y as u32, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(
                cfg,
                (
                    &a_dev,
                    &b_dev,
                    &h0_dev,
                    &mut out_dev,
                    batch as i32,
                    seq_len as i32,
                    hidden as i32,
                ),
            )?;
        }

        let out = dev.dtoh_sync_copy(&out_dev)?;
        Ok(out)
    }
}

// =============================================================================
// CPU Fallback Implementation
// =============================================================================

pub fn linear_scan_cpu(
    a: &[f32],
    b: &[f32],
    h0: &[f32],
    batch: usize,
    seq_len: usize,
    hidden: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * seq_len * hidden];

    for batch_idx in 0..batch {
        for h in 0..hidden {
            let mut h_state = h0[batch_idx * hidden + h];

            for t in 0..seq_len {
                let idx = batch_idx * seq_len * hidden + t * hidden + h;
                let a_t = a[idx];
                let b_t = b[idx];

                h_state = a_t * h_state + b_t;
                out[idx] = h_state;
            }
        }
    }

    out
}

// =============================================================================
// Public API
// =============================================================================

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

pub fn linear_scan_cuda(
    a: &[f32],
    b: &[f32],
    h0: &[f32],
    batch: usize,
    seq_len: usize,
    hidden: usize,
) -> Result<Vec<f32>, KernelError> {
    #[cfg(feature = "cuda")]
    {
        if cuda_impl::is_cuda_available() {
            cuda_impl::linear_scan_cuda(a, b, h0, batch, seq_len, hidden)
        } else {
            Ok(linear_scan_cpu(a, b, h0, batch, seq_len, hidden))
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        Ok(linear_scan_cpu(a, b, h0, batch, seq_len, hidden))
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

        let a: Vec<f32> = vec![0.9; batch * seq_len * hidden];
        let b: Vec<f32> = (0..batch * seq_len * hidden).map(|i| i as f32 * 0.1).collect();
        let h0: Vec<f32> = vec![0.0; batch * hidden];

        let result = linear_scan_cpu(&a, &b, &h0, batch, seq_len, hidden);

        assert_eq!(result.len(), batch * seq_len * hidden);
        // Values should be non-zero after the scan
        assert!(result.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_cpu_scan_identity() {
        // With a=1 and b=0 after first step, h should stay constant
        let batch = 1;
        let seq_len = 3;
        let hidden = 2;

        let a = vec![1.0f32; batch * seq_len * hidden];
        let b = vec![
            1.0, 2.0,  // t=0: h = 1*0 + b = [1, 2]
            0.0, 0.0,  // t=1: h = 1*[1,2] + 0 = [1, 2]
            0.0, 0.0,  // t=2: h = 1*[1,2] + 0 = [1, 2]
        ];
        let h0 = vec![0.0f32; batch * hidden];

        let result = linear_scan_cpu(&a, &b, &h0, batch, seq_len, hidden);

        // t=0: h = 1*0 + [1,2] = [1,2]
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 2.0).abs() < 1e-6);
        // t=1: h = 1*[1,2] + [0,0] = [1,2]
        assert!((result[2] - 1.0).abs() < 1e-6);
        assert!((result[3] - 2.0).abs() < 1e-6);
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

        let a: Vec<f32> = vec![0.9; batch * seq_len * hidden];
        let b: Vec<f32> = (0..batch * seq_len * hidden).map(|i| i as f32 * 0.1).collect();
        let h0: Vec<f32> = vec![0.0; batch * hidden];

        let cpu_result = linear_scan_cpu(&a, &b, &h0, batch, seq_len, hidden);
        let cuda_result = linear_scan_cuda(&a, &b, &h0, batch, seq_len, hidden)
            .expect("CUDA scan failed");

        assert_eq!(cpu_result.len(), cuda_result.len());
        for (i, (cpu, cuda)) in cpu_result.iter().zip(cuda_result.iter()).enumerate() {
            let diff = (cpu - cuda).abs();
            assert!(diff < 1e-4, "Mismatch at {}: CPU={}, CUDA={}, diff={}", i, cpu, cuda, diff);
        }
    }
}
