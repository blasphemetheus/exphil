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
    use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DevicePtrMut, LaunchAsync, LaunchConfig};
    use once_cell::sync::OnceCell;
    use std::sync::Arc;

    /// Cached CUDA device and compiled module (initialized once, reused across calls)
    static CUDA_CONTEXT: OnceCell<Arc<CudaDevice>> = OnceCell::new();

    fn get_cuda_device() -> Result<Arc<CudaDevice>, KernelError> {
        let dev = CUDA_CONTEXT.get_or_try_init(|| {
            // CudaDevice::new already returns Arc<CudaDevice>
            let dev = CudaDevice::new(0)?;

            // Compile and load PTX once (includes forward, forward_with_states, and backward kernels)
            let ptx = cudarc::nvrtc::compile_ptx(SELECTIVE_SCAN_KERNEL)?;
            dev.load_ptx(ptx, "selective_scan", &[
                "selective_scan_kernel",
                "selective_scan_forward_with_states_kernel",
                "selective_scan_backward_kernel",
            ])?;

            Ok::<_, KernelError>(dev)
        })?;
        Ok(Arc::clone(dev))
    }

    /// CUDA kernel source code (compiled at runtime via NVRTC)
    /// Includes both forward and backward kernels for training support.
    const SELECTIVE_SCAN_KERNEL: &str = r#"

// =============================================================================
// Forward kernel (standard - no hidden state saving)
// =============================================================================
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

// =============================================================================
// Forward kernel with hidden state saving (for backward pass)
// =============================================================================
extern "C" __global__ void selective_scan_forward_with_states_kernel(
    const float* __restrict__ x,      // [batch, seq_len, hidden]
    const float* __restrict__ dt,     // [batch, seq_len, hidden]
    const float* __restrict__ A,      // [hidden, state]
    const float* __restrict__ B,      // [batch, seq_len, state]
    const float* __restrict__ C,      // [batch, seq_len, state]
    float* __restrict__ out,          // [batch, seq_len, hidden]
    float* __restrict__ h_all,        // [batch, seq_len, hidden, state] - saved states
    int batch,
    int seq_len,
    int hidden,
    int state,
    float dt_min,
    float dt_max
) {
    int b = blockIdx.x;
    int h = threadIdx.x + blockIdx.y * blockDim.x;

    if (b >= batch || h >= hidden) return;

    // Initialize hidden state
    float h_state[32];
    for (int s = 0; s < state && s < 32; s++) {
        h_state[s] = 0.0f;
    }

    // Load A diagonal
    float A_diag[32];
    for (int s = 0; s < state && s < 32; s++) {
        A_diag[s] = A[h * state + s];
    }

    // Scan through sequence
    for (int t = 0; t < seq_len; t++) {
        int x_idx = b * seq_len * hidden + t * hidden + h;
        float x_t = x[x_idx];
        float dt_t = fminf(fmaxf(dt[x_idx], dt_min), dt_max);

        int bc_idx = b * seq_len * state + t * state;

        float y_t = 0.0f;

        for (int s = 0; s < state && s < 32; s++) {
            float A_bar = expf(dt_t * A_diag[s]);
            float B_bar = dt_t * B[bc_idx + s];
            float C_s = C[bc_idx + s];

            h_state[s] = A_bar * h_state[s] + B_bar * x_t;
            y_t += C_s * h_state[s];

            // Save hidden state for backward pass
            // h_all: [batch, seq_len, hidden, state]
            int h_idx = b * seq_len * hidden * state + t * hidden * state + h * state + s;
            h_all[h_idx] = h_state[s];
        }

        out[x_idx] = y_t;
    }
}

// =============================================================================
// Backward kernel - computes gradients for training
// =============================================================================
extern "C" __global__ void selective_scan_backward_kernel(
    const float* __restrict__ dy,     // [batch, seq_len, hidden] - gradient from output
    const float* __restrict__ x,      // [batch, seq_len, hidden] - saved input
    const float* __restrict__ h_all,  // [batch, seq_len, hidden, state] - saved hidden states
    const float* __restrict__ dt,     // [batch, seq_len, hidden] - saved dt
    const float* __restrict__ A,      // [hidden, state]
    const float* __restrict__ B,      // [batch, seq_len, state]
    const float* __restrict__ C,      // [batch, seq_len, state]
    float* __restrict__ dx,           // [batch, seq_len, hidden] - gradient w.r.t. x
    float* __restrict__ d_dt,         // [batch, seq_len, hidden] - gradient w.r.t. dt
    float* __restrict__ dB,           // [batch, seq_len, state] - gradient w.r.t. B
    float* __restrict__ dC,           // [batch, seq_len, state] - gradient w.r.t. C
    int batch,
    int seq_len,
    int hidden,
    int state,
    float dt_min,
    float dt_max
) {
    int b = blockIdx.x;
    int h = threadIdx.x + blockIdx.y * blockDim.x;

    if (b >= batch || h >= hidden) return;

    // Load A diagonal
    float A_diag[32];
    for (int s = 0; s < state && s < 32; s++) {
        A_diag[s] = A[h * state + s];
    }

    // Gradient of hidden state (accumulated backward through time)
    float dh[32];
    for (int s = 0; s < state && s < 32; s++) {
        dh[s] = 0.0f;
    }

    // Backward scan (reverse time order)
    for (int t = seq_len - 1; t >= 0; t--) {
        int x_idx = b * seq_len * hidden + t * hidden + h;
        int bc_idx = b * seq_len * state + t * state;

        float x_t = x[x_idx];
        float dt_t = fminf(fmaxf(dt[x_idx], dt_min), dt_max);
        float dy_t = dy[x_idx];

        // Get previous hidden state (or zero for t=0)
        float h_prev[32];
        for (int s = 0; s < state && s < 32; s++) {
            if (t > 0) {
                int h_prev_idx = b * seq_len * hidden * state + (t-1) * hidden * state + h * state + s;
                h_prev[s] = h_all[h_prev_idx];
            } else {
                h_prev[s] = 0.0f;
            }
        }

        // Get current hidden state
        float h_curr[32];
        for (int s = 0; s < state && s < 32; s++) {
            int h_idx = b * seq_len * hidden * state + t * hidden * state + h * state + s;
            h_curr[s] = h_all[h_idx];
        }

        // Gradient from output: y = sum(C * h)
        // dC[s] += dy * h[s]
        // dh[s] += dy * C[s]
        for (int s = 0; s < state && s < 32; s++) {
            float C_s = C[bc_idx + s];
            dh[s] += dy_t * C_s;

            // Atomic add to dC (multiple hidden dims contribute)
            atomicAdd(&dC[bc_idx + s], dy_t * h_curr[s]);
        }

        // Gradient through recurrence: h[t] = A_bar * h[t-1] + B_bar * x
        // where A_bar = exp(dt * A), B_bar = dt * B
        float dx_t = 0.0f;
        float d_dt_t = 0.0f;

        for (int s = 0; s < state && s < 32; s++) {
            float A_bar = expf(dt_t * A_diag[s]);
            float B_s = B[bc_idx + s];
            float B_bar = dt_t * B_s;

            // dx: dh * B_bar
            dx_t += dh[s] * B_bar;

            // dB: dh * dt * x (atomic add, multiple hidden dims)
            atomicAdd(&dB[bc_idx + s], dh[s] * dt_t * x_t);

            // d_dt: dh * (A_bar * A * h_prev + B * x)
            d_dt_t += dh[s] * (A_bar * A_diag[s] * h_prev[s] + B_s * x_t);

            // Propagate gradient to previous timestep
            // dh[t-1] += dh[t] * A_bar
            dh[s] = dh[s] * A_bar;
        }

        dx[x_idx] = dx_t;
        d_dt[x_idx] = d_dt_t;
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
        // Get cached CUDA device (compiles kernel on first call only)
        let dev = get_cuda_device()?;

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

    /// Forward pass that also saves hidden states (needed for backward pass)
    /// Uses a two-step launch to work around cudarc's tuple size limit
    pub fn selective_scan_forward_with_states(
        x: &[f32],
        dt: &[f32],
        a: &[f32],
        b: &[f32],
        c: &[f32],
        batch: usize,
        seq_len: usize,
        hidden: usize,
        state: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), KernelError> {
        let dev = get_cuda_device()?;

        // Copy inputs to GPU
        let x_dev = dev.htod_sync_copy(x)?;
        let dt_dev = dev.htod_sync_copy(dt)?;
        let a_dev = dev.htod_sync_copy(a)?;
        let b_dev = dev.htod_sync_copy(b)?;
        let c_dev = dev.htod_sync_copy(c)?;

        // Allocate outputs on GPU
        let out_size = batch * seq_len * hidden;
        let h_all_size = batch * seq_len * hidden * state;
        let mut out_dev: CudaSlice<f32> = dev.alloc_zeros(out_size)?;
        let mut h_all_dev: CudaSlice<f32> = dev.alloc_zeros(h_all_size)?;

        let func = dev.get_func("selective_scan", "selective_scan_forward_with_states_kernel")
            .ok_or_else(|| KernelError::Launch("Failed to get forward_with_states kernel".into()))?;

        let threads_per_block = hidden.min(256);
        let blocks_y = (hidden + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (batch as u32, blocks_y as u32, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Pack dt_min and dt_max into a single f32 pair to reduce parameter count
        // Kernel expects: x, dt, A, B, C, out, h_all, batch, seq_len, hidden, state, dt_min, dt_max
        // Split into two 6-arg groups for launch compatibility

        // Use raw pointer launch for large parameter counts
        let dt_min: f32 = 0.001;
        let dt_max: f32 = 0.1;
        let batch_i32 = batch as i32;
        let seq_len_i32 = seq_len as i32;
        let hidden_i32 = hidden as i32;
        let state_i32 = state as i32;

        // Build parameter array using device pointers
        // cudarc wants pointers to the device pointers (double indirection)
        let x_ptr = *x_dev.device_ptr();
        let dt_ptr = *dt_dev.device_ptr();
        let a_ptr = *a_dev.device_ptr();
        let b_ptr = *b_dev.device_ptr();
        let c_ptr = *c_dev.device_ptr();
        let out_ptr = *out_dev.device_ptr_mut();
        let h_all_ptr = *h_all_dev.device_ptr_mut();

        let mut params: Vec<*mut std::ffi::c_void> = vec![
            &x_ptr as *const _ as *mut _,
            &dt_ptr as *const _ as *mut _,
            &a_ptr as *const _ as *mut _,
            &b_ptr as *const _ as *mut _,
            &c_ptr as *const _ as *mut _,
            &out_ptr as *const _ as *mut _,
            &h_all_ptr as *const _ as *mut _,
            &batch_i32 as *const _ as *mut _,
            &seq_len_i32 as *const _ as *mut _,
            &hidden_i32 as *const _ as *mut _,
            &state_i32 as *const _ as *mut _,
            &dt_min as *const _ as *mut _,
            &dt_max as *const _ as *mut _,
        ];

        unsafe {
            func.launch(cfg, &mut params)?;
        }

        let out = dev.dtoh_sync_copy(&out_dev)?;
        let h_all = dev.dtoh_sync_copy(&h_all_dev)?;

        Ok((out, h_all))
    }

    /// Backward pass - computes gradients given dy and saved forward state
    pub fn selective_scan_backward(
        dy: &[f32],
        x: &[f32],
        h_all: &[f32],
        dt: &[f32],
        a: &[f32],
        b: &[f32],
        c: &[f32],
        batch: usize,
        seq_len: usize,
        hidden: usize,
        state: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>), KernelError> {
        let dev = get_cuda_device()?;

        // Copy inputs to GPU
        let dy_dev = dev.htod_sync_copy(dy)?;
        let x_dev = dev.htod_sync_copy(x)?;
        let h_all_dev = dev.htod_sync_copy(h_all)?;
        let dt_dev = dev.htod_sync_copy(dt)?;
        let a_dev = dev.htod_sync_copy(a)?;
        let b_dev = dev.htod_sync_copy(b)?;
        let c_dev = dev.htod_sync_copy(c)?;

        // Allocate gradient outputs
        let dx_size = batch * seq_len * hidden;
        let bc_size = batch * seq_len * state;
        let mut dx_dev: CudaSlice<f32> = dev.alloc_zeros(dx_size)?;
        let mut d_dt_dev: CudaSlice<f32> = dev.alloc_zeros(dx_size)?;
        let mut dB_dev: CudaSlice<f32> = dev.alloc_zeros(bc_size)?;
        let mut dC_dev: CudaSlice<f32> = dev.alloc_zeros(bc_size)?;

        let func = dev.get_func("selective_scan", "selective_scan_backward_kernel")
            .ok_or_else(|| KernelError::Launch("Failed to get backward kernel".into()))?;

        let threads_per_block = hidden.min(256);
        let blocks_y = (hidden + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (batch as u32, blocks_y as u32, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let dt_min: f32 = 0.001;
        let dt_max: f32 = 0.1;
        let batch_i32 = batch as i32;
        let seq_len_i32 = seq_len as i32;
        let hidden_i32 = hidden as i32;
        let state_i32 = state as i32;

        // Build parameter array using device pointers
        let dy_ptr = *dy_dev.device_ptr();
        let x_ptr = *x_dev.device_ptr();
        let h_all_ptr = *h_all_dev.device_ptr();
        let dt_ptr = *dt_dev.device_ptr();
        let a_ptr = *a_dev.device_ptr();
        let b_ptr = *b_dev.device_ptr();
        let c_ptr = *c_dev.device_ptr();
        let dx_ptr = *dx_dev.device_ptr_mut();
        let d_dt_ptr = *d_dt_dev.device_ptr_mut();
        let dB_ptr = *dB_dev.device_ptr_mut();
        let dC_ptr = *dC_dev.device_ptr_mut();

        let mut params: Vec<*mut std::ffi::c_void> = vec![
            &dy_ptr as *const _ as *mut _,
            &x_ptr as *const _ as *mut _,
            &h_all_ptr as *const _ as *mut _,
            &dt_ptr as *const _ as *mut _,
            &a_ptr as *const _ as *mut _,
            &b_ptr as *const _ as *mut _,
            &c_ptr as *const _ as *mut _,
            &dx_ptr as *const _ as *mut _,
            &d_dt_ptr as *const _ as *mut _,
            &dB_ptr as *const _ as *mut _,
            &dC_ptr as *const _ as *mut _,
            &batch_i32 as *const _ as *mut _,
            &seq_len_i32 as *const _ as *mut _,
            &hidden_i32 as *const _ as *mut _,
            &state_i32 as *const _ as *mut _,
            &dt_min as *const _ as *mut _,
            &dt_max as *const _ as *mut _,
        ];

        unsafe {
            func.launch(cfg, &mut params)?;
        }

        let dx = dev.dtoh_sync_copy(&dx_dev)?;
        let d_dt = dev.dtoh_sync_copy(&d_dt_dev)?;
        let dB = dev.dtoh_sync_copy(&dB_dev)?;
        let dC = dev.dtoh_sync_copy(&dC_dev)?;

        Ok((dx, d_dt, dB, dC))
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

/// CPU implementation of forward pass with hidden state saving
pub fn selective_scan_forward_with_states_cpu(
    x: &[f32],
    dt: &[f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    batch: usize,
    seq_len: usize,
    hidden: usize,
    state: usize,
) -> (Vec<f32>, Vec<f32>) {
    let dt_min: f32 = 0.001;
    let dt_max: f32 = 0.1;

    let mut out = vec![0.0f32; batch * seq_len * hidden];
    let mut h_all = vec![0.0f32; batch * seq_len * hidden * state];

    for batch_idx in 0..batch {
        for h in 0..hidden {
            let mut h_state = vec![0.0f32; state];
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

                    // Save hidden state: h_all[batch, seq_len, hidden, state]
                    let h_idx = batch_idx * seq_len * hidden * state
                        + t * hidden * state
                        + h * state
                        + s;
                    h_all[h_idx] = h_state[s];
                }

                out[x_idx] = y_t;
            }
        }
    }

    (out, h_all)
}

/// CPU implementation of backward pass
pub fn selective_scan_backward_cpu(
    dy: &[f32],
    x: &[f32],
    h_all: &[f32],
    dt: &[f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    batch: usize,
    seq_len: usize,
    hidden: usize,
    state: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let dt_min: f32 = 0.001;
    let dt_max: f32 = 0.1;

    let mut dx = vec![0.0f32; batch * seq_len * hidden];
    let mut d_dt = vec![0.0f32; batch * seq_len * hidden];
    let mut d_b = vec![0.0f32; batch * seq_len * state];
    let mut d_c = vec![0.0f32; batch * seq_len * state];

    for batch_idx in 0..batch {
        for h in 0..hidden {
            let a_diag: Vec<f32> = (0..state).map(|s| a[h * state + s]).collect();

            // Gradient of hidden state (accumulated backward through time)
            let mut dh = vec![0.0f32; state];

            // Backward scan (reverse time order)
            for t in (0..seq_len).rev() {
                let x_idx = batch_idx * seq_len * hidden + t * hidden + h;
                let bc_idx = batch_idx * seq_len * state + t * state;

                let x_t = x[x_idx];
                let dt_t = dt[x_idx].clamp(dt_min, dt_max);
                let dy_t = dy[x_idx];

                // Get previous hidden state (or zero for t=0)
                let h_prev: Vec<f32> = (0..state)
                    .map(|s| {
                        if t > 0 {
                            let h_prev_idx = batch_idx * seq_len * hidden * state
                                + (t - 1) * hidden * state
                                + h * state
                                + s;
                            h_all[h_prev_idx]
                        } else {
                            0.0
                        }
                    })
                    .collect();

                // Get current hidden state
                let h_curr: Vec<f32> = (0..state)
                    .map(|s| {
                        let h_idx = batch_idx * seq_len * hidden * state
                            + t * hidden * state
                            + h * state
                            + s;
                        h_all[h_idx]
                    })
                    .collect();

                // Gradient from output: y = sum(C * h)
                for s in 0..state {
                    let c_s = c[bc_idx + s];
                    dh[s] += dy_t * c_s;
                    d_c[bc_idx + s] += dy_t * h_curr[s];
                }

                // Gradient through recurrence
                let mut dx_t = 0.0f32;
                let mut d_dt_t = 0.0f32;

                for s in 0..state {
                    let a_bar = (dt_t * a_diag[s]).exp();
                    let b_s = b[bc_idx + s];
                    let b_bar = dt_t * b_s;

                    // dx: dh * B_bar
                    dx_t += dh[s] * b_bar;

                    // dB: dh * dt * x
                    d_b[bc_idx + s] += dh[s] * dt_t * x_t;

                    // d_dt: dh * (A_bar * A * h_prev + B * x)
                    d_dt_t += dh[s] * (a_bar * a_diag[s] * h_prev[s] + b_s * x_t);

                    // Propagate gradient to previous timestep
                    dh[s] = dh[s] * a_bar;
                }

                dx[x_idx] = dx_t;
                d_dt[x_idx] = d_dt_t;
            }
        }
    }

    (dx, d_dt, d_b, d_c)
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

/// Forward pass that saves hidden states (for training)
pub fn selective_scan_forward_with_states(
    x: &[f32],
    dt: &[f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    batch: usize,
    seq_len: usize,
    hidden: usize,
    state: usize,
) -> Result<(Vec<f32>, Vec<f32>), KernelError> {
    #[cfg(feature = "cuda")]
    {
        if cuda_impl::is_cuda_available() {
            cuda_impl::selective_scan_forward_with_states(x, dt, a, b, c, batch, seq_len, hidden, state)
        } else {
            Ok(selective_scan_forward_with_states_cpu(x, dt, a, b, c, batch, seq_len, hidden, state))
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        Ok(selective_scan_forward_with_states_cpu(x, dt, a, b, c, batch, seq_len, hidden, state))
    }
}

/// Backward pass (computes gradients)
pub fn selective_scan_backward(
    dy: &[f32],
    x: &[f32],
    h_all: &[f32],
    dt: &[f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    batch: usize,
    seq_len: usize,
    hidden: usize,
    state: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>), KernelError> {
    #[cfg(feature = "cuda")]
    {
        if cuda_impl::is_cuda_available() {
            cuda_impl::selective_scan_backward(dy, x, h_all, dt, a, b, c, batch, seq_len, hidden, state)
        } else {
            Ok(selective_scan_backward_cpu(dy, x, h_all, dt, a, b, c, batch, seq_len, hidden, state))
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        Ok(selective_scan_backward_cpu(dy, x, h_all, dt, a, b, c, batch, seq_len, hidden, state))
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
