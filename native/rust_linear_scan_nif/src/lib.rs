//! CUDA-accelerated Linear Scan NIF
//!
//! Implements the linear recurrence h = a*h + b via Rustler NIF with CUDA kernel.
//! This is the simplest SSM scan — used for benchmarking Rust-CUDA integration
//! against CUDA C (XLA), Futhark, Julia, and other backends.
//!
//! # Building
//!
//! ```bash
//! cd native/rust_linear_scan_nif
//! cargo build --release --features cuda
//! ```
//!
//! # Usage from Elixir
//!
//! ```elixir
//! alias ExPhil.Native.RustLinearScan
//! result = RustLinearScan.linear_scan(a, b, h0)
//! ```

use rustler::{Binary, NifResult, OwnedBinary, Error};
use std::io::Write;

mod kernel;

// =============================================================================
// NIF Functions
// =============================================================================

/// Perform linear scan on GPU: h[t] = a[t] * h[t-1] + b[t]
///
/// # Arguments
/// * `a` - Decay coefficients binary [batch * seq_len * hidden], f32
/// * `b` - Additive terms binary [batch * seq_len * hidden], f32
/// * `h0` - Initial hidden state binary [batch * hidden], f32
/// * `shape` - Tuple of (batch, seq_len, hidden)
///
/// # Returns
/// * Output tensor binary [batch * seq_len * hidden], f32
#[rustler::nif(schedule = "DirtyCpu")]
fn linear_scan_nif(
    a: Binary,
    b: Binary,
    h0: Binary,
    shape: (usize, usize, usize),
) -> NifResult<OwnedBinary> {
    let (batch, seq_len, hidden) = shape;

    let a_data = binary_to_f32_vec(&a)?;
    let b_data = binary_to_f32_vec(&b)?;
    let h0_data = binary_to_f32_vec(&h0)?;

    // Validate sizes
    let expected_3d = batch * seq_len * hidden;
    let expected_2d = batch * hidden;

    if a_data.len() != expected_3d {
        return Err(Error::Term(Box::new(format!(
            "a size mismatch: expected {}, got {}", expected_3d, a_data.len()
        ))));
    }
    if b_data.len() != expected_3d {
        return Err(Error::Term(Box::new(format!(
            "b size mismatch: expected {}, got {}", expected_3d, b_data.len()
        ))));
    }
    if h0_data.len() != expected_2d {
        return Err(Error::Term(Box::new(format!(
            "h0 size mismatch: expected {}, got {}", expected_2d, h0_data.len()
        ))));
    }

    let result = kernel::linear_scan_cuda(
        &a_data, &b_data, &h0_data,
        batch, seq_len, hidden
    ).map_err(|e| Error::Term(Box::new(e.to_string())))?;

    // Convert result to binary
    let mut output = OwnedBinary::new(result.len() * 4).unwrap();
    output.as_mut_slice().write_all(bytemuck::cast_slice(&result))
        .map_err(|_| Error::Term(Box::new("Failed to write output")))?;

    Ok(output)
}

/// Check if CUDA is available
#[rustler::nif]
fn cuda_available() -> bool {
    kernel::is_cuda_available()
}

/// Simple ping for testing NIF loading
#[rustler::nif]
fn ping() -> &'static str {
    "pong from rust_linear_scan_nif"
}

// =============================================================================
// Helpers
// =============================================================================

fn binary_to_f32_vec(binary: &Binary) -> NifResult<Vec<f32>> {
    if binary.len() % 4 != 0 {
        return Err(Error::Term(Box::new("Binary length not multiple of 4 (f32)")));
    }
    let slice: &[f32] = bytemuck::cast_slice(binary.as_slice());
    Ok(slice.to_vec())
}

// =============================================================================
// NIF Registration
// =============================================================================

rustler::init!("Elixir.ExPhil.Native.RustLinearScan");
