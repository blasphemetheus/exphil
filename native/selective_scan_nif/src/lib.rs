//! CUDA-accelerated Selective Scan NIF for Mamba SSM
//!
//! This NIF provides a fast implementation of the selective scan operation
//! used in Mamba state space models.
//!
//! # Building
//!
//! Requires CUDA toolkit installed. Build with:
//! ```bash
//! cd native/selective_scan_nif
//! cargo build --release --features cuda
//! ```
//!
//! # Usage from Elixir
//!
//! ```elixir
//! alias ExPhil.Native.SelectiveScan
//!
//! result = SelectiveScan.scan(x_binary, dt_binary, a_binary, b_binary, c_binary, shape)
//! ```

use rustler::{Binary, Env, NifResult, OwnedBinary, Error};
use std::io::Write;

mod kernel;

// =============================================================================
// NIF Functions
// =============================================================================

/// Perform selective scan on GPU
///
/// # Arguments
/// * `x` - Input tensor binary [batch, seq_len, hidden], f32
/// * `dt` - Delta/timestep binary [batch, seq_len, hidden], f32
/// * `a` - State transition binary [hidden, state], f32
/// * `b` - Input projection binary [batch, seq_len, state], f32
/// * `c` - Output projection binary [batch, seq_len, state], f32
/// * `shape` - Tuple of (batch, seq_len, hidden, state)
///
/// # Returns
/// * Output tensor binary [batch, seq_len, hidden], f32
#[rustler::nif(schedule = "DirtyCpu")]
fn selective_scan<'a>(
    env: Env<'a>,
    x: Binary<'a>,
    dt: Binary<'a>,
    a: Binary<'a>,
    b: Binary<'a>,
    c: Binary<'a>,
    shape: (usize, usize, usize, usize),
) -> NifResult<OwnedBinary> {
    let (batch, seq_len, hidden, state) = shape;

    // Convert binaries to f32 slices
    let x_data = binary_to_f32_slice(&x)?;
    let dt_data = binary_to_f32_slice(&dt)?;
    let a_data = binary_to_f32_slice(&a)?;
    let b_data = binary_to_f32_slice(&b)?;
    let c_data = binary_to_f32_slice(&c)?;

    // Validate sizes
    let expected_x = batch * seq_len * hidden;
    let expected_dt = batch * seq_len * hidden;
    let expected_a = hidden * state;
    let expected_b = batch * seq_len * state;
    let expected_c = batch * seq_len * state;

    if x_data.len() != expected_x {
        return Err(Error::Term(Box::new(format!(
            "x size mismatch: expected {}, got {}", expected_x, x_data.len()
        ))));
    }
    if dt_data.len() != expected_dt {
        return Err(Error::Term(Box::new("dt size mismatch")));
    }
    if a_data.len() != expected_a {
        return Err(Error::Term(Box::new("a size mismatch")));
    }
    if b_data.len() != expected_b {
        return Err(Error::Term(Box::new("b size mismatch")));
    }
    if c_data.len() != expected_c {
        return Err(Error::Term(Box::new("c size mismatch")));
    }

    // Perform computation
    let result = kernel::selective_scan_cuda(
        x_data, dt_data, a_data, b_data, c_data,
        batch, seq_len, hidden, state
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

/// Get CUDA device info
#[rustler::nif]
fn cuda_device_info() -> NifResult<String> {
    kernel::get_device_info()
        .map_err(|e| Error::Term(Box::new(e.to_string())))
}

/// Simple ping for testing NIF loading
#[rustler::nif]
fn ping() -> &'static str {
    "pong from selective_scan_nif"
}

// =============================================================================
// Helpers
// =============================================================================

fn binary_to_f32_slice(binary: &Binary) -> NifResult<&[f32]> {
    if binary.len() % 4 != 0 {
        return Err(Error::Term(Box::new("Binary length not multiple of 4 (f32)")));
    }
    Ok(bytemuck::cast_slice(binary.as_slice()))
}

// =============================================================================
// NIF Registration
// =============================================================================

rustler::init!(
    "Elixir.ExPhil.Native.SelectiveScan",
    [
        selective_scan,
        cuda_available,
        cuda_device_info,
        ping
    ]
);
