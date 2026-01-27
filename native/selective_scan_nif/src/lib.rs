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

use rustler::{Binary, NifResult, OwnedBinary, Error};
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
fn selective_scan(
    x: Binary,
    dt: Binary,
    a: Binary,
    b: Binary,
    c: Binary,
    shape: (usize, usize, usize, usize),
) -> NifResult<OwnedBinary> {
    let (batch, seq_len, hidden, state) = shape;

    // Convert binaries to f32 vectors (copies data)
    let x_data = binary_to_f32_vec(&x)?;
    let dt_data = binary_to_f32_vec(&dt)?;
    let a_data = binary_to_f32_vec(&a)?;
    let b_data = binary_to_f32_vec(&b)?;
    let c_data = binary_to_f32_vec(&c)?;

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
        &x_data, &dt_data, &a_data, &b_data, &c_data,
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

/// Forward pass that saves hidden states (for training backward pass)
///
/// Returns: packed binary containing [output, hidden_states] concatenated
/// - output: [batch, seq_len, hidden], f32
/// - hidden_states: [batch, seq_len, hidden, state], f32
///
/// The caller must know the sizes to split the binary:
/// - out_size = batch * seq_len * hidden * 4 bytes
/// - h_all_size = batch * seq_len * hidden * state * 4 bytes
#[rustler::nif(schedule = "DirtyCpu")]
fn selective_scan_forward_with_states(
    x: Binary,
    dt: Binary,
    a: Binary,
    b: Binary,
    c: Binary,
    shape: (usize, usize, usize, usize),
) -> NifResult<OwnedBinary> {
    let (batch, seq_len, hidden, state) = shape;

    let x_data = binary_to_f32_vec(&x)?;
    let dt_data = binary_to_f32_vec(&dt)?;
    let a_data = binary_to_f32_vec(&a)?;
    let b_data = binary_to_f32_vec(&b)?;
    let c_data = binary_to_f32_vec(&c)?;

    // Validate sizes
    let expected_x = batch * seq_len * hidden;
    let expected_a = hidden * state;
    let expected_bc = batch * seq_len * state;

    if x_data.len() != expected_x {
        return Err(Error::Term(Box::new(format!(
            "x size mismatch: expected {}, got {}", expected_x, x_data.len()
        ))));
    }
    if dt_data.len() != expected_x {
        return Err(Error::Term(Box::new("dt size mismatch")));
    }
    if a_data.len() != expected_a {
        return Err(Error::Term(Box::new("a size mismatch")));
    }
    if b_data.len() != expected_bc {
        return Err(Error::Term(Box::new("b size mismatch")));
    }
    if c_data.len() != expected_bc {
        return Err(Error::Term(Box::new("c size mismatch")));
    }

    let (out, h_all) = kernel::selective_scan_forward_with_states(
        &x_data, &dt_data, &a_data, &b_data, &c_data,
        batch, seq_len, hidden, state
    ).map_err(|e| Error::Term(Box::new(e.to_string())))?;

    // Pack both outputs into a single binary (out first, then h_all)
    let total_bytes = (out.len() + h_all.len()) * 4;
    let mut packed = OwnedBinary::new(total_bytes).unwrap();
    let slice = packed.as_mut_slice();

    let out_bytes = out.len() * 4;
    slice[..out_bytes].copy_from_slice(bytemuck::cast_slice(&out));
    slice[out_bytes..].copy_from_slice(bytemuck::cast_slice(&h_all));

    Ok(packed)
}

/// Backward pass - computes gradients
///
/// Returns: packed binary containing [dx, d_dt, dB, dC] concatenated
/// - dx: [batch, seq_len, hidden], f32
/// - d_dt: [batch, seq_len, hidden], f32
/// - dB: [batch, seq_len, state], f32
/// - dC: [batch, seq_len, state], f32
///
/// The caller must know the sizes to split the binary
#[rustler::nif(schedule = "DirtyCpu")]
fn selective_scan_backward(
    dy: Binary,
    x: Binary,
    h_all: Binary,
    dt: Binary,
    a: Binary,
    b: Binary,
    c: Binary,
    shape: (usize, usize, usize, usize),
) -> NifResult<OwnedBinary> {
    let (batch, seq_len, hidden, state) = shape;

    let dy_data = binary_to_f32_vec(&dy)?;
    let x_data = binary_to_f32_vec(&x)?;
    let h_all_data = binary_to_f32_vec(&h_all)?;
    let dt_data = binary_to_f32_vec(&dt)?;
    let a_data = binary_to_f32_vec(&a)?;
    let b_data = binary_to_f32_vec(&b)?;
    let c_data = binary_to_f32_vec(&c)?;

    // Validate sizes
    let expected_x = batch * seq_len * hidden;
    let expected_h_all = batch * seq_len * hidden * state;
    let expected_a = hidden * state;
    let expected_bc = batch * seq_len * state;

    if dy_data.len() != expected_x {
        return Err(Error::Term(Box::new("dy size mismatch")));
    }
    if x_data.len() != expected_x {
        return Err(Error::Term(Box::new("x size mismatch")));
    }
    if h_all_data.len() != expected_h_all {
        return Err(Error::Term(Box::new(format!(
            "h_all size mismatch: expected {}, got {}", expected_h_all, h_all_data.len()
        ))));
    }
    if dt_data.len() != expected_x {
        return Err(Error::Term(Box::new("dt size mismatch")));
    }
    if a_data.len() != expected_a {
        return Err(Error::Term(Box::new("a size mismatch")));
    }
    if b_data.len() != expected_bc {
        return Err(Error::Term(Box::new("b size mismatch")));
    }
    if c_data.len() != expected_bc {
        return Err(Error::Term(Box::new("c size mismatch")));
    }

    let (dx, d_dt, d_b, d_c) = kernel::selective_scan_backward(
        &dy_data, &x_data, &h_all_data, &dt_data, &a_data, &b_data, &c_data,
        batch, seq_len, hidden, state
    ).map_err(|e| Error::Term(Box::new(e.to_string())))?;

    // Pack all gradients into a single binary: [dx, d_dt, dB, dC]
    let total_bytes = (dx.len() + d_dt.len() + d_b.len() + d_c.len()) * 4;
    let mut packed = OwnedBinary::new(total_bytes).unwrap();
    let slice = packed.as_mut_slice();

    let mut offset = 0;
    let dx_bytes = dx.len() * 4;
    slice[offset..offset + dx_bytes].copy_from_slice(bytemuck::cast_slice(&dx));
    offset += dx_bytes;

    let d_dt_bytes = d_dt.len() * 4;
    slice[offset..offset + d_dt_bytes].copy_from_slice(bytemuck::cast_slice(&d_dt));
    offset += d_dt_bytes;

    let d_b_bytes = d_b.len() * 4;
    slice[offset..offset + d_b_bytes].copy_from_slice(bytemuck::cast_slice(&d_b));
    offset += d_b_bytes;

    slice[offset..].copy_from_slice(bytemuck::cast_slice(&d_c));

    Ok(packed)
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

rustler::init!("Elixir.ExPhil.Native.SelectiveScan");
