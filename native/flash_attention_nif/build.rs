//! Build script for FlashAttention NIF
//!
//! When the `cuda` feature is enabled, this compiles the CUDA kernels
//! using nvcc and links them with the Rust NIF.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Only compile CUDA when the feature is enabled
    #[cfg(feature = "cuda")]
    compile_cuda();

    // Tell Cargo to rerun if these files change
    println!("cargo:rerun-if-changed=cuda/flash_attention.cu");
    println!("cargo:rerun-if-changed=build.rs");
}

#[cfg(feature = "cuda")]
fn compile_cuda() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Detect CUDA compute capability
    let compute_cap = detect_compute_cap().unwrap_or_else(|| {
        // Default to Ampere (RTX 30xx)
        eprintln!("Warning: Could not detect GPU compute capability, defaulting to sm_86");
        "86".to_string()
    });

    let cuda_file = "cuda/flash_attention.cu";
    let object_file = out_dir.join("flash_attention.o");
    let lib_file = out_dir.join("libflash_attention_cuda.a");

    // Compile CUDA to object file
    let nvcc_status = Command::new("nvcc")
        .args([
            "-c",
            cuda_file,
            "-o",
            object_file.to_str().unwrap(),
            "-O3",
            "-std=c++17",
            &format!("-arch=sm_{}", compute_cap),
            "-Xcompiler", "-fPIC",
            "--expt-relaxed-constexpr",
        ])
        .status()
        .expect("Failed to execute nvcc. Is CUDA toolkit installed?");

    if !nvcc_status.success() {
        panic!("nvcc compilation failed");
    }

    // Create static library
    let ar_status = Command::new("ar")
        .args([
            "rcs",
            lib_file.to_str().unwrap(),
            object_file.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to execute ar");

    if !ar_status.success() {
        panic!("ar failed to create static library");
    }

    // Link directives
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=flash_attention_cuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=stdc++");

    // CUDA library path
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    } else {
        // Common CUDA installation paths
        for path in ["/usr/local/cuda/lib64", "/opt/cuda/lib64"] {
            if std::path::Path::new(path).exists() {
                println!("cargo:rustc-link-search=native={}", path);
                break;
            }
        }
    }
}

#[cfg(feature = "cuda")]
fn detect_compute_cap() -> Option<String> {
    // Try environment variable first
    if let Ok(cap) = env::var("CUDA_COMPUTE_CAP") {
        return Some(cap);
    }

    // Try nvidia-smi
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;

    if output.status.success() {
        let cap = String::from_utf8_lossy(&output.stdout);
        let cap = cap.trim().replace('.', "");
        if !cap.is_empty() {
            return Some(cap);
        }
    }

    None
}
