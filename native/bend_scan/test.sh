#!/usr/bin/env bash
# Test Bend linear scan implementation
#
# Runs the scan on tiny inputs to verify correctness.
# Tests all available backends: Rust (single-threaded), C (multi-threaded), CUDA (GPU).
#
# Usage:
#   cd native/bend_scan && bash test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Bend Linear Scan Test ==="

# Check Bend is installed
if ! command -v bend &>/dev/null; then
    echo "ERROR: bend not found. Install with: cargo install bend-lang"
    echo ""
    echo "Prerequisites:"
    echo "  - Rust toolchain (already in shell.nix)"
    echo "  - cargo install bend-lang"
    exit 1
fi

# Ensure ~/.cargo/bin is in PATH (after nix paths to avoid rustup shim conflicts)
export PATH="$PATH:$HOME/.cargo/bin"

echo "Bend version: $(bend --version 2>/dev/null || echo 'unknown')"
echo ""

# Expected output for hidden[0] (a=0.9, b=1.0, h0=0.0):
#   t=0: 1.0, t=1: 1.9, t=2: 2.71, t=3: 3.439
echo "Expected hidden[0]: [1.0, 1.9, 2.71, 3.439]"
echo "Expected hidden[1]: [2.0, 3.6, 4.88, 5.904]"
echo ""

# Test with Rust backend (single-threaded, always works)
echo "--- Rust backend (single-threaded) ---"
bend run linear_scan.bend 2>&1 || echo "FAILED (Rust backend)"

echo ""

# Test with C backend (multi-threaded)
echo "--- C backend (multi-threaded) ---"
bend run-c linear_scan.bend 2>&1 || echo "FAILED (C backend)"

echo ""

# Test with CUDA backend (GPU)
echo "--- CUDA backend (GPU) ---"
bend run-cu linear_scan.bend 2>&1 || echo "FAILED (CUDA backend — expected if no HVM2 GPU support)"

echo ""
echo "=== Test Complete ==="
echo ""
echo "Notes:"
echo "  - Bend uses f24 (24-bit floats), not IEEE 754 f32"
echo "  - Results may differ slightly from CUDA C reference"
echo "  - The CUDA backend uses HVM2's interaction net evaluator, NOT CUDA kernels"
echo "  - This is a learning exercise; production use is not recommended"
