#!/bin/bash
# Build the selective scan NIF
#
# Usage:
#   ./build.sh           # Build without CUDA (CPU fallback only)
#   ./build.sh --cuda    # Build with CUDA support (requires CUDA toolkit)
#
# The built library will be copied to priv/native/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/priv/native"

echo "Building selective_scan_nif..."
echo "  Project root: $PROJECT_ROOT"
echo "  Output dir: $OUTPUT_DIR"

# Parse arguments
CUDA_FEATURE=""
if [[ "$1" == "--cuda" ]]; then
    echo "  CUDA support: enabled"
    CUDA_FEATURE="--features cuda"

    # Check for CUDA
    if ! command -v nvcc &> /dev/null; then
        echo "ERROR: nvcc not found. Install CUDA toolkit or use 'cargo build' without --features cuda"
        exit 1
    fi
    echo "  CUDA version: $(nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')"
else
    echo "  CUDA support: disabled (CPU fallback)"
fi

# Build
cd "$SCRIPT_DIR"
cargo build --release $CUDA_FEATURE

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Copy library
if [[ -f "target/release/libselective_scan_nif.so" ]]; then
    cp target/release/libselective_scan_nif.so "$OUTPUT_DIR/"
    echo ""
    echo "SUCCESS: Built $OUTPUT_DIR/libselective_scan_nif.so"
elif [[ -f "target/release/libselective_scan_nif.dylib" ]]; then
    cp target/release/libselective_scan_nif.dylib "$OUTPUT_DIR/libselective_scan_nif.so"
    echo ""
    echo "SUCCESS: Built $OUTPUT_DIR/libselective_scan_nif.so (from .dylib)"
else
    echo "ERROR: Could not find built library"
    ls -la target/release/ | grep selective
    exit 1
fi

# Quick test
echo ""
echo "Testing NIF load..."
cd "$PROJECT_ROOT"
mix run -e '
  case ExPhil.Native.SelectiveScan.available?() do
    true ->
      IO.puts("✓ NIF loaded successfully")
      IO.puts("  CUDA available: #{ExPhil.Native.SelectiveScan.cuda_available?()}")
    false ->
      IO.puts("✗ NIF failed to load")
  end
'
