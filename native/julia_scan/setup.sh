#!/usr/bin/env bash
# Setup Julia dependencies for linear scan server
#
# Usage:
#   cd native/julia_scan && bash setup.sh
#
# On NixOS/WSL2 with CUDA, you may need:
#   export JULIA_CUDA_USE_BINARYBUILDER=false
# to use the system CUDA toolkit instead of Julia's artifact.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Julia Linear Scan Setup ==="
echo "Project: $SCRIPT_DIR"

# Check Julia is available
if ! command -v julia &>/dev/null; then
    echo "ERROR: julia not found. Add julia to shell.nix buildInputs."
    exit 1
fi

echo "Julia version: $(julia --version)"

# Check CUDA visibility
if command -v nvidia-smi &>/dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
fi

# Install Julia packages
echo ""
echo "Installing Julia packages..."
julia --project="$SCRIPT_DIR" -e '
using Pkg
Pkg.instantiate()
println("Packages installed successfully")

# Test CUDA availability
try
    using CUDA
    if CUDA.functional()
        println("CUDA functional: $(CUDA.name(CUDA.device()))")
    else
        println("CUDA not functional (CPU-only mode)")
    end
catch e
    println("CUDA.jl load failed: $e")
    println("Set JULIA_CUDA_USE_BINARYBUILDER=false to use system CUDA")
end
'

echo ""
echo "Setup complete. Test with:"
echo "  julia --project=$SCRIPT_DIR -e 'include(\"kernels.jl\"); println(\"OK\")'"
