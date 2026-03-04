#!/usr/bin/env bash
# Setup Mojo SDK for linear scan kernel
#
# Mojo is pre-1.0 and NOT in nixpkgs. This script attempts installation
# via Modular's installer. On NixOS/WSL2, this has a high chance of failure.
#
# Fallback: The server.py will use NumPy if Mojo is not available.
#
# Usage:
#   cd native/mojo_scan && bash setup.sh

set -euo pipefail

echo "=== Mojo Linear Scan Setup ==="

# Check if Mojo is already installed
if command -v mojo &>/dev/null; then
    echo "Mojo already installed: $(mojo --version)"
    echo "Testing kernel compilation..."
    mojo build linear_scan.mojo -o linear_scan.so --shared 2>/dev/null && \
        echo "Kernel compiled successfully" || \
        echo "Kernel compilation failed (expected on some platforms)"
    exit 0
fi

echo "Mojo not found in PATH."
echo ""
echo "Installation options:"
echo "  1. Install via Modular (Ubuntu/Debian):"
echo "     curl -s https://get.modular.com | sh -"
echo "     modular install mojo"
echo ""
echo "  2. Use Docker (recommended for NixOS):"
echo "     docker run -it --gpus all modular/mojo:latest"
echo ""
echo "  3. Skip Mojo (use NumPy fallback):"
echo "     The benchmark will still work using NumPy as baseline."
echo ""

# Check if we can install via Modular
if command -v modular &>/dev/null; then
    echo "Modular CLI found. Installing Mojo..."
    modular install mojo
    echo "Mojo installed. Re-run this script to verify."
else
    echo "Modular CLI not found."
    echo ""
    echo "On NixOS/WSL2, Mojo installation may not work directly."
    echo "The benchmark script will fall back to NumPy automatically."
    echo ""
    echo "Ensure python3 and msgpack are available:"
    echo "  pip install msgpack numpy"
fi

# Verify Python dependencies
echo ""
echo "Checking Python dependencies..."
python3 -c "import msgpack; import numpy; print('Python deps OK')" 2>/dev/null || \
    echo "Install with: pip install msgpack numpy"
