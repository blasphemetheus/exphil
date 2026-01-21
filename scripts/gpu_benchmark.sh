#!/bin/bash
# GPU Architecture Benchmark for ExPhil
#
# Benchmarks all architectures (MLP, Mamba, LSTM, GRU, Attention, Jamba) on GPU
# and generates a comparison report.
#
# Usage:
#   ./scripts/gpu_benchmark.sh --replays ./replays
#   ./scripts/gpu_benchmark.sh --replays ./replays --max-files 50 --epochs 5
#
# All arguments are passed through to benchmark_architectures.exs

set -e

# Set environment for GPU training
export EXLA_TARGET=cuda
export TF_CPP_MIN_LOG_LEVEL=3  # Suppress verbose XLA logs
export XLA_CACHE_PATH="${XLA_CACHE_PATH:-/tmp/exphil_xla_cache}"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║              ExPhil GPU Architecture Benchmark                  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check for nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader | head -1
    echo ""
else
    echo "Warning: nvidia-smi not found"
    echo ""
fi

echo "Environment:"
echo "  EXLA_TARGET=$EXLA_TARGET"
echo "  XLA_CACHE_PATH=$XLA_CACHE_PATH"
echo ""

# Default to larger batch for GPU if not specified
if [[ ! "$*" =~ "--batch-size" ]]; then
    echo "Note: Using default GPU batch size (256). Override with --batch-size N"
fi

echo "Running: mix run scripts/benchmark_architectures.exs $@"
echo ""

# Run benchmark
exec mix run scripts/benchmark_architectures.exs "$@"
