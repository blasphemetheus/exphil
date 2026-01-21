#!/bin/bash
# GPU Training Wrapper for ExPhil
#
# Convenience script for running training on NVIDIA GPUs (4090, 3090, etc.)
# Sets up the proper environment variables for CUDA acceleration.
#
# Usage:
#   ./scripts/gpu_train.sh --preset rtx4090_quick --replays ./replays
#   ./scripts/gpu_train.sh --preset rtx4090_standard --replays ./replays --wandb
#
# All arguments are passed through to train_from_replays.exs

set -e

# Set environment for GPU training
export EXLA_TARGET=cuda
export TF_CPP_MIN_LOG_LEVEL=3  # Suppress verbose XLA logs

# Optional: Set XLA cache for faster restarts
export XLA_CACHE_PATH="${XLA_CACHE_PATH:-/tmp/exphil_xla_cache}"

# Print GPU info
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    ExPhil GPU Training                          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check for nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1
    echo ""
else
    echo "Warning: nvidia-smi not found, GPU info unavailable"
    echo ""
fi

echo "Environment:"
echo "  EXLA_TARGET=$EXLA_TARGET"
echo "  XLA_CACHE_PATH=$XLA_CACHE_PATH"
echo ""
echo "Running: mix run scripts/train_from_replays.exs $@"
echo ""

# Run training
exec mix run scripts/train_from_replays.exs "$@"
