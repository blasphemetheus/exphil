#!/bin/bash
# Run each architecture in its own BEAM process to prevent XLA JIT cache
# accumulation. Each process starts with a clean GPU, avoiding the 90%
# VRAM saturation that occurs when 30+ architectures share a process.
#
# Usage:
#   ./scripts/benchmark_isolated.sh --replays /workspace/replays [flags]
#   ./scripts/benchmark_isolated.sh --replays /workspace/replays --grad-norms
#   ./scripts/benchmark_isolated.sh --replays /workspace/replays --only mlp,mamba,lstm
#   ./scripts/benchmark_isolated.sh --replays /workspace/replays --skip perceiver,ttt
#
# All flags except --only/--skip are passed through to benchmark_architectures.exs.
# --cache-embeddings is automatically enabled (required for isolated mode).
# --continue-on-error is automatically enabled (each arch is its own process).
#
# Results are merged at the end into a single JSON + ranking table.

set -euo pipefail

# All 34 architectures in default order
ALL_ARCHS="mlp gated_ssm mamba jamba lstm gru lstm_hybrid sliding_window zamba mamba_ssd s5 rwkv gla hgrn s4 s4d h3 griffin hawk xlstm retnet performer deltanet fnet perceiver ttt reservoir hopfield ntm liquid decision_transformer kan snn bayesian"

# Parse --only/--skip from args, pass everything else through
ONLY=""
SKIP=""
PASSTHROUGH=()
PREV=""

for arg in "$@"; do
  if [ "$PREV" = "--only" ]; then
    ONLY="$arg"
    PREV=""
    continue
  elif [ "$PREV" = "--skip" ]; then
    SKIP="$arg"
    PREV=""
    continue
  fi

  case "$arg" in
    --only) PREV="--only" ;;
    --skip) PREV="--skip" ;;
    *) PASSTHROUGH+=("$arg") ;;
  esac
done

# Filter architecture list
if [ -n "$ONLY" ]; then
  ARCHS=$(echo "$ONLY" | tr ',' ' ')
else
  ARCHS="$ALL_ARCHS"
fi

if [ -n "$SKIP" ]; then
  for skip_arch in $(echo "$SKIP" | tr ',' ' '); do
    ARCHS=$(echo "$ARCHS" | tr ' ' '\n' | grep -v "^${skip_arch}$" | tr '\n' ' ')
  done
fi

# Trim whitespace
ARCHS=$(echo "$ARCHS" | xargs)
NUM_ARCHS=$(echo "$ARCHS" | wc -w)

# Setup directories
PARTIAL_DIR=$(mktemp -d /tmp/exphil_bench_XXXXXX)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
mkdir -p "$LOG_DIR" checkpoints

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║          ExPhil Isolated Architecture Benchmark                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Architectures: $NUM_ARCHS"
echo "  List: $ARCHS"
echo "  Partial results: $PARTIAL_DIR"
echo "  Pass-through flags: ${PASSTHROUGH[*]:-none}"
echo ""

if command -v nvidia-smi &> /dev/null; then
  echo "GPU:"
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | head -1
  echo ""
fi

TOTAL_START=$(date +%s)
SUCCEEDED=0
FAILED=0
FAILED_LIST=""

for arch in $ARCHS; do
  IDX=$((SUCCEEDED + FAILED + 1))
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "  [$IDX/$NUM_ARCHS] $arch"
  echo "═══════════════════════════════════════════════════════════════"

  ARCH_START=$(date +%s)
  ARCH_LOG="$LOG_DIR/bench_${arch}_${TIMESTAMP}.log"

  # Run in isolated process with embedding cache
  if mix run scripts/benchmark_architectures.exs \
    --only "$arch" \
    --cache-embeddings \
    --continue-on-error \
    "${PASSTHROUGH[@]}" \
    2>&1 | tee "$ARCH_LOG"; then

    # Copy result if it exists
    if [ -L "checkpoints/benchmark_results_latest.json" ] || [ -f "checkpoints/benchmark_results_latest.json" ]; then
      cp -L checkpoints/benchmark_results_latest.json "$PARTIAL_DIR/${arch}.json" 2>/dev/null || true
    fi

    ARCH_END=$(date +%s)
    ARCH_ELAPSED=$((ARCH_END - ARCH_START))
    echo "  [$arch completed in ${ARCH_ELAPSED}s]"
    SUCCEEDED=$((SUCCEEDED + 1))
  else
    ARCH_END=$(date +%s)
    ARCH_ELAPSED=$((ARCH_END - ARCH_START))
    echo "  [$arch FAILED after ${ARCH_ELAPSED}s]"
    FAILED=$((FAILED + 1))
    FAILED_LIST="$FAILED_LIST $arch"
  fi

  # Show GPU memory between runs to verify it's being freed
  if command -v nvidia-smi &> /dev/null; then
    echo "  GPU memory after $arch: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | head -1)"
  fi
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Benchmark Complete"
echo "  Total time: ${TOTAL_ELAPSED}s ($((TOTAL_ELAPSED / 60))m $((TOTAL_ELAPSED % 60))s)"
echo "  Succeeded: $SUCCEEDED / $NUM_ARCHS"
if [ $FAILED -gt 0 ]; then
  echo "  Failed:$FAILED_LIST"
fi
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Count partial results
NUM_PARTIAL=$(ls "$PARTIAL_DIR"/*.json 2>/dev/null | wc -l)
if [ "$NUM_PARTIAL" -gt 0 ]; then
  echo "Merging $NUM_PARTIAL results..."
  mix run scripts/merge_benchmark_results.exs \
    --dir "$PARTIAL_DIR" \
    --output "checkpoints/benchmark_results_${TIMESTAMP}.json"
else
  echo "No results to merge (all architectures failed)."
fi

# Cleanup
rm -rf "$PARTIAL_DIR"
echo "Done."
