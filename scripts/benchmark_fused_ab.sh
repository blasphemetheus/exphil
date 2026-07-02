#!/usr/bin/env bash
# A/B Benchmark: Fused CUDA Kernels vs Nx Fallback
#
# Runs the same architectures twice:
#   A) EDIFICE_DISABLE_FUSED=0 — fused CUDA custom_call kernels (stablehlo.custom_call)
#   B) EDIFICE_DISABLE_FUSED=1 — pure Nx fallback (EXLA-compiled scan loop)
#
# Usage:
#   bash scripts/benchmark_fused_ab.sh [options]
#
# Options:
#   --replays PATH    Replay directory (default: ./replays/mewtwo)
#   --epochs N        Epochs per run (default: 3)
#   --archs LIST      Comma-separated architectures (default: min_gru,min_lstm,mamba,gru,deltanet)
#   --batch-size N    Batch size (default: 128)
#
# Requires: EDIFICE_LOCAL_NX=1 (local EXLA fork with fused kernel support)

set -euo pipefail

# Defaults
REPLAYS="./replays/mewtwo"
EPOCHS=3
ARCHS="min_gru,min_lstm,mamba,gru,deltanet"
BATCH_SIZE=128

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --replays) REPLAYS="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --archs) ARCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
RESULTS_DIR="checkpoints"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

echo "============================================"
echo "  Fused CUDA Kernel A/B Benchmark"
echo "============================================"
echo "Architectures: $ARCHS"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Replays: $REPLAYS"
echo "Timestamp: $TIMESTAMP"
echo ""

# Ensure we're using local EXLA fork
if [ "${EDIFICE_LOCAL_NX:-}" != "1" ]; then
  echo "WARNING: EDIFICE_LOCAL_NX not set. Setting it now."
  export EDIFICE_LOCAL_NX=1
fi

# Run A: Fused CUDA kernels enabled
echo "============================================"
echo "  [A] FUSED CUDA KERNELS (custom_call)"
echo "============================================"
echo ""

export EDIFICE_DISABLE_FUSED=0

mix run scripts/benchmark_architectures.exs \
  --replays "$REPLAYS" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --only "$ARCHS" \
  --cache-embeddings \
  --quiet \
  2>&1 | tee "$LOG_DIR/fused_ab_A_${TIMESTAMP}.log"

# Save A results
if [ -f "$RESULTS_DIR/benchmark_results_latest.json" ]; then
  cp "$(readlink -f "$RESULTS_DIR/benchmark_results_latest.json")" \
     "$RESULTS_DIR/fused_ab_A_${TIMESTAMP}.json"
  echo ""
  echo "[A] Results saved to $RESULTS_DIR/fused_ab_A_${TIMESTAMP}.json"
fi

# GC pause between runs
echo ""
echo "Pausing 5s between runs for GPU memory cleanup..."
sleep 5

# Run B: Fused kernels disabled (Nx fallback)
echo ""
echo "============================================"
echo "  [B] NX FALLBACK (no custom_call)"
echo "============================================"
echo ""

export EDIFICE_DISABLE_FUSED=1

mix run scripts/benchmark_architectures.exs \
  --replays "$REPLAYS" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --only "$ARCHS" \
  --cache-embeddings \
  --quiet \
  2>&1 | tee "$LOG_DIR/fused_ab_B_${TIMESTAMP}.log"

# Save B results
if [ -f "$RESULTS_DIR/benchmark_results_latest.json" ]; then
  cp "$(readlink -f "$RESULTS_DIR/benchmark_results_latest.json")" \
     "$RESULTS_DIR/fused_ab_B_${TIMESTAMP}.json"
  echo ""
  echo "[B] Results saved to $RESULTS_DIR/fused_ab_B_${TIMESTAMP}.json"
fi

# Compare results
echo ""
echo "============================================"
echo "  COMPARISON"
echo "============================================"
echo ""

python3 -c "
import json, sys

a_path = '$RESULTS_DIR/fused_ab_A_${TIMESTAMP}.json'
b_path = '$RESULTS_DIR/fused_ab_B_${TIMESTAMP}.json'

try:
    with open(a_path) as f:
        a_data = json.load(f)
    with open(b_path) as f:
        b_data = json.load(f)
except FileNotFoundError as e:
    print(f'Could not load results: {e}')
    sys.exit(1)

a_results = {r['id']: r for r in a_data['results']}
b_results = {r['id']: r for r in b_data['results']}

common = set(a_results.keys()) & set(b_results.keys())
if not common:
    print('No common architectures found between A and B runs.')
    sys.exit(1)

hdr = f\"{'Architecture':<25} {'Fused B/s':>10} {'Fallback B/s':>12} {'Speedup':>8} {'F.Val':>7} {'FB.Val':>7}\"
print(hdr)
print('-' * len(hdr))

for arch_id in sorted(common):
    a = a_results[arch_id]
    b = b_results[arch_id]

    # Use epoch 2+3 avg (post-JIT) for fair comparison
    a_epochs = a.get('epochs', [])
    b_epochs = b.get('epochs', [])

    if len(a_epochs) >= 3:
        a_bps = (a_epochs[1]['batches_per_sec'] + a_epochs[2]['batches_per_sec']) / 2
    else:
        a_bps = a['avg_batches_per_sec']

    if len(b_epochs) >= 3:
        b_bps = (b_epochs[1]['batches_per_sec'] + b_epochs[2]['batches_per_sec']) / 2
    else:
        b_bps = b['avg_batches_per_sec']

    speedup = a_bps / b_bps if b_bps > 0 else 0
    a_vl = a['final_val_loss']
    b_vl = b['final_val_loss']

    name = (a.get('name') or arch_id)[:25]
    marker = ' <<<' if speedup > 1.1 else ''
    print(f'{name:<25} {a_bps:>10.1f} {b_bps:>12.1f} {speedup:>7.2f}x {a_vl:>7.3f} {b_vl:>7.3f}{marker}')

print()
print('Speedup > 1.0 = fused kernels faster')
print('<<< marks architectures with >10% speedup')
"

echo ""
echo "Full logs:"
echo "  [A] Fused:    $LOG_DIR/fused_ab_A_${TIMESTAMP}.log"
echo "  [B] Fallback: $LOG_DIR/fused_ab_B_${TIMESTAMP}.log"
echo "  [A] Results:  $RESULTS_DIR/fused_ab_A_${TIMESTAMP}.json"
echo "  [B] Results:  $RESULTS_DIR/fused_ab_B_${TIMESTAMP}.json"
