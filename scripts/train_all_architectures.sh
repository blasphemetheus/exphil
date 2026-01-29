#!/bin/bash
# Train all architectures with identical hyperparameters for fair comparison
#
# Usage:
#   ./scripts/train_all_architectures.sh --character mewtwo --replays /workspace/replays/mewtwo
#   ./scripts/train_all_architectures.sh --character ganondorf --replays /workspace/replays/ganondorf --epochs 30
#
# Runs: MLP, LSTM, GRU, Mamba, Attention, Jamba
# Results saved to checkpoints/ with architecture prefix

set -e

# Default values
CHARACTER="mewtwo"
REPLAYS_DIR="/workspace/replays/mewtwo"
EPOCHS=20
BATCH_SIZE=128
HIDDEN_SIZES="256,256"
WINDOW_SIZE=60
NUM_LAYERS=4
FOCAL_GAMMA=2.0
PATIENCE=7
WARMUP_STEPS=500

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --character)
            CHARACTER="$2"
            shift 2
            ;;
        --replays)
            REPLAYS_DIR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --hidden-sizes)
            HIDDEN_SIZES="$2"
            shift 2
            ;;
        --window-size)
            WINDOW_SIZE="$2"
            shift 2
            ;;
        --num-layers)
            NUM_LAYERS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/architecture_comparison_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "ExPhil Architecture Comparison"
echo "=============================================="
echo "Character: $CHARACTER"
echo "Replays: $REPLAYS_DIR"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Hidden sizes: $HIDDEN_SIZES"
echo "Window size: $WINDOW_SIZE"
echo "Logs: $LOG_DIR"
echo "=============================================="
echo ""

# Common flags for all architectures
COMMON_FLAGS="--replays $REPLAYS_DIR \
  --train-character $CHARACTER \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --hidden-sizes $HIDDEN_SIZES \
  --lr-schedule cosine \
  --warmup-steps $WARMUP_STEPS \
  --early-stopping --patience $PATIENCE \
  --focal-loss --focal-gamma $FOCAL_GAMMA \
  --save-best"

run_training() {
    local arch=$1
    local extra_flags=$2
    local name="${arch}_${CHARACTER}_comparison_${TIMESTAMP}"
    local log_file="$LOG_DIR/${arch}.log"

    echo "[$(date +%H:%M:%S)] Starting $arch training..."

    if [[ -n "$DRY_RUN" ]]; then
        echo "  Would run: mix run scripts/train_from_replays.exs $COMMON_FLAGS $extra_flags --name $name"
        return 0
    fi

    if mix run scripts/train_from_replays.exs $COMMON_FLAGS $extra_flags --name "$name" 2>&1 | tee "$log_file"; then
        echo "[$(date +%H:%M:%S)] ✓ $arch completed successfully"
        return 0
    else
        echo "[$(date +%H:%M:%S)] ✗ $arch failed (see $log_file)"
        return 1
    fi
}

# Track results
declare -A RESULTS

# 1. MLP (no temporal)
echo ""
echo "=== 1/6: MLP (Single-Frame Baseline) ==="
if run_training "mlp" ""; then
    RESULTS["mlp"]="success"
else
    RESULTS["mlp"]="failed"
fi

# 2. LSTM
echo ""
echo "=== 2/6: LSTM ==="
if run_training "lstm" "--temporal --backbone lstm --window-size $WINDOW_SIZE"; then
    RESULTS["lstm"]="success"
else
    RESULTS["lstm"]="failed"
fi

# 3. GRU
echo ""
echo "=== 3/6: GRU ==="
if run_training "gru" "--temporal --backbone gru --window-size $WINDOW_SIZE"; then
    RESULTS["gru"]="success"
else
    RESULTS["gru"]="failed"
fi

# 4. Mamba (recommended for real-time play)
echo ""
echo "=== 4/6: Mamba (Recommended) ==="
if run_training "mamba" "--temporal --backbone mamba --window-size $WINDOW_SIZE --num-layers $NUM_LAYERS"; then
    RESULTS["mamba"]="success"
else
    RESULTS["mamba"]="failed"
fi

# 5. Sliding Window Attention
echo ""
echo "=== 5/6: Sliding Window Attention ==="
if run_training "attention" "--temporal --backbone sliding_window --window-size $WINDOW_SIZE --num-heads 4"; then
    RESULTS["attention"]="success"
else
    RESULTS["attention"]="failed"
fi

# 6. Jamba (Mamba + Attention Hybrid)
echo ""
echo "=== 6/6: Jamba (Hybrid) ==="
if run_training "jamba" "--temporal --backbone jamba --window-size $WINDOW_SIZE --num-layers 6"; then
    RESULTS["jamba"]="success"
else
    RESULTS["jamba"]="failed"
fi

# Summary
echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
echo ""
echo "Results:"
for arch in mlp lstm gru mamba attention jamba; do
    status="${RESULTS[$arch]:-skipped}"
    if [[ "$status" == "success" ]]; then
        echo "  ✓ $arch"
    else
        echo "  ✗ $arch ($status)"
    fi
done
echo ""
echo "Logs saved to: $LOG_DIR"
echo "Checkpoints saved to: checkpoints/"
echo ""
echo "To compare results:"
echo "  mix run scripts/eval_model.exs --compare \\"
for arch in mlp lstm gru mamba attention jamba; do
    echo "    checkpoints/${arch}_${CHARACTER}_comparison_${TIMESTAMP}_best_policy.bin \\"
done | head -n -1
echo ""
