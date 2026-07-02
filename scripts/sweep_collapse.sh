#!/usr/bin/env bash
# Run collapse sweep as separate processes (avoids JIT cache OOM)
# Each config gets a fresh BEAM with clean GPU state
#
# Usage: bash scripts/sweep_collapse.sh

echo "=== Mode Collapse Sweep (separate processes) ==="
echo ""

COMMON="--backbone mamba --replays ./replays/huggingface --max-files 200 --batch-size 16 --seed 42 --epochs 2"

run_config() {
    local name="$1"
    shift
    echo "---------- $name ----------"
    mix run scripts/train.exs $COMMON "$@" --name "sweep_$name" 2>&1 | grep -E 'Epoch [0-9]|diversity|pred=.*actual=|COLLAPSE|val_loss|train_loss|loss:|Error|error|SIGBUS|SIGSEGV|ms/it|Starting'
    echo ""
}

run_config "baseline" --focal-gamma 3.0 --learning-rate 1e-4 --lr-schedule cosine_restarts
run_config "gamma1_lr3e4_const" --focal-gamma 1.0 --learning-rate 3e-4 --lr-schedule constant
run_config "gamma1_lr3e4_bw5" --focal-gamma 1.0 --learning-rate 3e-4 --lr-schedule constant --button-weight 5.0

echo "=== Sweep Complete ==="
