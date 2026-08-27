#!/usr/bin/env bash
# Stage-2 retune sweep (HANDOFF 08-16 addendum): LR grid on the fixture
# drill, champion flags + --awbc (the g16 lineage), fixed-grad stack.
# Selection signal: epoch-2/3 loss + divergence check. Fixture loss is
# an OPTIMIZATION-HEALTH filter only — it never ranks deploy strength.
# Baseline lr = 2.0e-4 (dagger_drill default, tuned in the wrong-grad era).
set -uo pipefail
cd "$(dirname "$0")/../.."

OUT=eval_runs/0817_lr_sweep_fixture
SCRATCH=/tmp/claude-1000/-home-blewf-git-exphil/a1b28bc5-1a86-4d18-b4d4-02db3d1de64c/scratchpad

for LR in 2.0e-4 4.0e-4 8.0e-4; do
  echo "=== ARM lr=$LR start $(date +%H:%M:%S)"
  EXPHIL_GPU_MEMORY_FRACTION=0.5 mix run scripts/dagger_drill.exs \
    --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
    --max-epochs 3 --prev-action-dropout 0.6 --transition-weight 2.0 \
    --multi-delay "2,3,4" --pipeline-offset 2 --queue-depth 4 --with-delay-id \
    --scheduled-sampling 0.5 --ss-ramp 10 --awbc \
    --lr "$LR" \
    --out "$SCRATCH/lr_sweep_$LR.bin" \
    2>&1 | tee "$OUT/arm_$LR.log" | grep -aE "epoch [0-9]/|Converged|diverged|error|\*\*" | tail -4
done
echo "=== SWEEP DONE $(date +%H:%M:%S)"
grep -aH "epoch" "$OUT"/arm_*.log | grep -aE "epoch [23]/"
