#!/usr/bin/env bash
# nx-bug isolation probe (2026-08-20): 20-epoch g18-recipe run at the
# CURRENT nx checkout, classify calm/wild from the loss trajectory.
# Bradley's call 08-20: use correct gradients; isolate the nx bug.
#
# PRE-REGISTERED classifier (validated on all four 2e-4 runs, see
# eval_runs/0819_nxpin_probe/RESULTS.md): WILD = any one-epoch loss
# ratio > 8x within 20 epochs, or any epoch loss < 1e-4. Calm base
# a7497612 max ratio was 4.2x; HEAD runs hit 10.7x/153x/collapse by
# ep13. Gray zone 4-10x -> rerun before trusting the label.
#
# Usage: run_probe.sh <label>   (nx checkout is the caller's job; the
# script records it). No gates — dynamics only.
set -euo pipefail
cd "$(dirname "$0")/../.."
LABEL="${1:?usage: run_probe.sh <label>}"

NX_DESC=$(git -C "$HOME/git/nx" log --oneline -1 | head -1)
OUT=eval_runs/0820_nx_bisect
mkdir -p "$OUT"

export DOLPHIN_DIR="$HOME/.local/share/slippi/exi-ai/dolphin-emu-headless"
export ISO="$HOME/isos/melee.iso"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_SKIP_NIF_COMPILE=1

ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp,eval_runs/0802_d2pool/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

echo "=== PROBE ${LABEL} START $(date +%H:%M:%S) nx=[${NX_DESC}]"
EXPHIL_GPU_MEMORY_FRACTION=0.75 mix run scripts/dagger_drill.exs \
  --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
  --rollouts "$ROLL" --opening-replays "$OPEN" \
  --snippet-frames "eval_runs/0804_snippets_human_ad2/snippets.frames" \
  --opp-scramble-frames 12000 \
  --max-epochs 20 --prev-action-dropout 0.6 --transition-weight 2.0 \
  --multi-delay "2,3,4" --pipeline-offset 2 --queue-depth 4 --with-delay-id \
  --scheduled-sampling 0.5 --ss-ramp 10 --awbc \
  --lr 2.0e-4 \
  --out "checkpoints/nx_bisect_${LABEL}.bin" \
  2>&1 | tee "$OUT/${LABEL}_train.log" \
  | grep -aE "COLLAPSE|Converged|diverged|error" | tail -4

# Classify from the trajectory
grep -aoE "epoch [0-9]+/20: loss=[0-9.e-]+" "$OUT/${LABEL}_train.log" \
  | awk -F'loss=' '{print $2}' > "$OUT/${LABEL}_losses.txt"
python3 - "$OUT/${LABEL}_losses.txt" "$LABEL" << 'EOF'
import sys
losses = [float(x) for x in open(sys.argv[1])]
label = sys.argv[2]
ratios = [max(a, b) / max(min(a, b), 1e-30) for a, b in zip(losses, losses[1:])]
mx = max(ratios) if ratios else 0
tiny = min(losses) if losses else 1
verdict = "WILD" if (mx > 8 or tiny < 1e-4) else ("GRAY" if mx > 4 else "CALM")
print(f"=== PROBE {label} VERDICT: {verdict} (max one-epoch ratio {mx:.1f}x, "
      f"min loss {tiny:.2e}, {len(losses)} epochs)")
EOF
echo "=== PROBE ${LABEL} DONE $(date +%H:%M:%S)"