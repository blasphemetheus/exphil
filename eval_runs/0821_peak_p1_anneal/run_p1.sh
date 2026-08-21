#!/usr/bin/env bash
# PEAK SCIENCE P1 — anneal-at-peak: warm-start from ms_g19_ep4 (the
# 437.4 c438 peak), drop lr 10x to 2e-5, train 15 epochs, sweep every
# epoch. Question: do SMALL steps stay on the peak?
#
# PRE-REGISTERED (2026-08-21, before the run):
#   Known: full-lr continuation from a peak destroys it within ~2
#   epochs (0820_collapse_forensics: ep50 362.5 -> ~87 on both stacks;
#   fresh-optimizer warm start did not save it, so optimizer state is
#   not the lever — step SIZE is the remaining variable).
#   SS note: runs --ss-ramp 1 (full scheduled sampling from epoch 1).
#   ep4 itself trained at ss_p=0.2 (ramp 4/10); surviving at FULL ss
#   pressure + low lr is the stronger result and matches how any
#   "anneal phase" would deploy.
#   Reads:
#     P1-STAY   most epochs gate >=390 -> small steps preserve peaks;
#               "find peak -> anneal" becomes a two-phase recipe and
#               ends the stopping-point lottery. Follow-up: does
#               anneal IMPROVE (consolidate d2/d4/mewtwo profile)?
#     P1-DRIFT  gates decay over epochs (390 -> basin) -> peak-leaving
#               is not step-size-driven; the attractor pulls at any lr
#               -> P2's SS mechanism becomes the prime suspect.
#     P1-HOP    immediate basin (epoch 1 already ~90) -> the peak is
#               unstable to ANY continued training; only selection
#               (sweep) can harvest peaks. Anneal idea dead.
#   NO CROWN from stand numbers (g6 rule).
set -euo pipefail
cd "$(dirname "$0")/../.."

export DOLPHIN_DIR="$HOME/.local/share/slippi/exi-ai/dolphin-emu-headless"
export ISO="$HOME/isos/melee.iso"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_SKIP_NIF_COMPILE=1

OUT=eval_runs/0821_peak_p1_anneal
ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp,eval_runs/0802_d2pool/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

echo "=== P1 TRAIN $(date +%H:%M:%S) (anneal-at-peak: init ms_g19_ep4, lr 2e-5, 15ep)"
EXPHIL_GPU_MEMORY_FRACTION=0.75 mix run scripts/dagger_drill.exs \
  --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
  --rollouts "$ROLL" --opening-replays "$OPEN" \
  --snippet-frames "eval_runs/0804_snippets_human_ad2/snippets.frames" \
  --opp-scramble-frames 12000 \
  --max-epochs 15 --prev-action-dropout 0.6 --transition-weight 2.0 \
  --multi-delay "2,3,4" --pipeline-offset 2 --queue-depth 4 --with-delay-id \
  --scheduled-sampling 0.5 --ss-ramp 1 --awbc \
  --lr 2.0e-5 \
  --target-loss 1.0e-9 \
  --init-from checkpoints/ms_g19_ep4.bin \
  --snapshot-all \
  --out checkpoints/ms_p1_anneal.bin \
  2>&1 | tee "$OUT/train.log" \
  | grep -aE "AWBC|COLLAPSE|Warm-started|Converged|diverged|exported|error" | tail -6
[ -f checkpoints/ms_p1_anneal.bin ] || { echo "=== P1 TRAIN FAILED" >&2; exit 1; }

echo "=== P1 GATE-SWEEP $(date +%H:%M:%S)"
bash scripts/gate_sweep.sh checkpoints/ms_p1_anneal "$OUT/sweep" --confirm
echo "=== P1 DONE $(date +%H:%M:%S). Score vs P1-STAY/DRIFT/HOP."