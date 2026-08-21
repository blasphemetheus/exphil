#!/usr/bin/env bash
# g19 = FIRST BEHAVIORAL-GATE-SWEEP ARM: champion recipe + --awbc at
# 2e-4 (g18 recipe), per-epoch snapshots, checkpoint selected by
# post-run gate-sweep instead of loss.
#
# PRE-REGISTERED (2026-08-20, before the run; follows
# eval_runs/0820_collapse_forensics/RESULTS.md — chain skill lives on
# transient peaks, loss anti-correlates with behavior):
#   --target-loss 1e-9 disables the convergence exit (below the
#   collapse guard's 1e-5, so the guard fires first) — the run goes the
#   full 60 epochs to MAP the peak structure, not stop on last-batch
#   lottery. Plateau exit cannot fire (<100-epoch history).
#   References: g16 253.6 c203 (loss-selected, ep58); g18a2 419.4 c415
#   (lucky ep13 exit); mediocre attractor ~87; ep50-peak 362.5.
#   Reads:
#     G1 argmax-gate snapshot >= 304/min (the +20% bar) AND >> the
#        final epoch's gate -> gate-sweep IS the recipe lever; adopt
#        for all future arms (sweep cost ~30 min).
#     G2 peak structure: record argmax epoch, peak height, and WIDTH
#        (#epochs gating >= 300). Narrow peaks (1-2 epochs) mean the
#        recipe needs denser gating; wide peaks mean any near-peak
#        exit works.
#     G3 all snapshots < 304 -> peaks are rarer than one run samples;
#        variance program resumes.
#   Confirmation: argmax re-gated x3 fox + x1 mewtwo (--confirm).
#   NO CROWN from stand numbers (g6 rule).
#
# Run from repo root inside devenv shell. NO-MIX while training.
set -euo pipefail
cd "$(dirname "$0")/../.."

export DOLPHIN_DIR="$HOME/.local/share/slippi/exi-ai/dolphin-emu-headless"
export ISO="$HOME/isos/melee.iso"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_SKIP_NIF_COMPILE=1

OUT=eval_runs/0820_g19_gatesweep
ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp,eval_runs/0802_d2pool/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

echo "=== G19 TRAIN $(date +%H:%M:%S) (champion+awbc, 2e-4, 60ep, snapshot-all, no early exit)"
EXPHIL_GPU_MEMORY_FRACTION=0.75 mix run scripts/dagger_drill.exs \
  --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
  --rollouts "$ROLL" --opening-replays "$OPEN" \
  --snippet-frames "eval_runs/0804_snippets_human_ad2/snippets.frames" \
  --opp-scramble-frames 12000 \
  --max-epochs 60 --prev-action-dropout 0.6 --transition-weight 2.0 \
  --multi-delay "2,3,4" --pipeline-offset 2 --queue-depth 4 --with-delay-id \
  --scheduled-sampling 0.5 --ss-ramp 10 --awbc \
  --lr 2.0e-4 \
  --target-loss 1.0e-9 \
  --snapshot-all \
  --out checkpoints/ms_g19.bin \
  2>&1 | tee "$OUT/train.log" \
  | grep -aE "AWBC|COLLAPSE|Converged|diverged|exported|error" | tail -6
[ -f checkpoints/ms_g19.bin ] || { echo "=== G19 TRAIN FAILED" >&2; exit 1; }

echo "=== G19 GATE-SWEEP $(date +%H:%M:%S)"
bash scripts/gate_sweep.sh checkpoints/ms_g19 "$OUT/sweep" --confirm

echo "=== G19 DONE $(date +%H:%M:%S). Score vs prereg reads G1-G3."