#!/usr/bin/env bash
# CYCLE 4a: platform-competence repair by y-AUGMENTATION (synthetic
# grounded-at-height exemplars; no new recordings).
#
# Basis (all 2026-08-04, INTERP_ROADMAP_V2 log): the absorber's causal
# channel is OWN-Y alone (patch kill test: y->23.45 silences the X head,
# mean −0.99 -> −6.0); the relapse dashboard shows platX_fire = 0.0 for
# EVERY lineage checkpoint including g10b (which avoids platforms but
# cannot JC on them). The multishine cycle is position-independent, so
# +23.45 whole-list shifts are label-preserving training data for exactly
# the missing context.
#
# Recipe: cycle-3b's (aligned human snippets kept) + --y-augment 0.25.
#
# Prereg:
#   P1 dashboard platX_fire(g11) rises well off 0 (>= 0.05) AND stand d3
#      >= 300 => synthetic competence repair works; the y-OOD hole closes
#      without recordings. Bonus if YS chains rise toward the FD level.
#   P2 platX stays ~0, stand holds => altitude alone is not the missing
#      ingredient at training time (dynamics? embedding interaction) —
#      cycle 4b records REAL platform play instead.
#   P3 stand regresses => the augmentation poisons the core cycle;
#      shrink FRAC or gate augmented lists out of clean-cycle batches.
# Scored: dashboard platX (primary), stand chains (1 run, deterministic),
# YS chains (3 runs, run-level). CHAINS never qtrace.
set -uo pipefail
cd "$(dirname "$0")/.."
export ISO="${ISO:-$HOME/isos/melee.iso}"
export DOLPHIN_DIR="$HOME/.config/Slippi Launcher/netplay-beta-nixos"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_SKIP_NIF_COMPILE=1

ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp,eval_runs/0802_d2pool/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

echo "=== CYCLE4A TRAIN $(date +%H:%M:%S)"
EXPHIL_GPU_MEMORY_FRACTION=0.7 mix run scripts/dagger_drill.exs \
  --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
  --rollouts "$ROLL" --opening-replays "$OPEN" \
  --snippet-frames "eval_runs/0804_snippets_human_ad2/snippets.frames" \
  --y-augment 0.25 \
  --max-epochs 60 --prev-action-dropout 0.6 --transition-weight 2.0 \
  --multi-delay "2,3,4" --pipeline-offset 2 --queue-depth 4 --with-delay-id \
  --scheduled-sampling 0.5 --ss-ramp 10 \
  --out checkpoints/ms_g11_yaug.bin \
  2>&1 | grep -aE "Snippets:|Y-augment:|misaligned|Converged|diverged|exported|error|\*\*" | tail -8
[ -f checkpoints/ms_g11_yaug.bin ] || { echo "=== CYCLE4A FAILED" >&2; exit 1; }

echo "=== CYCLE4A GATE 1: dashboard platX (the primary readout) $(date +%H:%M:%S)"
XLA_TARGET=cpu mix run scripts/relapse_dashboard.exs \
  --policies "checkpoints/ms_g11_yaug.bin,checkpoints/ms_g10b_human.bin" \
  --out eval_runs/interp/relapse_dashboard_g11.json 2>&1 | grep -aE "policy|ms_g"

echo "=== CYCLE4A GATE 2: stand d3 (deterministic -> 1 run) $(date +%H:%M:%S)"
EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
  checkpoints/ms_g11_yaug.bin eval_runs/0804_cycle4a_stand \
  --runs 1 --dummy stand --runner sync \
  -- --frame-delay 3 --headless --emulation-speed 0 --blocking-input --slippi-port 51442

echo "=== CYCLE4A GATE 3: Yoshi's Story (3 runs) $(date +%H:%M:%S)"
EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
  checkpoints/ms_g11_yaug.bin eval_runs/0804_cycle4a_ys \
  --runs 3 --dummy stand --runner sync \
  -- --frame-delay 3 --stage yoshis_story --headless --emulation-speed 0 \
     --blocking-input --slippi-port 51442

echo "=== CYCLE4A chains"
for phase in stand ys; do
  echo "--- $phase"
  EXLA_TARGET=host mix run scripts/analyze_shine_source.exs \
    eval_runs/0804_cycle4a_$phase/r*.slp 2>&1 | grep -aE "replay |r[0-9] "
done
echo "=== CYCLE4A rung 0 (opponent-sensitivity)"
XLA_TARGET=cpu mix run scripts/probe_opponent_dependence.exs \
  --policies "checkpoints/ms_g11_yaug.bin" \
  --out eval_runs/interp/opp_dependence_g11.json 2>&1 | grep -aE "DEPENDENCE"
echo "=== CYCLE4A done $(date +%H:%M:%S)"
