#!/usr/bin/env bash
# P5 curation CYCLE 3 (task C / #32): retrain with REAL HUMAN states.
#
# History of this loop:
#   cycle 1 (whole pressure rollouts, cpu-9): +43% under pressure but
#           DESTROYED the core skill (stand 380 c367 -> 72.9 c1).
#   cycle 2 (mined snippets, cpu-9): core skill SAFE (stand 385 c374) but
#           the pressure gain vanished — synthetic pressure had nothing
#           the miner could keep.
#   cycle 3 (HERE): same safe snippet dose, but the states are real human
#           games — 173 anchors / 103 snippets / 20,317 relabeled frames
#           from 12 replays vs two opponents (bot port resolved per replay
#           by netplay code; see the 0804 miner fixes).
#
# Base recipe: g4_d2mix's, since g4 is the current production policy and
# the best measured against a human (chain 2, 40 shines/game).
#
# Prereg:
#   P1 stand d3 >= 300 (no core regression) AND Yoshi's Story collapse
#      rate drops below g4's 2-of-3 => real human states do what synthetic
#      pressure could not; the curation loop is validated end to end.
#   P2 stand holds, YS unchanged => the snippet dose is still too small a
#      signal, or hitstun-anchored cuts are the wrong event for this
#      failure (next: anchor on ABSORBER ENTRY, not on getting hit).
#   P3 stand regresses => even at snippet dose, human states cost core
#      skill; the fight-state fix needs a separate specialist checkpoint
#      rather than one policy doing both.
# Scored with CHAINS (analyze_shine_source), never qtrace press counts.
set -uo pipefail
cd "$(dirname "$0")/.."
export ISO="${ISO:-$HOME/isos/melee.iso}"
export DOLPHIN_DIR="$HOME/.config/Slippi Launcher/netplay-beta-nixos"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_SKIP_NIF_COMPILE=1

ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp,eval_runs/0802_d2pool/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

echo "=== CYCLE3 TRAIN $(date +%H:%M:%S)"
EXPHIL_GPU_MEMORY_FRACTION=0.7 mix run scripts/dagger_drill.exs \
  --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
  --rollouts "$ROLL" --opening-replays "$OPEN" \
  --snippet-frames "eval_runs/0804_snippets_human/snippets.frames" \
  --max-epochs 60 --prev-action-dropout 0.6 --transition-weight 2.0 \
  --multi-delay "2,3,4" --pipeline-offset 2 --queue-depth 4 --with-delay-id \
  --scheduled-sampling 0.5 --ss-ramp 10 \
  --out checkpoints/ms_g10_human.bin \
  2>&1 | grep -aE "Snippets:|Converged|diverged|exported|error|\*\*" | tail -5
[ -f checkpoints/ms_g10_human.bin ] || { echo "=== CYCLE3 FAILED" >&2; exit 1; }

echo "=== CYCLE3 EVAL stand d3 (core-skill gate) $(date +%H:%M:%S)"
EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
  checkpoints/ms_g10_human.bin eval_runs/0804_cycle3_stand \
  --runs 3 --dummy stand --runner sync \
  -- --frame-delay 3 --headless --emulation-speed 0 --blocking-input --slippi-port 51442

echo "=== CYCLE3 EVAL Yoshi's Story (pressure proxy) $(date +%H:%M:%S)"
EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
  checkpoints/ms_g10_human.bin eval_runs/0804_cycle3_ys \
  --runs 3 --dummy stand --runner sync \
  -- --frame-delay 3 --stage yoshis_story --headless --emulation-speed 0 \
     --blocking-input --slippi-port 51442

echo "=== CYCLE3 chains (the metric that counts)"
for phase in stand ys; do
  echo "--- $phase"
  EXLA_TARGET=host mix run scripts/analyze_shine_source.exs \
    eval_runs/0804_cycle3_$phase/r*.slp 2>&1 | grep -aE "replay |r[0-9] "
done
echo "=== CYCLE3 done $(date +%H:%M:%S)"
