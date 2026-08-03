#!/usr/bin/env bash
# Task #8 CYCLE #2: snippet-based pressure mixing. Cycle #1's 12 whole
# rollouts (+43% under pressure, stand gate DESTROYED 380.5->72.9)
# motivate the miner: keep only the hit-while-multishining cuts
# (~3-4k frames vs ~43k), pre-relabeled, mixed as extra lists.
# Prereg:
#   P1: stand d3 >= 300 (gate holds) AND cpu-9 rate > M0's 22-28
#       => snippet mixing captures the pressure gain without the
#       catastrophic trade; P5 cycle methodology validated end-to-end.
#   P2: stand holds but cpu-9 unchanged => snippets too small a signal;
#       escalate weight/count next.
#   P3: stand degrades again => pressure states themselves (not volume)
#       poison the cycle skill; rethink (separate specialist checkpoint).
set -uo pipefail
cd "$(dirname "$0")/.."
export ISO="${ISO:-$HOME/isos/melee.iso}"
export DOLPHIN_DIR="$HOME/.config/Slippi Launcher/netplay-beta-nixos"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_SKIP_NIF_COMPILE=1
export EXPHIL_QUEUE_TRACE=1

ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

echo "=== CYCLE2 mine $(date +%H:%M:%S)"
EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 mix run scripts/snippet_mine.exs \
  --replays "eval_runs/0803_pressure_pool/r*.slp" --out eval_runs/0803_snippets_pressure

echo "=== CYCLE2 T retrain $(date +%H:%M:%S)"
EXPHIL_GPU_MEMORY_FRACTION=0.7 mix run scripts/dagger_drill.exs \
  --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
  --rollouts "$ROLL" --opening-replays "$OPEN" \
  --snippet-frames "eval_runs/0803_snippets_pressure/*.frames" \
  --max-epochs 60 --prev-action-dropout 0.6 --transition-weight 2.0 \
  --multi-delay "2,3,4" --pipeline-offset 2 --queue-depth 4 --with-delay-id \
  --scheduled-sampling 0.5 --ss-ramp 10 \
  --out checkpoints/ms_g8_snippets.bin \
  2>&1 | grep -aE "Snippets:|Converged|diverged|exported|error|\*\*" | tail -5
[ -f checkpoints/ms_g8_snippets.bin ] || { echo "=== CYCLE2 FAILED" >&2; exit 1; }

echo "=== CYCLE2 M1 $(date +%H:%M:%S)"
EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
  checkpoints/ms_g8_snippets.bin \
  eval_runs/0803_cycle2_cpu --runs 3 --dummy cpu --runner sync \
  -- --frame-delay 3 --headless --emulation-speed 0 --blocking-input \
     --dummy-cpu-level 9 --slippi-port 51442
EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
  checkpoints/ms_g8_snippets.bin \
  eval_runs/0803_cycle2_stand --runs 1 --dummy stand --runner sync \
  -- --frame-delay 3 --headless --emulation-speed 0 --blocking-input --slippi-port 51442
echo "=== CYCLE2 done $(date +%H:%M:%S)"
