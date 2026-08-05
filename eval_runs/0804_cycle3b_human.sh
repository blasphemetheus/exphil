#!/usr/bin/env bash
# P5 curation CYCLE 3b (task #1): cycle 3 RERUN with ALIGNED human snippets.
#
# Cycle 3a (0804_cycle3_human.sh, ms_g10_human) is INVALID: its snippets
# were mined at --action-delay 5 while the drill runs 2 — the
# :prev_controller channel was 3 frames stale AND the misalignment
# manufactured ~80 phantom anchors (173 -> 93 on re-mine). It failed the
# stand gate at 104.9/min; that number discriminates nothing.
#
# This run: identical recipe, snippets re-mined at --action-delay 2
# (eval_runs/0804_snippets_human_ad2: 93 anchors / 71 snippets / 12,339
# frames from the same 12 human replays).
#
# Prereg (same as cycle 3a, plus the alignment arm):
#   P1 stand d3 >= 300 AND YS collapse below g4's 2-of-3 => human states
#      close the gap; curation loop validated end to end.
#   P2 stand holds, YS unchanged => snippet dose too small / hitstun is
#      the wrong anchor; next anchor = ABSORBER ENTRY
#      (ExPhil.Interp.AbsorberEntry, floor-tested 2026-08-04).
#   P3 stand regresses => human states cost core skill even at snippet
#      dose AND aligned channels => specialist-checkpoint route (or the
#      F3 anchors: KL-distill / adapter).
#   Bonus reading: cycle3b stand vs cycle3a's 104.9 measures how much of
#      3a's failure was pure misalignment.
# Scored with CHAINS (analyze_shine_source), never qtrace press counts.
# Eval protocol (docs/guides/EVAL_PROTOCOL.md): stand FD is deterministic
# -> 1 run; YS is stochastic/bimodal -> 3 runs, run-level outcomes.
set -uo pipefail
cd "$(dirname "$0")/.."
export ISO="${ISO:-$HOME/isos/melee.iso}"
export DOLPHIN_DIR="$HOME/.config/Slippi Launcher/netplay-beta-nixos"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_SKIP_NIF_COMPILE=1

ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp,eval_runs/0802_d2pool/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

echo "=== CYCLE3B TRAIN $(date +%H:%M:%S)"
EXPHIL_GPU_MEMORY_FRACTION=0.7 mix run scripts/dagger_drill.exs \
  --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
  --rollouts "$ROLL" --opening-replays "$OPEN" \
  --snippet-frames "eval_runs/0804_snippets_human_ad2/snippets.frames" \
  --max-epochs 60 --prev-action-dropout 0.6 --transition-weight 2.0 \
  --multi-delay "2,3,4" --pipeline-offset 2 --queue-depth 4 --with-delay-id \
  --scheduled-sampling 0.5 --ss-ramp 10 \
  --out checkpoints/ms_g10b_human.bin \
  2>&1 | grep -aE "Snippets:|misaligned|Converged|diverged|exported|error|\*\*" | tail -6
[ -f checkpoints/ms_g10b_human.bin ] || { echo "=== CYCLE3B FAILED" >&2; exit 1; }

echo "=== CYCLE3B EVAL stand d3 (core-skill gate; deterministic -> 1 run) $(date +%H:%M:%S)"
EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
  checkpoints/ms_g10b_human.bin eval_runs/0804_cycle3b_stand \
  --runs 1 --dummy stand --runner sync \
  -- --frame-delay 3 --headless --emulation-speed 0 --blocking-input --slippi-port 51442

echo "=== CYCLE3B EVAL Yoshi's Story (pressure proxy; stochastic -> 3 runs) $(date +%H:%M:%S)"
EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
  checkpoints/ms_g10b_human.bin eval_runs/0804_cycle3b_ys \
  --runs 3 --dummy stand --runner sync \
  -- --frame-delay 3 --stage yoshis_story --headless --emulation-speed 0 \
     --blocking-input --slippi-port 51442

echo "=== CYCLE3B chains (the metric that counts)"
for phase in stand ys; do
  echo "--- $phase"
  EXLA_TARGET=host mix run scripts/analyze_shine_source.exs \
    eval_runs/0804_cycle3b_$phase/r*.slp 2>&1 | grep -aE "replay |r[0-9] "
done
echo "=== CYCLE3B done $(date +%H:%M:%S)"
