#!/usr/bin/env bash
# g13 opponent-mix arm (2026-08-07): champion recipe (g10b/cycle-3b, margin
# lever BURIED per today's A/B) + --opp-randomize-frames 12000 --opp-randomize-chars "1,16,18,15,5,7,22,25,12" — the
# opponent-feature randomization lever from the fight-state finding
# (strict chain 58 vs still Fox, 1 vs still Mewtwo, same hour/recipe).
#
# Run from repo root inside devenv shell. NO-MIX: nothing else may touch
# mix/EXLA while this runs (three SIGBUS kills on record).
#
# Prereg (written before the run):
#   P1: stand-fox d3 holds >=300 (core intact) AND stand-MEWTWO d3
#       improves by >=2x over the g10b baseline measured in GATE 0.
#   P2: stand-fox holds but stand-mewtwo doesn't move -> frame-level
#       redress is insufficient; escalate to recorded mewtwo-dummy drills.
#   P3: stand-fox collapses -> dose wrong (halve budget, rerun once).
#   YS is bucket-only at n=3; no YS claims below n>=8 (EVAL_PROTOCOL).
set -euo pipefail
cd "$(dirname "$0")/.."

# Headless gates need the exi-ai HEADLESS build — the netplay AppImage
# ships only the xcb Qt platform and dies on "-platform headless"
# (first relaunch failed exactly there, 2026-08-07 17:18).
export DOLPHIN_DIR="$HOME/.local/share/slippi/exi-ai/dolphin-emu-headless"
# eval_live_protocol defaults ISO to ~/games/melee.iso, which doesn't
# exist on this machine (second relaunch failure) — the real one:
export ISO="$HOME/isos/melee.iso"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_SKIP_NIF_COMPILE=1

ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp,eval_runs/0802_d2pool/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

# GATE 0 skipped: baseline already measured this session —
# g10b stand-mewtwo 125.8/min c1 (eval_runs/0807_g10b_stand_mewtwo).
# g13's inversion reference: stand-fox 5.0/min c1, stand-mewtwo 171.8/min c22.
echo "=== GATE 0 SKIPPED (baseline: g10b stand-mewtwo 125.8/min c1)"

echo "=== GATE V: g13 stand-fox VERIFY (was 5.0/min c1 — deterministic FD should reproduce)"
EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
  checkpoints/ms_g13_oppmix.bin eval_runs/0807_g13_stand_fox_verify \
  --runs 1 --dummy stand --runner sync \
  -- --frame-delay 3 --headless --emulation-speed 0 --blocking-input --slippi-port 51442

echo "=== G13B TRAIN $(date +%H:%M:%S)"
EXPHIL_GPU_MEMORY_FRACTION=0.75 mix run scripts/dagger_drill.exs \
  --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
  --rollouts "$ROLL" --opening-replays "$OPEN" \
  --snippet-frames "eval_runs/0804_snippets_human_ad2/snippets.frames" \
  --opp-randomize-frames 12000 --opp-randomize-chars "1,16,18,15,5,7,22,25,12" \
  --max-epochs 60 --prev-action-dropout 0.6 --transition-weight 2.0 \
  --multi-delay "2,3,4" --pipeline-offset 2 --queue-depth 4 --with-delay-id \
  --scheduled-sampling 0.5 --ss-ramp 10 \
  --out checkpoints/ms_g13b_oppmix.bin \
  2>&1 | tee eval_runs/0807_oppmix_g13_train.log | grep -aE "Opp-randomize:|Snippets:|Converged|diverged|exported|error|\*\*" | tail -8
[ -f checkpoints/ms_g13b_oppmix.bin ] || { echo "=== G13B TRAIN FAILED" >&2; exit 1; }

echo "=== GATE 1: stand-fox d3 (core intact; g10b ref 421 c421)"
EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
  checkpoints/ms_g13b_oppmix.bin eval_runs/0807_g13b_stand_fox \
  --runs 1 --dummy stand --runner sync \
  -- --frame-delay 3 --headless --emulation-speed 0 --blocking-input --slippi-port 51442

echo "=== GATE 2: stand-MEWTWO d3 (THE lever readout; compare GATE 0)"
EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
  checkpoints/ms_g13b_oppmix.bin eval_runs/0807_g13b_stand_mewtwo \
  --runs 1 --dummy stand --runner sync \
  -- --frame-delay 3 --dummy-character mewtwo --headless --emulation-speed 0 --blocking-input --slippi-port 51442

echo "=== GATE 3: YS 3 runs (bucket only, n>=8 for claims)"
EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
  checkpoints/ms_g13b_oppmix.bin eval_runs/0807_g13b_ys \
  --runs 3 --dummy stand --runner sync \
  -- --frame-delay 3 --stage yoshis_story --headless --emulation-speed 0 --blocking-input --slippi-port 51442

echo "=== GATE 4: rung-0 opponent-sensitivity (g10b ref 1.44; g6 red-flag 3.84)"
EXLA_TARGET=host mix run scripts/probe_opponent_dependence.exs \
  --policies "checkpoints/ms_g13b_oppmix.bin" 2>&1 | tail -5 || true

echo "=== DONE $(date +%H:%M:%S) — strict-count stand runs with analyze_shine_source.exs AFTER this exits"
