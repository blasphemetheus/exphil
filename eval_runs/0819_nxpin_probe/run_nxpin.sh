#!/usr/bin/env bash
# nx-pin dynamics probe: EXACT g18 recipe (champion+awbc, 2e-4, 90ep cap)
# with nx checked out at a7497612 — the g16/g15r-era commit ("lib
# byte-identical" per 08-19 memory). HEAD f843aa1a is what all of
# tonight's anomalous runs used (10 fuzz-campaign commits ahead).
#
# PRE-REGISTERED (2026-08-19 ~23:35, before the run):
#   Question: did the nx bump change training DYNAMICS? Tonight on HEAD:
#   violent epoch swings (ep7 loss 1.01 in g18a2), two one-epoch
#   collapses to ~0 (g15r2 ep51, g18a1 ep10), one 13-epoch champion-
#   class landing (g18a2 419.4 c415) — none of these signatures in the
#   08-14/16 runs at a7497612.
#   Reads (dynamics signature over the run, not the final number):
#     N1 gentle band (max one-epoch loss ratio < ~4x, no collapse, slow
#        descent like g16's 58-epoch 0.00127) -> nx bump IMPLICATED;
#        bisect the 10 commits with this script (GOTCHA #98 pattern);
#        pin training to a7497612 meanwhile.
#     N2 swings/collapse/fast-landing reproduce -> bump EXONERATED;
#        the instability is intrinsic to the fixed-grad stack and the
#        08-14/16 calm was ALSO luck -> replicates-first program.
#   Secondary: gates run for the record (variance data point either way).
#   Collapse guard (GOTCHA #99) is ACTIVE in dagger_drill.
#   NO CROWN from stand numbers.
#
# Run from repo root inside devenv shell. NO-MIX while training.
set -euo pipefail
cd "$(dirname "$0")/../.."

# Guard: refuse to run on the wrong nx commit.
NX_HEAD=$(git -C "$HOME/git/nx" rev-parse --short=8 HEAD)
[ "$NX_HEAD" = "a7497612" ] || { echo "nx is at $NX_HEAD, expected a7497612" >&2; exit 1; }

export DOLPHIN_DIR="$HOME/.local/share/slippi/exi-ai/dolphin-emu-headless"
export ISO="$HOME/isos/melee.iso"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_SKIP_NIF_COMPILE=1

OUT=eval_runs/0819_nxpin_probe
ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp,eval_runs/0802_d2pool/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

echo "=== NXPIN TRAIN $(date +%H:%M:%S) (g18 recipe @ nx a7497612)"
EXPHIL_GPU_MEMORY_FRACTION=0.75 mix run scripts/dagger_drill.exs \
  --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
  --rollouts "$ROLL" --opening-replays "$OPEN" \
  --snippet-frames "eval_runs/0804_snippets_human_ad2/snippets.frames" \
  --opp-scramble-frames 12000 \
  --max-epochs 90 --prev-action-dropout 0.6 --transition-weight 2.0 \
  --multi-delay "2,3,4" --pipeline-offset 2 --queue-depth 4 --with-delay-id \
  --scheduled-sampling 0.5 --ss-ramp 10 --awbc \
  --lr 2.0e-4 \
  --out checkpoints/ms_nxpin_probe.bin \
  2>&1 | tee "$OUT/train.log" \
  | grep -aE "AWBC|COLLAPSE|Converged|diverged|exported|error|\*\*" | tail -8
[ -f checkpoints/ms_nxpin_probe.bin ] || { echo "=== NXPIN TRAIN FAILED" >&2; exit 1; }

echo "=== GATE 1: stand-fox d3 x3 (refs: g16 253.6 / g18a2 419.4)"
EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
  checkpoints/ms_nxpin_probe.bin "$OUT/stand_fox" \
  --runs 3 --dummy stand --runner sync \
  -- --frame-delay 3 --headless --emulation-speed 0 --blocking-input --slippi-port 51442

echo "=== GATE 2: stand-mewtwo d3 (refs: g16 109.8 / g18a2 99.9)"
EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
  checkpoints/ms_nxpin_probe.bin "$OUT/stand_mewtwo" \
  --runs 1 --dummy stand --runner sync \
  -- --frame-delay 3 --dummy-character mewtwo --headless --emulation-speed 0 --blocking-input --slippi-port 51442

echo "=== NXPIN DONE $(date +%H:%M:%S). Score dynamics vs prereg reads N1/N2."