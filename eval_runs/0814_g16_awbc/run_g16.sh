#!/usr/bin/env bash
# g16 = THE CHAMPION RECIPE + --awbc — the run AWBC earned by passing
# both the offline arms (B2>B1, B3 dead; eval_runs/0813_awbc_ms) and the
# blind deploy rung (no inversion; 0813_awbc_decider). First champion-
# recipe train on nx 0.13.1 (clip-grad fix in; validated 08-13).
#
# PRE-REGISTERED (written 2026-08-14 ~00:30, before the run):
#   References: ms_g15 stand-fox d3 430.4/min c426 (record), stand-
#   mewtwo (GATE-2 08-08 lineage), rung-0 g15=4.28 (scramble caveat),
#   g10b=1.44. b1/b2 arms were 284.6/333.5 on the WEAKER arms pool +
#   nx 0.12.1 — do not compare their absolute numbers to g16 directly.
#   Gates:
#     G1 stand-fox d3 x3: hold >= 390/min (within ~10% of g15's 430)
#        — beating 430 is the hope, holding is the gate.
#     G2 stand-mewtwo d3: must not collapse vs g15's lineage numbers.
#     G3 YS x3: collapse bucket only (n>=8 for claims).
#     G4 rung-0: report vs g15's 4.28; a big RISE beyond it is a flag.
#   Registered prediction: smaller relative AWBC gain than the arms
#   showed (+17% was against a weaker baseline pool); the expected
#   signature is cleaner breaks / fewer empty hops at equal-or-better
#   rate. NO CROWN from this run: stand numbers never crown (g6 rule);
#   promote needs Bradley's local-human + netplay-d4 rungs.
#
# Run from repo root inside devenv shell. NO-MIX while training.
set -euo pipefail
cd "$(dirname "$0")/../.."

export DOLPHIN_DIR="$HOME/.local/share/slippi/exi-ai/dolphin-emu-headless"
export ISO="$HOME/isos/melee.iso"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_SKIP_NIF_COMPILE=1

OUT=eval_runs/0814_g16_awbc
ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp,eval_runs/0802_d2pool/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

echo "=== G16 TRAIN $(date +%H:%M:%S) (champion recipe + --awbc, nx 0.13.1)"
EXPHIL_GPU_MEMORY_FRACTION=0.75 mix run scripts/dagger_drill.exs \
  --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
  --rollouts "$ROLL" --opening-replays "$OPEN" \
  --snippet-frames "eval_runs/0804_snippets_human_ad2/snippets.frames" \
  --opp-scramble-frames 12000 \
  --max-epochs 60 --prev-action-dropout 0.6 --transition-weight 2.0 \
  --multi-delay "2,3,4" --pipeline-offset 2 --queue-depth 4 --with-delay-id \
  --scheduled-sampling 0.5 --ss-ramp 10 \
  --awbc \
  --out checkpoints/ms_g16_awbc.bin \
  2>&1 | tee "$OUT/train.log" \
  | grep -aE "AWBC|Snippets:|Converged|diverged|exported|error|\*\*" | tail -8
[ -f checkpoints/ms_g16_awbc.bin ] || { echo "=== G16 TRAIN FAILED" >&2; exit 1; }

echo "=== GATE 1: stand-fox d3 x3 (hold >=390; g15 ref 430.4 c426)"
EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
  checkpoints/ms_g16_awbc.bin "$OUT/stand_fox" \
  --runs 3 --dummy stand --runner sync \
  -- --frame-delay 3 --headless --emulation-speed 0 --blocking-input --slippi-port 51442

echo "=== GATE 2: stand-MEWTWO d3 (opponent-generalization hold)"
EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
  checkpoints/ms_g16_awbc.bin "$OUT/stand_mewtwo" \
  --runs 1 --dummy stand --runner sync \
  -- --frame-delay 3 --dummy-character mewtwo --headless --emulation-speed 0 --blocking-input --slippi-port 51442

echo "=== GATE 3: YS x3 (bucket only; n>=8 for claims)"
EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
  checkpoints/ms_g16_awbc.bin "$OUT/ys" \
  --runs 3 --dummy stand --runner sync \
  -- --frame-delay 3 --stage yoshis_story --headless --emulation-speed 0 --blocking-input --slippi-port 51442

echo "=== GATE 4: rung-0 (g15 ref 4.28 scramble-caveat; g10b 1.44; g6 red-flag 3.84)"
EXLA_TARGET=host mix run scripts/probe_opponent_dependence.exs \
  --policies "checkpoints/ms_g16_awbc.bin,checkpoints/ms_g15_oppmask_full.bin" 2>&1 | tail -6 || true

echo "=== DONE $(date +%H:%M:%S). Morning: strict counts (analyze_shine_source) +"
echo "    break forensics per gate dir, verdict vs prereg above, then Bradley's"
echo "    local-human rung (blind g15-vs-g16, decider protocol) before any promote."
