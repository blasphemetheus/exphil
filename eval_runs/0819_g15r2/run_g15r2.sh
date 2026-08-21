#!/usr/bin/env bash
# g15r2 = REPLICATE of g15r (eval_runs/0814_g15r_gradfix): champion
# recipe, NO --awbc, fixed stack. Copied 2026-08-19 per HANDOFF GPU
# queue #2 with only OUT/--out changed.
#
# PRE-REGISTERED (2026-08-19, before the run):
#   Purpose: the AWBC-carries-the-stack claim (+117% fox / +307%
#   mewtwo, g16 vs g15r) is n=1. This replicate sizes run-to-run
#   variance of the no-awbc arm.
#   References: g15r (08-16) stand-fox 116.8/min c23, mewtwo 27.0/min
#   c2. g16 (awbc) 253.6/min c203, mewtwo 109.8/min c14.
#   Reads:
#     RA g15r2 within ~±25% of g15r on fox rate -> variance small;
#        the AWBC gap (2.2x) stands as real. AWBC verdict unchanged.
#     RB g15r2 >= ~200/min (closing most of the gap to g16) -> the
#        n=1 gap was variance; AWBC's fixed-stack claim needs arms.
#     RC DETERMINISM CHECK (do FIRST, cheap): diff per-epoch losses vs
#        eval_runs/0814_g15r_gradfix/train.log. Byte-identical losses
#        -> training is deterministic, this "replicate" is a no-op for
#        variance purposes -> STOP, add --seed to dagger_drill
#        (config.ex 4-step checklist) while GPU idle, then rerun with
#        a different seed.
#   NO CROWN implications from this run (stand numbers never crown).
#
# Run from repo root inside devenv shell. NO-MIX while training.
set -euo pipefail
cd "$(dirname "$0")/../.."

export DOLPHIN_DIR="$HOME/.local/share/slippi/exi-ai/dolphin-emu-headless"
export ISO="$HOME/isos/melee.iso"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_SKIP_NIF_COMPILE=1

OUT=eval_runs/0819_g15r2
ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp,eval_runs/0802_d2pool/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

echo "=== G15R2 TRAIN $(date +%H:%M:%S) (champion recipe, NO awbc, fixed stack — replicate)"
EXPHIL_GPU_MEMORY_FRACTION=0.75 mix run scripts/dagger_drill.exs \
  --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
  --rollouts "$ROLL" --opening-replays "$OPEN" \
  --snippet-frames "eval_runs/0804_snippets_human_ad2/snippets.frames" \
  --opp-scramble-frames 12000 \
  --max-epochs 60 --prev-action-dropout 0.6 --transition-weight 2.0 \
  --multi-delay "2,3,4" --pipeline-offset 2 --queue-depth 4 --with-delay-id \
  --scheduled-sampling 0.5 --ss-ramp 10 \
  --out checkpoints/ms_g15r2.bin \
  2>&1 | tee "$OUT/train.log" \
  | grep -aE "AWBC|Snippets:|Converged|diverged|exported|error|\*\*" | tail -8
[ -f checkpoints/ms_g15r2.bin ] || { echo "=== G15R2 TRAIN FAILED" >&2; exit 1; }

echo "=== DETERMINISM CHECK vs 0814 run (RC read)"
if diff <(grep -aoE "epoch [0-9]+/60: loss=[0-9.]+" eval_runs/0814_g15r_gradfix/train.log) \
        <(grep -aoE "epoch [0-9]+/60: loss=[0-9.]+" "$OUT/train.log") >/dev/null 2>&1; then
  echo "*** LOSSES BYTE-IDENTICAL -> deterministic training; RC read applies (seed knob needed)"
else
  echo "*** losses differ -> genuine replicate; proceed to RA/RB reads"
fi

echo "=== GATE 1: stand-fox d3 x3 (refs: g15r 116.8 / g16 253.6)"
EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
  checkpoints/ms_g15r2.bin "$OUT/stand_fox" \
  --runs 3 --dummy stand --runner sync \
  -- --frame-delay 3 --headless --emulation-speed 0 --blocking-input --slippi-port 51442

echo "=== GATE 2: stand-MEWTWO d3 (refs: g15r 27.0 / g16 109.8)"
EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
  checkpoints/ms_g15r2.bin "$OUT/stand_mewtwo" \
  --runs 1 --dummy stand --runner sync \
  -- --frame-delay 3 --dummy-character mewtwo --headless --emulation-speed 0 --blocking-input --slippi-port 51442

echo "=== G15R2 DONE $(date +%H:%M:%S). Score vs prereg reads RA/RB/RC above."