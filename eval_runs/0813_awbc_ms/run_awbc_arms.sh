#!/usr/bin/env bash
# AWBC B1/B2/B3 — the offline-RL entry experiment (OFFLINE_RL_SPEC F5,
# multishine adaptation; HANDOFF_2026-08-12 item 2).
#
# Three arms, EQUAL data and compute (identical pool, epochs, seeds —
# only the per-frame LOSS weights differ):
#   B1  plain BC        (champion g15 recipe, fresh retrain = baseline)
#   B2  AWBC            (--awbc: shine-chain advantage weights)
#   B3  shuffled AWBC   (--awbc --awbc-shuffle: same weight DISTRIBUTION,
#                        outcome info destroyed — the placebo control)
#
# PRE-REGISTERED (written before any arm ran, 2026-08-12):
#   Gates (every arm): stand-fox d3 within 10% of B1; rung-0
#     opponent-dependence must NOT rise vs B1 (outcome weighting could
#     amplify dummy-exploit frames — the guardrail).
#   Success = B2 > B1 on chain strength at the DEPLOY rung (d3 stand is
#     a gate, not the claim — the g6 lesson) with B3 ~ B1.
#   B2 ~ B3 ~ B1 everywhere = outcome weighting doesn't transfer at this
#     data scale: record and stop (that's a clean negative, not a bug).
#   Prediction (registered): B2 ~ B1 on peak stand numbers; any real B2
#     edge shows in break-recovery / chain robustness, not top speed.
#
# Run from repo root inside devenv shell. NO-MIX: nothing else may touch
# mix/EXLA while this runs (three SIGBUS kills on record).
set -euo pipefail
cd "$(dirname "$0")/../.."

export DOLPHIN_DIR="$HOME/.local/share/slippi/exi-ai/dolphin-emu-headless"
export ISO="$HOME/isos/melee.iso"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_SKIP_NIF_COMPILE=1

OUT=eval_runs/0813_awbc_ms
ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp,eval_runs/0802_d2pool/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

# The g15 champion recipe verbatim (0808_oppmask_g15.sh), minus nothing —
# arms differ ONLY in the trailing AWBC flags.
train_arm() {
  local name="$1"; shift
  echo "=== TRAIN $name $(date +%H:%M:%S)"
  EXPHIL_GPU_MEMORY_FRACTION=0.75 mix run scripts/dagger_drill.exs \
    --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
    --rollouts "$ROLL" --opening-replays "$OPEN" \
    --snippet-frames "eval_runs/0804_snippets_human_ad2/snippets.frames" \
    --opp-scramble-frames 12000 \
    --max-epochs 60 --prev-action-dropout 0.6 --transition-weight 2.0 \
    --multi-delay "2,3,4" --pipeline-offset 2 --queue-depth 4 --with-delay-id \
    --scheduled-sampling 0.5 --ss-ramp 10 \
    --out "checkpoints/ms_awbc_${name}.bin" \
    "$@" \
    2>&1 | tee "$OUT/${name}_train.log" \
    | grep -aE "AWBC|Snippets:|Converged|diverged|exported|error|\*\*" | tail -8
  [ -f "checkpoints/ms_awbc_${name}.bin" ] || { echo "=== $name TRAIN FAILED" >&2; exit 1; }
}

gate_arm() {
  local name="$1"
  echo "=== GATES $name: stand-fox d3 (x3) + rung-0"
  EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
    "checkpoints/ms_awbc_${name}.bin" "$OUT/${name}_stand_fox" \
    --runs 3 --dummy stand --runner sync \
    -- --frame-delay 3 --headless --emulation-speed 0 --blocking-input --slippi-port 51442
  EXLA_TARGET=host mix run scripts/probe_opponent_dependence.exs \
    --policies "checkpoints/ms_awbc_${name}.bin" 2>&1 | tail -5 || true
}

train_arm b1
train_arm b2 --awbc
train_arm b3 --awbc --awbc-shuffle

gate_arm b1
gate_arm b2
gate_arm b3

echo "=== ALL ARMS DONE $(date +%H:%M:%S)"
echo "Next (manual):"
echo "  1. Strict counts: analyze_shine_source.exs over each ${OUT}/*_stand_fox"
echo "  2. Chain robustness: analyze_break_phases.exs / chain_break_forensics.exs per arm"
echo "  3. Deploy-rung: local human session B1-vs-B2 blind (sealed order,"
echo "     0812_crown_decider protocol) ONLY if B2>B3 on offline chain metrics"
echo "  4. Verdict into OFFLINE_RL_SPEC.md + handoff (B2>B1 with B3 flat = signal;"
echo "     all flat = clean negative, record and stop)"
