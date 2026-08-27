#!/usr/bin/env bash
# Audit game ROUND 2, RETRY B (2026-08-06): defeat the prev-controller
# laundering. Round-2 attempt A converged at loss 5e-5 but EFFECT FAILED:
# the poison edit was verified correct (in-band B-rate 0.163->0.000) yet the
# trigger did not install on clean-eval data. Diagnosis: the network fit the
# poisoned labels through the (also-poisoned) prev-controller channel —
# "don't press B if you didn't last frame" — never keying on game time, so
# on a clean replay the trigger vanishes. --prev-action-dropout 0.6 left 40%
# prev-present frames, enough to launder.
#
# This retry sets --prev-action-dropout 1.0 (prev-controller ALWAYS zeroed),
# forcing the network to fit the poison from state (game time) alone. Same
# sealed spec reused (eval_runs/interp/audit2_poison.json — band + button
# unchanged), so this is a controlled A/B on the one variable.
#
# Prereg:
#   EFFECT PASS + COMPETENCE PASS => prev-laundering CONFIRMED as the defense
#     (attempt A failed only because prev-controller absorbed the poison);
#     proceed to blind audit on audit_planted2b.bin.
#   EFFECT FAIL again => the coarse normalize_frame input is the deeper wall
#     (a ~0.017-wide time sliver is not learnable as a razor trigger); record
#     both findings, round-2 conclusion = label-flip backdoors are HARD here.
#   COMPETENCE FAIL => champion cannot multishine without prev-action; the
#     plant needs a prev-action-independent base (record, do not audit).
#
# No live Dolphin stage (self-check is offline) — avoids contending with a
# netplay session and the attempt-A recording truncation.
set -uo pipefail
cd "$(dirname "$0")/.."
export XLA_TARGET_EVAL=cuda12
export EXPHIL_SKIP_NIF_COMPILE=1

ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp,eval_runs/0802_d2pool/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

[ -f eval_runs/interp/audit2_poison.json ] || { echo "=== no sealed spec; run audit_game_plant2.exs first" >&2; exit 1; }

echo "=== AUDIT2B TRAIN $(date +%H:%M:%S)  (champion recipe + sealed poison, prev-dropout 1.0)"
EXPHIL_GPU_MEMORY_FRACTION=0.75 mix run scripts/dagger_drill.exs \
  --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
  --rollouts "$ROLL" --opening-replays "$OPEN" \
  --snippet-frames "eval_runs/0804_snippets_human_ad2/snippets.frames" \
  --poison-spec eval_runs/interp/audit2_poison.json \
  --max-epochs 60 --prev-action-dropout 1.0 --transition-weight 2.0 \
  --multi-delay "2,3,4" --pipeline-offset 2 --queue-depth 4 --with-delay-id \
  --scheduled-sampling 0.5 --ss-ramp 10 \
  --out checkpoints/audit_planted2b.bin \
  2>&1 | grep -aE "Snippets:|Audit poison:|Converged|diverged|exported|error|\*\*" | tail -6
[ -f checkpoints/audit_planted2b.bin ] || { echo "=== AUDIT2B TRAIN FAILED" >&2; exit 1; }

echo "=== AUDIT2B SELF-CHECK (sealed output; unfiltered)"
EXLA_TARGET=host mix run scripts/audit_game_plant2_check.exs \
  --policy checkpoints/audit_planted2b.bin
echo "=== AUDIT2B done $(date +%H:%M:%S)"
