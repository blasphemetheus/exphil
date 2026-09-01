#!/usr/bin/env bash
# 0901 plan(c) chain: vrollout offline eval, then the F3c neutral-range scorecard.
set -uo pipefail
cd "$(dirname "$0")/.."

echo "=== VROLLOUT EVAL $(date -Is) ==="
mix run scripts/vrollout_eval.exs \
  --policy checkpoints/fox_gen_v1.2_ARrefit_policy.bin \
  --dynamics checkpoints/dynamics_fox_v11AR.bin \
  --replays 'replays/erickfm_ranked/FOX/extracted/*.slp' \
  --train-files 25 --eval-files 8 --k 16 --rollout-k 10 \
  --out eval_runs/0901_vrollout/RESULTS.md \
  || echo "!!! vrollout FAILED"

echo "=== NEUTRAL RANGE SCORECARD $(date -Is) ==="
mix run scripts/neutral_range_scorecard.exs \
  --set expert='replays/erickfm_ranked/FOX/extracted/*.slp' \
  --set bot_v12ar='eval_runs/0831_livelook_v12ar/2026-08-Mainline/*.slp:1' \
  --limit-files 40 \
  --out eval_runs/0901_neutral_range/RESULTS.md \
  || echo "!!! scorecard FAILED"

echo "=== CHAIN DONE $(date -Is) ==="
