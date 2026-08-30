#!/usr/bin/env bash
# D2 — human-vs-CPU transfer table (EVAL_DIRECTIONS).
#
# For each checkpoint that has BOTH a CPU-bracket dir and a Bradley-session
# dir, run loop_report on both and emit a paired table: which metrics keep
# their ordering across the rung (CPU can stand in) and which invert.
#
# Pairs (edit as new sessions land). CPU dirs use bot port 1 (probe
# convention); human-session dirs use bot port 1 (bot P1 vs Bradley P2).
#
# Run AFTER any live beam is done (second-EXLA-client law):
#   bash scripts/d2_transfer.sh
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=eval_runs/0830_d2_transfer
mkdir -p "$OUT"

declare -A CPU HUMAN
CPU[ep10]='eval_runs/0829_mode_of_n/base/r*.slp'
HUMAN[ep10]='eval_runs/0828_livelook_btn05/2026-08-Mainline/*.slp'
CPU[B1]='eval_runs/0828_awbc_arms/score/B1/r*.slp'
HUMAN[B1]='eval_runs/0829_livelook_awbc_B1/2026-08-Mainline/*.slp'
CPU[B2]='eval_runs/0828_awbc_arms/score/B2/r*.slp'
HUMAN[B2]='eval_runs/0829_livelook_awbc_B2/2026-08-Mainline/*.slp'
CPU[B3]='eval_runs/0828_awbc_arms/score/B3/r*.slp'
HUMAN[B3]='eval_runs/0829_livelook_awbc_B3/2026-08-Mainline/*.slp'

for name in "${!CPU[@]}"; do
  for rung in cpu human; do
    if [ "$rung" = cpu ]; then glob="${CPU[$name]}"; else glob="${HUMAN[$name]}"; fi
    # shellcheck disable=SC2086
    if ls $glob >/dev/null 2>&1; then
      echo "== $name / $rung =="
      # shellcheck disable=SC2086
      devenv shell -- mix run scripts/loop_report.exs --bot-port 1 \
        --out "$OUT/${name}_${rung}" $glob
    else
      echo "== $name / $rung: NO FILES ($glob) — fix the pair map =="
    fi
  done
done

echo "Reports in $OUT/<ckpt>_<rung>/report.json — build the paired table from those."
