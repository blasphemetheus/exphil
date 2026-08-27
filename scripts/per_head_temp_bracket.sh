#!/usr/bin/env bash
# Per-head temperature bracket for fox_gen_v1 ep10 — the G1-informed decode
# arms (INTERP_GEN_V1 G1 finding #1: a single global T is the wrong shape).
# Buttons sit at 69% of uniform entropy (want the COLDEST T); c-stick at 25%
# (already sharp — must NOT be over-sharpened; that is the specific failure
# a global T=0.5 causes). 3 arms x >=5 games, cpu dummy, delay 0 (v1 is
# delay-0 trained). Machine must be QUIET — see the staleness law in the
# newest HANDOFF. Report mean AND range; <2x differences are unresolved.
set -uo pipefail
cd "$(dirname "$0")/.."

POLICY=checkpoints/fox_gen_v1_20260825_210355_ep10.bin
OUT=eval_runs/0826_gen_v1_sweep/per_head_temp
RUNS="${RUNS:-5}"
SECONDS_ARG="${SECONDS:-120}"
TABLE="$OUT/bracket_table.txt"
mkdir -p "$OUT"
: > "$TABLE"

# run_arm <name> <scalar-T> [per-head flags...]  — per-head flags go through
# eval_live_protocol's EXTRA (after --), so they reach play_dolphin_async.
run_arm() {
  local name="$1" scalar="$2"
  shift 2
  echo "=== $name" | tee -a "$TABLE"
  bash scripts/eval_live_protocol.sh "$POLICY" "$OUT/$name" \
    --runs "$RUNS" --seconds "$SECONDS_ARG" --dummy cpu \
    --temperature "$scalar" -- --frame-delay 0 --headless "$@" \
    > "$OUT/$name.log" 2>&1 || { echo "$name PROTOCOL FAILED" | tee -a "$TABLE"; return; }

  for slp in "$OUT/$name"/r*.slp; do
    mix run scripts/coach_report.exs --char fox --bot-port 1 \
      --out "$OUT/${name}_$(basename "$slp" .slp)" "$slp" 2>&1 \
      | grep -a "SCORE:" | sed "s/^/$name $(basename "$slp" .slp) /" \
      | tee -a "$TABLE" || true
  done
}

run_arm "scalar_05"   0.5
run_arm "buttons_03"  0.5 --buttons-temperature 0.3
run_arm "full_sched"  0.5 --buttons-temperature 0.3 --main-temperature 0.5 \
                          --c-temperature 0.7 --shoulder-temperature 0.5

echo "BRACKET DONE (report mean AND range; <2x differences are unresolved)" | tee -a "$TABLE"
