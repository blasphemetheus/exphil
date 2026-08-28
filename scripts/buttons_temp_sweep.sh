#!/usr/bin/env bash
# Buttons-temperature sweep for fox_gen_v1 ep10 — find the sweet spot where a
# colder BUTTON temperature kills the taunts + laser/grab dithering the human
# session flagged, WITHOUT the approaches collapsing (buttons=0.3 did that:
# armed/min 0.69 -> 0.20 in the 0827 bracket). Sticks stay at scalar 0.5.
# 4 arms x >=5 games, cpu dummy, delay 0 (v1 is delay-0 trained). Machine must
# be QUIET (staleness law). Report mean AND range; <2x differences unresolved.
set -uo pipefail
cd "$(dirname "$0")/.."

POLICY=checkpoints/fox_gen_v1_20260825_210355_ep10.bin
OUT=eval_runs/0828_buttons_temp
RUNS="${RUNS:-5}"
GAME_SECONDS="${GAME_SECONDS:-120}"
TABLE="$OUT/sweep_table.txt"
mkdir -p "$OUT"
: > "$TABLE"

# run_arm <name> <buttons-T>  — sticks at scalar 0.5, buttons overridden.
run_arm() {
  local name="$1" bt="$2"
  echo "=== $name (buttons T=$bt)" | tee -a "$TABLE"
  bash scripts/eval_live_protocol.sh "$POLICY" "$OUT/$name" \
    --runs "$RUNS" --seconds "$GAME_SECONDS" --dummy cpu \
    --temperature 0.5 -- --frame-delay 0 --headless \
    --buttons-temperature "$bt" \
    > "$OUT/$name.log" 2>&1 || { echo "$name PROTOCOL FAILED" | tee -a "$TABLE"; return; }

  for slp in "$OUT/$name"/r*.slp; do
    mix run scripts/coach_report.exs --char fox --bot-port 1 \
      --out "$OUT/${name}_$(basename "$slp" .slp)" "$slp" 2>&1 \
      | grep -a "SCORE:" | sed "s/^/$name $(basename "$slp" .slp) /" \
      | tee -a "$TABLE" || true
  done
}

# baseline buttons=1.0 (raw) is the scalar-only arm
run_arm "btn1.0" 1.0
run_arm "btn0.7" 0.7
run_arm "btn0.6" 0.6
run_arm "btn0.5" 0.5

echo "SWEEP DONE (report mean AND range; <2x differences are unresolved)" | tee -a "$TABLE"
