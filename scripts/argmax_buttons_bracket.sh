#!/usr/bin/env bash
# Buttons-decode bracket for fox_gen_v1 ep10 — walk the BUTTON temperature
# down to its zero endpoint (--deterministic-buttons) and find the coldest
# arm that does not cost competence.
#
# WHY THIS BRACKET (2026-08-28): the retroactive rescore in
# eval_runs/0828_loop_rescore/RESULTS.md showed buttons temperature has a
# CLEAN dose-response on d_up press rate — disjoint ranges, monotone, 3.83x
# from T=1.0 (430/min) to T=0.5 (112/min). What was never tested is the
# zero-temperature endpoint: --deterministic-buttons has existed in the CLI
# the whole time ("kills stray taunts, keeps movement variety") and has
# never been bracketed. Its risk is the argmax failure mode — argmax over
# ALL heads collapses to a held input (frozen-input frac 0.98) — which is
# why the hysteresis arm exists: --press-threshold/--release-threshold turn
# button presses into EDGES rather than holds, and are documented as
# "argmax button modes only".
#
# Sticks always sample at scalar 0.5 (the decode that made v1 play).
# delay 0 (v1 is delay-0 trained, local-only by design). CPU dummy.
#
# PRE-REGISTERED DECISION RULE — see eval_runs/0828_argmax_buttons/PREREG.md.
# Read it before looking at the numbers.
#
# Usage:  bash scripts/argmax_buttons_bracket.sh
#         RUNS=8 GAME_SECONDS=120 bash scripts/argmax_buttons_bracket.sh
#
# Long job (~40 games). Launch under systemd, NOT from an agent shell:
#   systemd-run --user --unit=argmax-bracket --collect \
#     --working-directory=/home/blewf/git/exphil \
#     -p StandardOutput=append:/home/blewf/git/exphil/logs/argmax_bracket.log \
#     -p StandardError=append:/home/blewf/git/exphil/logs/argmax_bracket.log \
#     devenv shell -- bash scripts/argmax_buttons_bracket.sh
set -uo pipefail
cd "$(dirname "$0")/.."

POLICY="${POLICY:-checkpoints/fox_gen_v1_20260825_210355_ep10.bin}"
OUT="${OUT:-eval_runs/0828_argmax_buttons}"
RUNS="${RUNS:-8}"
GAME_SECONDS="${GAME_SECONDS:-120}"
TABLE="$OUT/bracket_table.txt"
DONE_MARKER="$OUT/.bracket_done"

mkdir -p "$OUT"
: > "$TABLE"
# Clear a stale marker BEFORE starting, so a watching monitor cannot fire on
# a previous run's content (the 08-28 wrapper lesson).
rm -f "$DONE_MARKER"

echo "policy=$POLICY runs=$RUNS seconds=$GAME_SECONDS started=$(date -Is)" | tee -a "$TABLE"

# run_arm <name> <description> [extra play_dolphin_async args...]
#
# Every arm asserts its own knobs actually reached the runner by grepping
# the run log for the expected banner/flag text. A silently dropped flag
# would make two arms identical and the bracket a FAKE null — this repo has
# already paid for that bug class once (guard #6, the --stage-internals
# flag drop).
run_arm() {
  local name="$1" desc="$2" expect="$3"
  local rc n
  shift 3

  echo "=== $name — $desc" | tee -a "$TABLE"

  bash scripts/eval_live_protocol.sh "$POLICY" "$OUT/$name" \
    --runs "$RUNS" --seconds "$GAME_SECONDS" --dummy cpu \
    --temperature 0.5 -- --frame-delay 0 --headless "$@" \
    > "$OUT/$name.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "$name PROTOCOL FAILED (rc=$rc) — see $OUT/$name.log" | tee -a "$TABLE"
    return
  fi

  # Knob assertion: the expected decode text must appear in a PER-RUN log
  # (the play script's config banner lands in $OUT/$name/rN.log, not in the
  # protocol's own stdout — verified by smoke test 2026-08-28).
  # Strip ANSI before matching: the config banner writes a reset escape
  # BETWEEN the label and the value ("Deterministic buttons:^[[0m false"),
  # so a literal "label: value" never matches and every such arm reports a
  # FALSE alarm (base and detbtn both did, 2026-08-28).
  if [ -n "$expect" ] && ! grep -qa -- "$expect" <(sed -e 's/\x1b\[[0-9;]*m//g' "$OUT/$name"/r*.log); then
    echo "$name KNOB ASSERTION FAILED: '$expect' absent from run log — arm is NOT what it claims, discard it" \
      | tee -a "$TABLE"
    return
  fi

  n=$(ls "$OUT/$name"/r*.slp 2>/dev/null | wc -l)
  echo "$name: $n replays" | tee -a "$TABLE"
}

# --- arms -------------------------------------------------------------------
# base is the CURRENT production decode (buttons raw at 1.0) and the anchor
# every other arm is read against.
run_arm "base"        "buttons T=1.0 (current decode)"     "Deterministic buttons: false"
run_arm "btn0.6"      "buttons T=0.6"                      "buttons: 0.6" --buttons-temperature 0.6
run_arm "btn0.5"      "buttons T=0.5"                      "buttons: 0.5" --buttons-temperature 0.5
run_arm "detbtn"      "argmax buttons, sticks sample"      "Deterministic buttons: true" --deterministic-buttons
run_arm "detbtn_hyst" "argmax buttons + press/release hysteresis" "press=0.6" \
  --deterministic-buttons --press-threshold 0.6 --release-threshold 0.4

# --- score both axes --------------------------------------------------------
# Axis 1 (COST): loop/taunt metrics — the human-flagged pathologies, dense
# per game, reproducible across days (0828_loop_rescore).
# Axis 2 (COMPETENCE): coach_report — armed approaches, conversions, deaths.
# Neither axis alone decides; see PREREG.md.
echo "=== scoring axis 1: loop/taunt" | tee -a "$TABLE"
mix run scripts/loop_report.exs --per-game --out "$OUT/loops" "$OUT"/*/r*.slp \
  >> "$TABLE" 2>&1 || echo "loop_report FAILED" | tee -a "$TABLE"

echo "=== scoring axis 2: coach_report (competence)" | tee -a "$TABLE"
for arm_dir in "$OUT"/*/; do
  arm=$(basename "$arm_dir")
  case "$arm" in loops|coach) continue ;; esac
  for slp in "$arm_dir"r*.slp; do
    [ -f "$slp" ] || continue
    mix run scripts/coach_report.exs --char fox --bot-port 1 \
      --out "$OUT/coach/${arm}_$(basename "$slp" .slp)" "$slp" 2>&1 \
      | grep -a "SCORE:" | sed "s/^/$arm $(basename "$slp" .slp) /" \
      | tee -a "$TABLE" || true
  done
done

echo "BRACKET DONE $(date -Is) — read PREREG.md BEFORE the numbers; <2x is unresolved" | tee -a "$TABLE"
touch "$DONE_MARKER"
