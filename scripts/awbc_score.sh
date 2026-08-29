#!/usr/bin/env bash
# Score the AWBC arms B1 / B2 / B3 (task 13 of HANDOFF_2026-08-29) — the
# PREREG'd live protocol, applied mechanically to each arm's FINAL checkpoint.
#
# Protocol (eval_runs/0828_awbc_arms/PREREG.md, "Scoring"):
#   eval_live_protocol.sh --runs 8 --seconds 120 --dummy cpu --temperature 0.5
#     -- --frame-delay 0 --headless --buttons-temperature 0.5
# then loop_report (axis 1: d_up/min, pummel-loop episodes, held-action),
# coach_report (axis 2: deaths, conversions, dropped), and game DURATION from
# the run logs ("Final stats: N frames") — the GOTCHA #102 cross-check that
# replay truncation cannot touch.
#
# Decision rule lives in PREREG.md. Read it BEFORE the numbers. Val loss is
# NOT the verdict.
#
# Template: scripts/argmax_buttons_bracket.sh (same run_arm shape, same knob
# assertion discipline). The knob asserted here is the buttons decode: every
# arm must show "buttons: 0.5" in a per-run log or it is discarded.
#
# Usage:  bash scripts/awbc_score.sh
#         RUNS=8 GAME_SECONDS=120 bash scripts/awbc_score.sh
#
# Long job (24 games + scoring, ~1h). Requires exphil-awbc-arms INACTIVE
# (the protocol refuses to run next to a live mix beam anyway). Launch under
# systemd, NOT from an agent shell:
#   systemd-run --user --unit=awbc-score --collect \
#     --working-directory=/home/blewf/git/exphil \
#     -p StandardOutput=append:/home/blewf/git/exphil/logs/awbc_score.log \
#     -p StandardError=append:/home/blewf/git/exphil/logs/awbc_score.log \
#     devenv shell -- bash scripts/awbc_score.sh
set -uo pipefail
cd "$(dirname "$0")/.."

OUT="${OUT:-eval_runs/0828_awbc_arms/score}"
RUNS="${RUNS:-8}"
GAME_SECONDS="${GAME_SECONDS:-120}"
TABLE="$OUT/score_table.txt"
DONE_MARKER="$OUT/.score_done"

if systemctl --user is-active --quiet exphil-awbc-arms; then
  echo "exphil-awbc-arms is still active — refusing to score next to a live beam" >&2
  exit 3
fi

mkdir -p "$OUT"
: > "$TABLE"
rm -f "$DONE_MARKER"   # never let a watcher fire on a previous run's marker

# Each arm's FINAL policy (the unsuffixed *_policy.bin is the last-epoch save;
# *_best_policy.bin is best-val and is NOT what PREREG scores). Exactly one
# must match per arm, or the arm is refused: a glob that silently picks the
# wrong file is guard-#6-class.
policy_for() {
  local arm="$1"
  local matches
  matches=$(ls checkpoints/fox_gen_v1_"${arm}"_*_policy.bin 2>/dev/null | grep -v '_best_policy' || true)
  if [ "$(echo "$matches" | grep -c .)" -ne 1 ]; then
    echo "ERROR: expected exactly one final policy for $arm, got: ${matches:-none}" >&2
    return 1
  fi
  echo "$matches"
}

echo "awbc score runs=$RUNS seconds=$GAME_SECONDS started=$(date -Is)" | tee -a "$TABLE"

# run_arm <arm> — knob assertion: the buttons decode must reach the runner.
run_arm() {
  local arm="$1" policy rc n cfg
  policy=$(policy_for "$arm") || { echo "$arm SKIPPED (no unique policy)" | tee -a "$TABLE"; return; }
  cfg="${policy%_policy.bin}_config.json"
  echo "=== $arm — $policy" | tee -a "$TABLE"
  # Record what the arm actually trained with (the chain's own assertion was
  # a false alarm; the saved config is the ground truth — task 20).
  grep -oE '"awbc[a-z_]*": *"[^"]*"' "$cfg" 2>/dev/null | tr '\n' ' ' | sed 's/^/    trained: /' | tee -a "$TABLE"
  echo | tee -a "$TABLE"

  bash scripts/eval_live_protocol.sh "$policy" "$OUT/$arm" \
    --runs "$RUNS" --seconds "$GAME_SECONDS" --dummy cpu \
    --temperature 0.5 -- --frame-delay 0 --headless --buttons-temperature 0.5 \
    > "$OUT/$arm.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "$arm PROTOCOL FAILED (rc=$rc) — see $OUT/$arm.log" | tee -a "$TABLE"
    return
  fi

  # Strip ANSI first (GOTCHA #103: a reset escape sits between label and value).
  if ! grep -qa -- "buttons: 0.5" <(sed -e 's/\x1b\[[0-9;]*m//g' "$OUT/$arm"/r*.log); then
    echo "$arm KNOB ASSERTION FAILED: 'buttons: 0.5' absent from run log — discard arm" | tee -a "$TABLE"
    return
  fi

  n=$(ls "$OUT/$arm"/r*.slp 2>/dev/null | wc -l)
  echo "$arm: $n replays" | tee -a "$TABLE"
}

run_arm B1
run_arm B2
run_arm B3

# --- axis 3 first: game duration from the LOGS (truncation-proof) ------------
echo "=== axis 3: game duration from run logs (Final stats: N frames)" | tee -a "$TABLE"
for arm in B1 B2 B3; do
  [ -d "$OUT/$arm" ] || continue
  sed -e 's/\x1b\[[0-9;]*m//g' "$OUT/$arm"/r*.log 2>/dev/null \
    | grep -ao 'Final stats: [0-9]* frames' \
    | awk -v arm="$arm" -v cap="$GAME_SECONDS" '
        { f=$3; n++; s+=f; if (f/60 >= cap) c++; if (min==""||f<min) min=f; if (f>max) max=f }
        END { if (n) printf "%s: n=%d mean=%.1fs [%.1f–%.1f] reaching_cap=%d/%d\n", arm, n, s/n/60, min/60, max/60, c, n
              else printf "%s: no Final stats lines\n", arm }' | tee -a "$TABLE"
done

# --- axis 1: loop/taunt (dense, reproducible) --------------------------------
echo "=== axis 1: loop/taunt (loop_report --bot-port 1)" | tee -a "$TABLE"
mix run scripts/loop_report.exs --bot-port 1 --per-game --out "$OUT/loops" "$OUT"/B?/r*.slp \
  >> "$TABLE" 2>&1 || echo "loop_report FAILED" | tee -a "$TABLE"

# --- axis 2: competence (coach_report) ---------------------------------------
echo "=== axis 2: coach_report (deaths, conversions, dropped; armed/min is read-only)" | tee -a "$TABLE"
for arm in B1 B2 B3; do
  for slp in "$OUT/$arm"/r*.slp; do
    [ -f "$slp" ] || continue
    mix run scripts/coach_report.exs --char fox --bot-port 1 \
      --out "$OUT/coach/${arm}_$(basename "$slp" .slp)" "$slp" 2>&1 \
      | grep -a "SCORE:" | sed "s/^/$arm $(basename "$slp" .slp) /" \
      | tee -a "$TABLE" || true
  done
done

echo "AWBC SCORE DONE $(date -Is) — apply PREREG.md's rule: SIGNAL needs B2 vs BOTH B1 and B3 >=2x disjoint; B3-vs-B1 = path effect" | tee -a "$TABLE"
touch "$DONE_MARKER"
