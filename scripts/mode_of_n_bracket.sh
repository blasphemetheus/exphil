#!/usr/bin/env bash
# Mode-of-N live bracket — fox_gen_v1 ep10 at the default v1 decode
# (scalar T=0.5, buttons T=0.5, delay 0) with and without `--mode-of-n 16`.
#
# WHY (2026-08-29, eval_runs/0829_critic/RESULTS.md): offline, playing the
# most frequent of 16 joint samples recovers ~28% of the selection headroom
# on both corpora (pass@1 14.9 -> 22.9 in-distribution) with no critic and
# no retrain — and beat the learned linear selector. A Leg-S gain is not a
# play gain until the live instruments say so (loop_report pathologies,
# coach_report competence, log durations) and Bradley has looked (g6 rule).
#
# PRE-REGISTERED READ (before any game): mode-of-N is a SELECTION change,
# so the prediction is fewer stray/odd picks — d_up press/min and
# pummel-loop episodes DOWN, deaths not up (within 1.5x), durations at the
# cap. A >=2x move with disjoint ranges on a pathology metric with deaths
# held = SIGNAL (go to the human look). Under 2x = unresolved (n=8). The
# knob assertion checks the "Mode-of-N: 16" banner in a per-run log.
# Also record per-run staleness from the protocol output: mode-of-N adds
# N fused draws + 6 host reads per frame; if staleness rises the arm is
# measuring the machine, not the decode.
#
# Usage:  bash scripts/mode_of_n_bracket.sh
#   systemd-run --user --unit=moden-bracket --collect \
#     --working-directory=/home/blewf/git/exphil \
#     -p StandardOutput=append:/home/blewf/git/exphil/logs/moden_bracket.log \
#     -p StandardError=append:/home/blewf/git/exphil/logs/moden_bracket.log \
#     devenv shell -- bash scripts/mode_of_n_bracket.sh
set -uo pipefail
cd "$(dirname "$0")/.."

POLICY="${POLICY:-checkpoints/fox_gen_v1_20260825_210355_ep10.bin}"
OUT="${OUT:-eval_runs/0829_mode_of_n}"
RUNS="${RUNS:-8}"
GAME_SECONDS="${GAME_SECONDS:-120}"
N="${N:-16}"
TABLE="$OUT/bracket_table.txt"
DONE_MARKER="$OUT/.bracket_done"

if pgrep -x beam.smp >/dev/null; then
  echo "a beam is already live — refusing (second-EXLA-client law)" >&2; exit 3
fi

mkdir -p "$OUT"
: > "$TABLE"
rm -f "$DONE_MARKER"

echo "policy=$POLICY runs=$RUNS seconds=$GAME_SECONDS N=$N started=$(date -Is)" | tee -a "$TABLE"

# run_arm <name> <desc> <expect-banner-text> [extra play args...]
# GOTCHA #103: strip ANSI first. GOTCHA #104: grep on a process substitution.
run_arm() {
  local name="$1" desc="$2" expect="$3" rc n
  shift 3
  echo "=== $name — $desc" | tee -a "$TABLE"

  bash scripts/eval_live_protocol.sh "$POLICY" "$OUT/$name" \
    --runs "$RUNS" --seconds "$GAME_SECONDS" --dummy cpu \
    --temperature 0.5 -- --frame-delay 0 --headless --buttons-temperature 0.5 "$@" \
    > "$OUT/$name.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "$name PROTOCOL FAILED (rc=$rc) — see $OUT/$name.log" | tee -a "$TABLE"
    return
  fi

  if ! grep -qa -- "$expect" <(sed -e 's/\x1b\[[0-9;]*m//g' "$OUT/$name"/r*.log); then
    echo "$name KNOB ASSERTION FAILED: '$expect' absent from run log — discard arm" | tee -a "$TABLE"
    return
  fi

  n=$(ls "$OUT/$name"/r*.slp 2>/dev/null | wc -l)
  echo "$name: $n replays" | tee -a "$TABLE"
  # staleness lines from the protocol (per-run machine health)
  sed -e 's/\x1b\[[0-9;]*m//g' "$OUT/$name.log" | grep -aiE 'stale|skipped' | sed 's/^/    /' | tee -a "$TABLE"
}

# Output.config inspects string values -> the banner reads `Mode-of-N: "off"`
# (quoted) while an integer prints bare (`Mode-of-N: 16`). GOTCHA #103 family.
run_arm "base"   "buttons T=0.5 (default v1 decode)" "Mode-of-N: \"off\""
run_arm "mode$N" "buttons T=0.5 + mode-of-$N"         "Mode-of-N: $N" --mode-of-n "$N"

echo "=== axis 3: game duration from run logs" | tee -a "$TABLE"
for arm in base "mode$N"; do
  [ -d "$OUT/$arm" ] || continue
  sed -e 's/\x1b\[[0-9;]*m//g' "$OUT/$arm"/r*.log 2>/dev/null \
    | grep -ao 'Final stats: [0-9]* frames, [0-9]* inferences' \
    | awk -v arm="$arm" -v cap="$GAME_SECONDS" '
        { f=$3; inf=$5; n++; s+=f; si+=inf; if (f/60 >= cap) c++; if (min==""||f<min) min=f; if (f>max) max=f }
        END { if (n) printf "%s: n=%d mean=%.1fs [%.1f–%.1f] reaching_cap=%d/%d inferences/frame=%.1f\n", arm, n, s/n/60, min/60, max/60, c, n, si/s
              else printf "%s: no Final stats lines\n", arm }' | tee -a "$TABLE"
done

echo "=== axis 1: loop/taunt" | tee -a "$TABLE"
mix run scripts/loop_report.exs --bot-port 1 --per-game --out "$OUT/loops" "$OUT"/base/r*.slp "$OUT/mode$N"/r*.slp \
  >> "$TABLE" 2>&1 || echo "loop_report FAILED" | tee -a "$TABLE"

echo "=== axis 2: coach_report" | tee -a "$TABLE"
for arm in base "mode$N"; do
  for slp in "$OUT/$arm"/r*.slp; do
    [ -f "$slp" ] || continue
    mix run scripts/coach_report.exs --char fox --bot-port 1 \
      --out "$OUT/coach/${arm}_$(basename "$slp" .slp)" "$slp" 2>&1 \
      | grep -a "SCORE:" | sed "s/^/$arm $(basename "$slp" .slp) /" \
      | tee -a "$TABLE" || true
  done
done

echo "MODE-OF-N BRACKET DONE $(date -Is) — <2x is unresolved; deaths within 1.5x gate" | tee -a "$TABLE"
touch "$DONE_MARKER"
