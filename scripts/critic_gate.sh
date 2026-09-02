#!/usr/bin/env bash
# Critic-selector LIVE gate (EVAL_DIRECTIONS task 2, pre-registered
# HANDOFF_2026-09-02 §2.2) — the only untested-live knob after mode-of-N's
# live death. Two arms, standard rung (8 x 120 s vs CPU, T=0.5/buttons
# 0.5, d0 headless):
#   CRITIC  --critic eval_runs/0901_critic_v13/critic.nx --critic-k 16
#   BASE    plain sampling (the same decode the v1.3 live look used)
#
# PRE-REGISTERED GATE (read before the numbers): frozen-input <= 0.20,
# 7/8 runs to cap, F1 airdodge-in-danger not worse (edge_scorecard),
# F2 commitment rate (commitment_scorecard), F3 approach_delta
# (neutral_range_scorecard). Offline expectations: selector-mode +6.2
# in-dist / +2.1 fresh; L9 says treat any offline-only justification as
# void without this gate.
#
#   systemd-run --user --unit=critic-gate --collect \
#     --working-directory=$PWD \
#     -p StandardOutput=append:$PWD/logs/critic_gate.log \
#     -p StandardError=append:$PWD/logs/critic_gate.log \
#     devenv shell -- bash scripts/critic_gate.sh
set -uo pipefail
cd "$(dirname "$0")/.."

OUT="${OUT:-eval_runs/0902_critic_gate}"
RUNS="${RUNS:-8}"
GAME_SECONDS="${GAME_SECONDS:-120}"
POLICY="${POLICY:-checkpoints/fox_gen_v1.3_ARrefit_policy.bin}"
CRITIC="${CRITIC:-eval_runs/0901_critic_v13/critic.nx}"
CRITIC_K="${CRITIC_K:-16}"
EXPERT_GLOB="replays/erickfm_ranked/FOX/extracted/*.slp"
TABLE="$OUT/gate_table.txt"

if pgrep -x beam.smp >/dev/null; then
  echo "a beam is already live — refusing (second-EXLA-client law)" >&2; exit 3
fi

mkdir -p "$OUT"
: > "$TABLE"

echo "critic gate runs=$RUNS seconds=$GAME_SECONDS policy=$POLICY critic=$CRITIC k=$CRITIC_K started=$(date -Is)" | tee -a "$TABLE"

# run_arm <name> <critic_expected: yes|no> [extra flags...]
run_arm() {
  local name="$1" critic_expected="$2"; shift 2
  echo "=== $name" | tee -a "$TABLE"

  bash scripts/eval_live_protocol.sh "$POLICY" "$OUT/$name" \
    --runs "$RUNS" --seconds "$GAME_SECONDS" --dummy cpu \
    --temperature 0.5 -- --frame-delay 0 --headless --buttons-temperature 0.5 "$@" \
    > "$OUT/$name.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "$name PROTOCOL FAILED (rc=$rc) — see $OUT/$name.log" | tee -a "$TABLE"
    return
  fi

  local clean
  clean=$(sed -e 's/\x1b\[[0-9;]*m//g' "$OUT/$name"/r*.log)

  # knob assertions (GOTCHA #103/#104 discipline)
  if ! grep -qa -- "buttons: 0.5" <(echo "$clean"); then
    echo "$name KNOB ASSERTION FAILED: 'buttons: 0.5' absent — discard arm" | tee -a "$TABLE"
    return
  fi

  if [ "$critic_expected" = yes ]; then
    if ! grep -qa "Critic-selector decode ACTIVE" <(echo "$clean"); then
      echo "$name ASSERTION FAILED: critic banner absent — discard arm" | tee -a "$TABLE"
      return
    fi
  else
    if grep -qa "Critic-selector decode ACTIVE" <(echo "$clean"); then
      echo "$name ASSERTION FAILED: critic banner present in BASE arm — discard arm" | tee -a "$TABLE"
      return
    fi
  fi

  echo "$name: $(ls "$OUT/$name"/r*.slp 2>/dev/null | wc -l) replays" | tee -a "$TABLE"
  echo "$clean" | grep -ao 'Staleness: [0-9]*/[0-9]* sends stale ([0-9.]*%)' | sed 's/^/    /' | tee -a "$TABLE"
}

run_arm CRITIC yes --critic "$CRITIC" --critic-k "$CRITIC_K"
run_arm BASE no

echo "=== durations (7/8-to-cap gate)" | tee -a "$TABLE"
for arm in CRITIC BASE; do
  [ -d "$OUT/$arm" ] || continue
  sed -e 's/\x1b\[[0-9;]*m//g' "$OUT/$arm"/r*.log 2>/dev/null \
    | grep -ao 'Final stats: [0-9]* frames, [0-9]* inferences' \
    | awk -v arm="$arm" -v cap="$GAME_SECONDS" '
        { f=$3; n++; s+=f; if (f/60 >= cap) c++ }
        END { if (n) printf "%s: n=%d mean=%.1fs reaching_cap=%d/%d\n", arm, n, s/n/60, c, n
              else printf "%s: no Final stats lines\n", arm }' | tee -a "$TABLE"
done

echo "=== loop_report (frozen-input <= 0.20 gate)" | tee -a "$TABLE"
mix run scripts/loop_report.exs --bot-port 1 --per-game --out "$OUT/loops" \
  "$OUT"/CRITIC/r*.slp "$OUT"/BASE/r*.slp >> "$TABLE" 2>&1 || echo "loop_report FAILED" | tee -a "$TABLE"

echo "=== F1: edge_scorecard (airdodge-in-danger, routes)" | tee -a "$TABLE"
mix run scripts/edge_scorecard.exs \
  --set expert="$EXPERT_GLOB" --expert expert --expert-limit 600 \
  --set CRITIC="$OUT/CRITIC/r*.slp" \
  --set BASE="$OUT/BASE/r*.slp" \
  --out "$OUT/edge_scorecard.md" >> "$TABLE" 2>&1 || echo "edge_scorecard FAILED" | tee -a "$TABLE"

echo "=== F2: commitment_scorecard" | tee -a "$TABLE"
mix run scripts/commitment_scorecard.exs \
  --set expert="$EXPERT_GLOB" --expert expert --expert-limit 600 \
  --set CRITIC="$OUT/CRITIC/r*.slp" \
  --set BASE="$OUT/BASE/r*.slp" \
  --out "$OUT/commitment.md" >> "$TABLE" 2>&1 || echo "commitment FAILED" | tee -a "$TABLE"

echo "=== F3: neutral_range_scorecard (approach_delta)" | tee -a "$TABLE"
mix run scripts/neutral_range_scorecard.exs \
  --set expert="$EXPERT_GLOB" \
  --set CRITIC="$OUT/CRITIC/r*.slp" \
  --set BASE="$OUT/BASE/r*.slp" \
  --limit-files 40 \
  --out "$OUT/neutral_range.md" >> "$TABLE" 2>&1 || echo "neutral_range FAILED" | tee -a "$TABLE"

echo "CRITIC GATE DONE $(date -Is) — apply the pre-registered gate BEFORE reading exploratory numbers; then the human look (g6 rule)" | tee -a "$TABLE"
