#!/usr/bin/env bash
# 8a score — v1.1-ARhead vs v1.1-INDhead (AUTOREGRESSIVE_HEAD_PLAN §6).
#
# READ THE PRE-REGISTERED DECISION RULE (plan §6) BEFORE THE NUMBERS.
# Primary: A2 recovery first-route (up-B + double-jump share, airdodge
# share, died-given-route) and the live joint coincidence P(stick up | B)
# from the arms' own replays. Secondary: B2 TV, C4 % at death, durations,
# loop_report, no-collapse.
#
# Protocol: the standard rung — 8 x 120 s vs CPU, T=0.5 / buttons 0.5,
# delay 0, headless. ep10's banked base games (eval_runs/0829_mode_of_n/base)
# ride along as the unchanged-checkpoint reference; expert baselines are the
# same char-mixed port-1 set the original A2 run used (comparability; the
# fox-only re-read is a separate follow-up — see 0830_corpus_mix).
#
# Launch under systemd, never from an agent shell:
#   systemd-run --user --unit=arhead-score --collect \
#     --working-directory=/home/blewf/git/exphil \
#     -p StandardOutput=append:/home/blewf/git/exphil/logs/arhead_score.log \
#     -p StandardError=append:/home/blewf/git/exphil/logs/arhead_score.log \
#     devenv shell -- bash scripts/arhead_score.sh
set -uo pipefail
cd "$(dirname "$0")/.."

OUT="${OUT:-eval_runs/0830_arhead_score}"
RUNS="${RUNS:-8}"
GAME_SECONDS="${GAME_SECONDS:-120}"
AR_POLICY="${AR_POLICY:-checkpoints/fox_gen_v1.1_ARhead_policy.bin}"
IND_POLICY="${IND_POLICY:-checkpoints/fox_gen_v1.1_INDhead_policy.bin}"
EXPERT_GLOB="replays/erickfm_ranked/FOX/extracted/*.slp"
TABLE="$OUT/score_table.txt"
DONE_MARKER="$OUT/.score_done"

if pgrep -x beam.smp >/dev/null; then
  echo "a beam is already live — refusing (second-EXLA-client law)" >&2; exit 3
fi

mkdir -p "$OUT"
: > "$TABLE"
rm -f "$DONE_MARKER"

echo "arhead score runs=$RUNS seconds=$GAME_SECONDS started=$(date -Is)" | tee -a "$TABLE"

# run_arm <name> <policy> <ar_expected: yes|no>
run_arm() {
  local name="$1" policy="$2" ar_expected="$3" rc n clean
  echo "=== $name — $policy" | tee -a "$TABLE"

  bash scripts/eval_live_protocol.sh "$policy" "$OUT/$name" \
    --runs "$RUNS" --seconds "$GAME_SECONDS" --dummy cpu \
    --temperature 0.5 -- --frame-delay 0 --headless --buttons-temperature 0.5 \
    > "$OUT/$name.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "$name PROTOCOL FAILED (rc=$rc) — see $OUT/$name.log" | tee -a "$TABLE"
    return
  fi

  clean=$(sed -e 's/\x1b\[[0-9;]*m//g' "$OUT/$name"/r*.log)

  # knob assertions (GOTCHA #103/#104 discipline)
  if ! grep -qa -- "buttons: 0.5" <(echo "$clean"); then
    echo "$name KNOB ASSERTION FAILED: 'buttons: 0.5' absent — discard arm" | tee -a "$TABLE"
    return
  fi

  if [ "$ar_expected" = yes ]; then
    if ! grep -qa "Autoregressive controller head ACTIVE" <(echo "$clean"); then
      echo "$name HEAD ASSERTION FAILED: AR banner absent — discard arm" | tee -a "$TABLE"
      return
    fi
  else
    if grep -qa "Autoregressive controller head ACTIVE" <(echo "$clean"); then
      echo "$name HEAD ASSERTION FAILED: AR banner present in IND arm — discard arm" | tee -a "$TABLE"
      return
    fi
  fi

  n=$(ls "$OUT/$name"/r*.slp 2>/dev/null | wc -l)
  echo "$name: $n replays" | tee -a "$TABLE"
  echo "$clean" | grep -aiE 'stale|skipped' | sed 's/^/    /' | tee -a "$TABLE"
}

run_arm AR  "$AR_POLICY"  yes
run_arm IND "$IND_POLICY" no

# --- axis 3: durations from the run logs (truncation-proof, GOTCHA #102) ----
echo "=== durations from run logs" | tee -a "$TABLE"
for arm in AR IND; do
  [ -d "$OUT/$arm" ] || continue
  sed -e 's/\x1b\[[0-9;]*m//g' "$OUT/$arm"/r*.log 2>/dev/null \
    | grep -ao 'Final stats: [0-9]* frames, [0-9]* inferences' \
    | awk -v arm="$arm" -v cap="$GAME_SECONDS" '
        { f=$3; inf=$5; n++; s+=f; si+=inf; if (f/60 >= cap) c++; if (min==""||f<min) min=f; if (f>max) max=f }
        END { if (n) printf "%s: n=%d mean=%.1fs [%.1f-%.1f] reaching_cap=%d/%d inferences/frame=%.1f\n", arm, n, s/n/60, min/60, max/60, c, n, si/s
              else printf "%s: no Final stats lines\n", arm }' | tee -a "$TABLE"
done

# --- pathologies (collapse gate) --------------------------------------------
echo "=== loop_report" | tee -a "$TABLE"
mix run scripts/loop_report.exs --bot-port 1 --per-game --out "$OUT/loops" \
  "$OUT"/AR/r*.slp "$OUT"/IND/r*.slp >> "$TABLE" 2>&1 || echo "loop_report FAILED" | tee -a "$TABLE"

# --- PRIMARY 1: A2 recovery/edgeguard scorecards ----------------------------
echo "=== A2 edge_scorecard (PRIMARY — routes + died-given-route)" | tee -a "$TABLE"
mix run scripts/edge_scorecard.exs \
  --set expert="$EXPERT_GLOB" --expert expert --expert-limit 600 \
  --set AR="$OUT/AR/r*.slp" \
  --set IND="$OUT/IND/r*.slp" \
  --set ep10_cpu='eval_runs/0829_mode_of_n/base/r*.slp' \
  --out "$OUT/edge_scorecard.md" >> "$TABLE" 2>&1 || echo "edge_scorecard FAILED" | tee -a "$TABLE"

# --- PRIMARY 2: live joint coincidence P(stick up | B) ----------------------
# joint_head_audit measures the PLAYER stream of whatever glob it gets; on
# the arms' own replays its pair table IS the live coincidence.
echo "=== joint coincidence from arm replays (PRIMARY — P(up|B) vs marginal)" | tee -a "$TABLE"
for arm in AR IND; do
  [ -d "$OUT/$arm" ] || continue
  mix run scripts/joint_head_audit.exs --replays "$OUT/$arm/r*.slp" --port 1 \
    --out "$OUT/coincidence_$arm.md" >> "$TABLE" 2>&1 \
    || echo "joint_head_audit $arm FAILED" | tee -a "$TABLE"
done

# --- secondary: C4 death forensics + B2 TV ----------------------------------
echo "=== C4 death_classifier" | tee -a "$TABLE"
mix run scripts/death_classifier.exs \
  --set expert="$EXPERT_GLOB" --expert expert --expert-limit 600 \
  --set AR="$OUT/AR/r*.slp" --set IND="$OUT/IND/r*.slp" \
  --out "$OUT/death_classifier.md" >> "$TABLE" 2>&1 || echo "death_classifier FAILED" | tee -a "$TABLE"

echo "=== B2 TV (situation_hist)" | tee -a "$TABLE"
mix run scripts/situation_hist.exs \
  --set expert="$EXPERT_GLOB" --expert expert --expert-limit 600 \
  --set AR="$OUT/AR/r*.slp" --set IND="$OUT/IND/r*.slp" \
  --set ep10_cpu='eval_runs/0829_mode_of_n/base/r*.slp' \
  --out "$OUT/situation_hist.md" >> "$TABLE" 2>&1 || echo "situation_hist FAILED" | tee -a "$TABLE"

echo "ARHEAD SCORE DONE $(date -Is) — apply plan §6: SIGNAL needs AR route shares >=2x IND with died-given-route lower, TV not worse than floor, no collapse; then the human look gates the recipe (g6 rule)" | tee -a "$TABLE"
touch "$DONE_MARKER"
