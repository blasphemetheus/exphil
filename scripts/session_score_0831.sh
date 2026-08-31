#!/usr/bin/env bash
# Score the 0831 live-look sessions (v1.1-AR and v1.1-IND vs Bradley)
# with the full human-rung instrument set. B1's 0829 human session rides
# along as the human-rung reference.
#
#   systemd-run --user --unit=session-score --working-directory=$PWD \
#     --collect -p StandardOutput=append:$PWD/logs/session_score_0831.log \
#     -p StandardError=append:$PWD/logs/session_score_0831.log \
#     devenv shell -- bash scripts/session_score_0831.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

OUT=eval_runs/0831_session_score
mkdir -p "$OUT" logs
EXPERT_GLOB="replays/erickfm_ranked/FOX/extracted/*.slp"
AR_GLOB="eval_runs/0831_livelook_v11ar/2026-08-Mainline/*.slp"
IND_GLOB="eval_runs/0831_livelook_v11ind/2026-08-Mainline/*.slp"
B1_GLOB="eval_runs/0829_livelook_awbc_B1/2026-08-Mainline/*.slp"

run() { echo "=== $1 $(date -Is)"; shift; "$@" || echo "!!! FAILED: $*"; }

run loop_report mix run scripts/loop_report.exs --bot-port 1 --per-game \
  --out "$OUT/loops" $AR_GLOB $IND_GLOB

run coach_AR mix run scripts/coach_report.exs --char fox --bot-port 1 \
  --out "$OUT/coach_AR" $AR_GLOB
run coach_IND mix run scripts/coach_report.exs --char fox --bot-port 1 \
  --out "$OUT/coach_IND" $IND_GLOB

run edge_scorecard mix run scripts/edge_scorecard.exs \
  --set expert="$EXPERT_GLOB" --expert expert --expert-limit 600 \
  --set AR_human="$AR_GLOB" --set IND_human="$IND_GLOB" --set B1_human="$B1_GLOB" \
  --out "$OUT/edge_scorecard.md"

run death_classifier mix run scripts/death_classifier.exs \
  --set expert="$EXPERT_GLOB" --expert expert --expert-limit 600 \
  --set AR_human="$AR_GLOB" --set IND_human="$IND_GLOB" --set B1_human="$B1_GLOB" \
  --out "$OUT/death_classifier.md"

run situation_hist mix run scripts/situation_hist.exs \
  --set expert="$EXPERT_GLOB" --expert expert --expert-limit 600 \
  --set AR_human="$AR_GLOB" --set IND_human="$IND_GLOB" --set B1_human="$B1_GLOB" \
  --out "$OUT/situation_hist.md"

run punish_quality mix run scripts/punish_quality.exs \
  --set expert="$EXPERT_GLOB" --expert expert --expert-limit 400 \
  --set AR_human="$AR_GLOB" --set IND_human="$IND_GLOB" --set B1_human="$B1_GLOB" \
  --out "$OUT/punish_quality.md"

run reaction_latency mix run scripts/reaction_latency.exs \
  --set expert="$EXPERT_GLOB" --expert expert --expert-limit 300 \
  --set AR_human="$AR_GLOB" --set IND_human="$IND_GLOB" --set B1_human="$B1_GLOB" \
  --out "$OUT/reaction_latency.md"

run neutral_exchange mix run scripts/neutral_exchange.exs \
  --set expert="$EXPERT_GLOB" --expert expert \
  --set AR_human="$AR_GLOB" --set IND_human="$IND_GLOB" --set B1_human="$B1_GLOB" \
  --out "$OUT/neutral_exchange.md"

echo "=== SESSION SCORE DONE $(date -Is)"
