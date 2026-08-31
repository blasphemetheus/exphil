#!/usr/bin/env bash
# EVAL_DIRECTIONS remaining instruments — first runs (2026-08-30).
# Each step is independent (|| true): a bug in one new instrument must not
# kill the rest. Sets: expert (per-file fox detect where supported),
# the 8a bracket-2 arms, ep10's banked CPU games, and the B1 human session.
#
#   systemd-run --user --unit=instruments-first --collect \
#     --working-directory=/home/blewf/git/exphil \
#     -p StandardOutput=append:/home/blewf/git/exphil/logs/instruments_first.log \
#     -p StandardError=append:/home/blewf/git/exphil/logs/instruments_first.log \
#     devenv shell -- bash scripts/instruments_first_run.sh
set -uo pipefail
cd "$(dirname "$0")/.."

EXPERT='replays/erickfm_ranked/FOX/extracted/*.slp'
AR='eval_runs/0830_arhead_score2/AR/r*.slp'
IND='eval_runs/0830_arhead_score2/IND/r*.slp'
EP10='eval_runs/0829_mode_of_n/base/r*.slp'
B1H='eval_runs/0829_livelook_awbc_B1/2026-08-Mainline/*.slp'

if pgrep -x beam.smp >/dev/null; then
  echo "a beam is already live — refusing (second-EXLA-client law)" >&2; exit 3
fi

echo "=== E3 expert pathology baselines $(date -Is)"
mix run scripts/expert_pathology.exs --replays "$EXPERT" --char-id 2 --limit-files 400 \
  --out eval_runs/0830_expert_pathology/RESULTS.md || true

echo "=== C2 punish quality $(date -Is)"
mix run scripts/punish_quality.exs \
  --set expert="$EXPERT" --expert expert --expert-limit 400 \
  --set AR="$AR" --set IND="$IND" --set ep10_cpu="$EP10" --set B1_human="$B1H" \
  --out eval_runs/0830_punish_quality/RESULTS.md || true

echo "=== C3 reaction latency $(date -Is)"
mix run scripts/reaction_latency.exs \
  --set expert="$EXPERT" --expert expert --expert-limit 300 \
  --set AR="$AR" --set IND="$IND" --set ep10_cpu="$EP10" --set B1_human="$B1H" \
  --out eval_runs/0830_reaction_latency/RESULTS.md || true

echo "=== B1 action-family match $(date -Is)"
mix run scripts/action_family_match.exs \
  --set expert="$EXPERT" --expert expert --expert-limit 400 \
  --set AR="$AR" --set IND="$IND" --set ep10_cpu="$EP10" --set B1_human="$B1H" \
  --out eval_runs/0830_family_match/RESULTS.md || true

echo "=== D3 decode sensitivity (banked buttons sweep) $(date -Is)"
mix run scripts/decode_sensitivity.exs \
  --json eval_runs/0828_loop_rescore/buttons_sweep/report.json --knobs 0.5,0.6,0.7,1.0 \
  --out eval_runs/0830_decode_sensitivity/RESULTS.md || true

echo "=== C1 noise floor (HANDOFF task 4) $(date -Is)"
mix run scripts/neutral_exchange.exs \
  --set expert="$EXPERT" --expert expert --expert-limit 800 \
  --set F05_sweep='eval_runs/0828_buttons_temp/btn0.5/*.slp' \
  --set F05_bracket='eval_runs/0828_argmax_buttons/btn0.5/*.slp' \
  --set F05_base="$EP10" \
  --out eval_runs/0829_neutral_exchange/floor.md || true

echo "=== D2 transfer table runner $(date -Is)"
bash scripts/d2_transfer.sh || true

echo "INSTRUMENTS FIRST-RUN DONE $(date -Is)"
