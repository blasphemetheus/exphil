#!/usr/bin/env bash
# EVAL_DIRECTIONS batch 2 (2026-08-29 late): D1b, E2, A2, C1, C4, B3 —
# sequential, one beam at a time. Each stage's rc goes to $OUT/CHAIN.md.
#
#   systemd-run --user --unit=evaldir2 --collect \
#     --working-directory=/home/blewf/git/exphil \
#     -p StandardOutput=append:/home/blewf/git/exphil/logs/eval_directions_chain.log \
#     -p StandardError=append:/home/blewf/git/exphil/logs/eval_directions_chain.log \
#     devenv shell -- bash scripts/eval_directions_chain.sh
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=eval_runs/0829_evaldir2; mkdir -p "$OUT" logs
CHAIN="$OUT/CHAIN.md"; rm -f "$OUT/.done"

if pgrep -x beam.smp >/dev/null; then echo "a beam is live — refusing" >&2; exit 3; fi

stage() { local name="$1" log="$2"; shift 2
  echo "=== $name start $(date -Is)" | tee -a "$CHAIN"
  "$@" > "$log" 2>&1; local rc=$?
  echo "=== $name end $(date -Is) rc=$rc log=$log" | tee -a "$CHAIN"; return $rc; }

EXPERT="replays/erickfm_ranked/FOX/extracted/*.slp"
EP10=checkpoints/fox_gen_v1_20260825_210355_ep10.bin
BOTSETS=(
  --set ep10_cpu='eval_runs/0829_mode_of_n/base/*.slp'
  --set btn06_cpu='eval_runs/0828_argmax_buttons/btn0.6/*.slp'
  --set mode16_cpu='eval_runs/0829_mode_of_n/mode16/*.slp'
  --set ep10_T1_human='eval_runs/0828_session/*.slp'
  --set ep10_human='eval_runs/0828_livelook_btn05/2026-08-Mainline/*.slp'
  --set B1_human='eval_runs/0829_livelook_awbc_B1/2026-08-Mainline/*.slp'
  --set B2_human='eval_runs/0829_livelook_awbc_B2/2026-08-Mainline/*.slp'
  --set B3_human='eval_runs/0829_livelook_awbc_B3/2026-08-Mainline/*.slp'
)

# D1b — TV-distance noise floor: the D1 batches as separate sets
stage D1b_tv_floor logs/evaldir_d1b.log mix run scripts/situation_hist.exs \
  --set expert="$EXPERT" --expert expert --expert-limit 800 \
  --set F05_sweep0828='eval_runs/0828_buttons_temp/btn0.5/*.slp' \
  --set F05_bracket0828='eval_runs/0828_argmax_buttons/btn0.5/*.slp' \
  --set F05_base0829='eval_runs/0829_mode_of_n/base/*.slp' \
  --set F10_live0826='eval_runs/0826_gen_v1_sweep/live_ep10/*.slp' \
  --set F10_scalar0826='eval_runs/0826_gen_v1_sweep/per_head_temp/scalar_05/*.slp' \
  --set F10_base0828='eval_runs/0828_argmax_buttons/base/*.slp' \
  --set F10_btn10_0828='eval_runs/0828_buttons_temp/btn1.0/*.slp' \
  --out eval_runs/0829_noise_floor/tv_floor.md

# E2 — rare-event coverage on the FULL corpus
stage E2_rare_events logs/evaldir_e2.log mix run scripts/rare_event_coverage.exs \
  --replays "$EXPERT" --port 1 --stride 5 --out eval_runs/0829_rare_events/RESULTS.md

# A2 — edgeguard / recovery scorecards
stage A2_edge_scorecard logs/evaldir_a2.log mix run scripts/edge_scorecard.exs \
  --set expert="$EXPERT" --expert expert --expert-limit 800 "${BOTSETS[@]}" \
  --out eval_runs/0829_edge_scorecard/RESULTS.md

# C1 — neutral exchanges
stage C1_neutral_exchange logs/evaldir_c1.log mix run scripts/neutral_exchange.exs \
  --set expert="$EXPERT" --expert expert --expert-limit 800 "${BOTSETS[@]}" \
  --out eval_runs/0829_neutral_exchange/RESULTS.md

# C4 — death classifier
stage C4_death_classifier logs/evaldir_c4.log mix run scripts/death_classifier.exs \
  --set expert="$EXPERT" --expert expert --expert-limit 800 "${BOTSETS[@]}" \
  --out eval_runs/0829_death_classifier/RESULTS.md

# B3 — entropy by situation (GPU; ~19 labels x 80 frames x 61 inferences)
stage B3_entropy logs/evaldir_b3.log mix run scripts/interp_entropy_by_situation.exs \
  --policy "$EP10" --replays "$EXPERT" --port 1 --limit-files 20 --frames-per-label 80 \
  --out eval_runs/0829_entropy/RESULTS.md

echo "CHAIN DONE $(date -Is)" | tee -a "$CHAIN"; touch "$OUT/.done"
