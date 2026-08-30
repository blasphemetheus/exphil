#!/usr/bin/env bash
# D1 — noise floors: the same checkpoint + decode, scored across batches
# played on different days. Answers "which metrics can ever resolve a real
# difference?" (EVAL_DIRECTIONS.md D1).
#
# Two identical-condition families exist in the bank (fox_gen_v1 ep10 vs
# CPU dummy, FD, delay 0, 120 s):
#   F05  scalar T=0.5 + buttons T=0.5   — 0828_buttons_temp/btn0.5 (5),
#        0828_argmax_buttons/btn0.5 (7 readable), 0829_mode_of_n/base (8)
#   F10  scalar T=0.5, buttons raw 1.0  — 0826 live_ep10 (3), 0826
#        per_head_temp/scalar_05 (5), 0828_argmax_buttons/base (8),
#        0828_buttons_temp/btn1.0 (5)
#
# loop_report groups by parent directory, so each batch is one row; coach
# is per game. Output: per metric, batch means, the max/min ratio BETWEEN
# batches (the day-to-day drift a real effect must exceed), and the pooled
# per-game spread.
#
#   systemd-run --user --unit=noise-floor --collect \
#     --working-directory=/home/blewf/git/exphil \
#     -p StandardOutput=append:/home/blewf/git/exphil/logs/noise_floor.log \
#     -p StandardError=append:/home/blewf/git/exphil/logs/noise_floor.log \
#     devenv shell -- bash scripts/noise_floor.sh
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=eval_runs/0829_noise_floor
mkdir -p "$OUT"
T="$OUT/raw.txt"; : > "$T"

declare -A FAM
FAM[F05]="eval_runs/0828_buttons_temp/btn0.5 eval_runs/0828_argmax_buttons/btn0.5 eval_runs/0829_mode_of_n/base"
FAM[F10]="eval_runs/0826_gen_v1_sweep/live_ep10 eval_runs/0826_gen_v1_sweep/per_head_temp/scalar_05 eval_runs/0828_argmax_buttons/base eval_runs/0828_buttons_temp/btn1.0"

for fam in F05 F10; do
  echo "=== $fam loop" | tee -a "$T"
  files=$(for d in ${FAM[$fam]}; do find "$d" -maxdepth 1 -name 'r*.slp' -size +150k; done | sort)
  mix run scripts/loop_report.exs --bot-port 1 --per-game --out "$OUT/loops_$fam" $files 2>&1 \
    | sed -E 's/\x1b\[[0-9;]*m//g' | grep -aE '^\|' >> "$T"
  echo "=== $fam coach" | tee -a "$T"
  for f in $files; do
    batch=$(basename "$(dirname "$f")")_$(basename "$(dirname "$(dirname "$f")")")
    mix run scripts/coach_report.exs --char fox --bot-port 1 --out "$OUT/coach_${fam}_${batch}_$(basename "$f" .slp)" "$f" 2>&1 \
      | grep -a 'SCORE:' | sed "s|^|$fam $batch $(basename "$f" .slp) |" >> "$T"
  done
done
echo "NOISE FLOOR DONE $(date -Is)" | tee -a "$T"
