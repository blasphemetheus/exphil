#!/usr/bin/env bash
set -euo pipefail
cd /home/blewf/git/exphil
OUT=eval_runs/0915_float_input/coverage01
export EXLA_TARGET=host
exec > >(tee -a "$OUT/progress.log") 2>&1
SOURCES=(eval_runs/0914_coverage_round/rollouts/*/r*.slp eval_runs/local_multishine_20260913_224806/2026-09-Mainline/*.slp eval_runs/0911_g23a_live/cpu_rollouts/r{1,2}.slp)
sha256sum "${SOURCES[@]}" > "$OUT/sources.sha256"
echo "$(date -Is) Mine raw source handoffs"
mix run --no-start scripts/mine_coverage_handoffs.exs --out "$OUT/mined.json" --per-replay-neutral 3 --per-replay-hit 3 --min-frame 240 --max-frame 4800 --gap 300 "${SOURCES[@]}" > "$OUT/mine.log" 2>&1
jq '.entries|length' "$OUT/mined.json"
echo "$(date -Is) Execute teachers with exact prefixes"
mix run scripts/scenario_suite.exs --driver teacher --audit-teacher-labels --reaction-delay 4 --character fox --prefix-history committed \
 --manifest "$OUT/mined.json" --runs 1 --window 360 --response-opponent neutral --live-af --no-orphan-sweep --quiet --trace-policy-inputs \
 --float-ports 1,2 --no-pipe-shim --dolphin /home/blewf/.local/share/slippi/exi-ai-float-v3/dolphin-emu-headless \
 --out "$OUT/teacher.json" --run-dir "$OUT/teacher" > "$OUT/teacher.log" 2>&1
jq '{runs:(.runs|length), errors:.errored_runs, diverged:.diverged_runs, passes:([.runs[]|select(.pass==true)]|length), exact:([.runs[]|select(.prefix_audit.valid==true)]|length)}' "$OUT/teacher.json"
jq '.runs |= map(select(.error == null and .diverged == false and .pass == true and .truncated == null and .prefix_audit.valid == true and (.timing_valid == true or .timing_valid == null)))' "$OUT/teacher.json" > "$OUT/teacher_qualified.json"
mix run --no-start scripts/check_recovery_targets.exs --scores "$OUT/teacher_qualified.json" --delay 4 --allow-on-loop --out "$OUT/targets_d4.json" > "$OUT/targets_d4.log" 2>&1
jq -e '.valid == true' "$OUT/targets_d4.json"
mix run scripts/prepare_recorded_context.exs --out-dir "$OUT/clips" --delay 4 --queue-depth 5 --context 20 --reports "$OUT/targets_d4.json" --policy none --tag-source > "$OUT/clips.log" 2>&1
elixir "$OUT/split.exs" "$OUT"
echo "$(date -Is) Teacher/export stages done"
