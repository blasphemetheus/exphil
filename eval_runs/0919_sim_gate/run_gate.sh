#!/usr/bin/env bash
# Sim validation gate (MELEE_SIM_USES.md): run melee-sim-light's strict replay
# validator on OUR replays, using its existing binaries (--no-build so we never
# race another session's rebuild). Output: <gate>/results/<name>.txt + summary.
set -u
SIM="${SIM:-$HOME/git/melee-sim-light}"
G=/home/blewf/git/exphil/eval_runs/0919_sim_gate
mkdir -p "$G/results"
cd "$SIM" || exit 1
[ -x .venv/bin/python ] || { echo "no .venv in $SIM"; exit 1; }
frames="${FRAMES:-0}"
jq -r '.replays[] | [.path, .source] | @tsv' "$G/manifest.json" | while IFS=$'\t' read -r p src; do
  name="$(basename "${p%.slp}")"
  MSL_DATA_DIR="$SIM/data" .venv/bin/python -m tools.validation.validate_replay "$p" --backend native --no-build --frames "$frames" --timing > "$G/results/$name.txt" 2>&1
  echo "$src	$name	exit $?	$(grep -aoE '(EXACT|exact|PASS|FAIL|CLASSIFIED|mismatch)[^\n]{0,80}' "$G/results/$name.txt" | head -1)"
done | tee "$G/summary.tsv"
