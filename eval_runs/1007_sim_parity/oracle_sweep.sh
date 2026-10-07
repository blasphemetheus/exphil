#!/usr/bin/env bash
# Corpus-level declaration oracle (2026-10-07): one flag toggled from the
# baseline per run, on the 1,122 recorder-2.0.1 FD Fox games. Pass count =
# games exact end to end. Baseline = dolphin-legacy key + cardinals off +
# 0.84 shield drop off (313).
set -u
cd /home/blewf/git/msl-parity
D=/home/blewf/git/exphil/eval_runs/1007_sim_parity
source .venv/bin/activate
export MSL_DATA_DIR=$PWD/data
V="python -m tools.validation.validate_replay --backend native --no-build --diagnostic --workers ${WORKERS:-4}"
B="--no-ucf-cardinals-1-0 --no-ucf-shield-drop-084"
run() { name=$1; shift; xargs -d '\n' -a "$D/fd_fox_v2.txt" $V "$@" > "$D/oracle_$name.log" 2>&1
  echo "$name: $(grep -E '^DIAGNOSTIC +([0-9,]+)/\1 ' "$D/oracle_$name.log" | wc -l) exact  | $(grep -E '^\[native\] summary' "$D/oracle_$name.log" | cut -c1-80)"; }
run fnmsubs_retail       --fnmsubs-profile retail $B
run no_shield_sdi        --fnmsubs-profile dolphin-legacy $B --no-ucf-shield-sdi
run no_sdi               --fnmsubs-profile dolphin-legacy $B --no-ucf-sdi
run no_shield_drop_ext   --fnmsubs-profile dolphin-legacy $B --no-ucf-shield-drop-extended
run cardinals_on         --fnmsubs-profile dolphin-legacy --ucf-cardinals-1-0 --no-ucf-shield-drop-084
run shield_drop_084_on   --fnmsubs-profile dolphin-legacy --no-ucf-cardinals-1-0 --ucf-shield-drop-084
echo ORACLE DONE
