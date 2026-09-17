#!/usr/bin/env bash
# One round = matched-seed fits then offline gates, for a given dir prefix.
cd /home/blewf/git/exphil/eval_runs/0915_fox_v3_preflight
bash run_matched_seeds.sh "$1" && bash run_offline_gates.sh "$1"
