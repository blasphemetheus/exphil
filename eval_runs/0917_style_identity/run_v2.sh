#!/usr/bin/env bash
set -u
cd /home/blewf/git/exphil
R=eval_runs/0917_style_identity
bash $R/run_fingerprints.sh
devenv shell -- env EXPHIL_GPU=0 mix run scripts/style_calibrate.exs --rows erickfm=$R/erickfm_fox.jsonl --rows yeti=$R/yeti_fox.jsonl --out $R/calibration.json --min-games 5 --quiet > $R/calibration.log 2>&1
echo "calibrate exit $?"
devenv shell -- env EXPHIL_GPU=0 mix run scripts/style_metric_eval.exs --erickfm $R/erickfm_fox.jsonl --yeti $R/yeti_fox.jsonl --out $R/metric_eval.json --quiet > $R/metric_eval.log 2>&1
echo "metric eval exit $?"
