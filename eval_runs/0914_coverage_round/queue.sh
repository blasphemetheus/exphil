#!/usr/bin/env bash
# stages 2 + 3 after stage 1 (collect.sh) finished
cd "$(dirname "$0")/../.."
bash eval_runs/0914_coverage_round/gen2.sh
bash eval_runs/0914_coverage_round/teach.sh
bash eval_runs/0914_coverage_round/train.sh
