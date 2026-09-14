#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
O=eval_runs/0914_coverage_round
{ sed -n '1,/^export EXLA_TARGET/p' $O/teach.sh; echo 'say "=== resume 2d (clips reused)"'; sed -n '/=== stage 2d/,$p' $O/teach.sh; } > $O/teach_resume.sh
bash $O/teach_resume.sh
grep -q "train 0 handoffs" $O/split.log && { echo "split still empty"; exit 1; }
bash $O/train.sh
