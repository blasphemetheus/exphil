#!/usr/bin/env bash
# resume from stage 2b (teacher with --audit-teacher-labels), mined.json reused
cd "$(dirname "$0")/../.."
O=eval_runs/0914_coverage_round
{ sed -n '1,/^export EXLA_TARGET/p' $O/teach.sh; echo 'say "=== resume 2b (mined.json reused; teacher with --audit-teacher-labels)"'; sed -n '/=== stage 2b/,$p' $O/teach.sh; } > $O/teach_resume.sh
bash $O/teach_resume.sh
bash $O/train.sh
