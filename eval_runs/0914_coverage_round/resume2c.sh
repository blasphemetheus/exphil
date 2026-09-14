#!/usr/bin/env bash
# resume: stage 2c/2d (validate/export/split) from the existing teacher.json, then stage 3
cd "$(dirname "$0")/../.."
O=eval_runs/0914_coverage_round
rm -rf $O/round21 $O/clips $O/clips_train $O/clips_heldout $O/manifest_train.json $O/manifest_heldout.json $O/targets_d4.* $O/teacher_qualified.json
mv $O/round21 $O/round21_attempt0_no_clips 2>/dev/null
# run only 2c/2d of teach.sh: extract from the "stage 2c" marker on
sed -n '/=== stage 2c/,$p' $O/teach.sh > $O/teach_2c.part
{ sed -n '1,/^export EXLA_TARGET/p' $O/teach.sh; echo 'say "=== resume 2c (teacher.json reused)"'; cat $O/teach_2c.part; } > $O/teach_resume.sh
bash $O/teach_resume.sh
bash $O/train.sh
