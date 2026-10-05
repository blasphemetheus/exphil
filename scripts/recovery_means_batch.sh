#!/usr/bin/env bash
# Recovery-means scorecard over the coherence queue's models (2026-10-05):
# the windowed recipe pair, the plain baseline and the carried-state run.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
for name in evt2_ck8 evt2_ck8w3 base bptt_evt2_ck8_e5 "$@"; do
  out=eval_runs/1001_queue/$name
  echo "== $name ($(date +%H:%M))"
  mix run --no-compile scripts/recovery_means.exs --policy checkpoints/coh_$name/model_best_policy.bin --label "$name" \
    --out "$out/recovery_means.json" > "$out/recovery_means.log" 2>&1
  grep RESULT "$out/recovery_means.log" | sed 's/^\[[0-9:]*\] //'
done
echo "RECMEANS DONE ($(date +%H:%M))"
