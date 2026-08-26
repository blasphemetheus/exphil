#!/usr/bin/env bash
# Offline agreement sweep for fox_gen_v1 epochs (load-tolerant: no live
# frame loop). One fixture replay per epoch; extend FIXTURES for more.
set -uo pipefail
cd "$(dirname "$0")/.."

PREFIX=checkpoints/fox_gen_v1_20260825_210355
OUT=eval_runs/0826_gen_v1_sweep/offline_agreement.txt
mkdir -p "$(dirname "$OUT")"
: > "$OUT"

FIX=$(ls replays/erickfm_ranked/FOX/extracted/*.slp | head -1)
echo "fixture: $FIX" | tee -a "$OUT"

for ep in 1 2 3 4 5 6 7 8 9 10; do
  echo "=== ep${ep}" | tee -a "$OUT"
  mix run scripts/eval_policy_on_fixture.exs \
    --policy "${PREFIX}_ep${ep}.bin" --fixture "$FIX" 2>&1 \
    | grep -aiE "FIXTURE_AGREEMENT|press rates|policy: B|fixture: B" \
    | tee -a "$OUT" || echo "ep${ep} FAILED" | tee -a "$OUT"
done
echo "SWEEP DONE" | tee -a "$OUT"
