#!/usr/bin/env bash
# Low-loss guard forensics (Astra, 2026-09-12 23:00): the guard restored best
# params at epochs 34/36/38/43 on losses < 1e-5. Under item-14 labels the pool
# is fully fittable, so "loss < 1e-5 == collapse" (GOTCHA #99) may be stale.
# Compare the REJECTED snapshots (their own weights: --snapshot-all exports
# BEFORE the guard) against the accepted neighbours by numerical and
# behavioral evidence, none of it the trainer's own loss.
set -uo pipefail
cd "$(dirname "$0")/../.."
export XLA_TARGET_EVAL=cuda12 EXPHIL_GPU_MEMORY_FRACTION=0.25
O=eval_runs/0912_g26_phase; N=checkpoints/ms_g26a
echo "=== 1. params/logits finiteness + independent BCE ($(date +%H:%M))"
mix run scripts/probe_snapshot_health.exs --policies "${N}_ep33.bin,${N}_ep34.bin,${N}_ep35.bin,${N}_ep36.bin,${N}_ep37.bin,${N}_ep38.bin,${N}_ep42.bin,${N}_ep43.bin" --delay-id 4 --offset 4 --out $O/guard_health.json 2>&1 | grep -aE "ms_g26a_ep|wrote|error"
echo "=== 2. coverage maps: accepted ep33 vs rejected ep34/ep38 ($(date +%H:%M))"
for ep in 33 34 38; do
  mix run scripts/probe_ms_coverage_map.exs --policy ${N}_ep$ep.bin --delay-id 4 --temperature 1.0 --out $O/map_guard_ep$ep.json > $O/map_guard_ep$ep.log 2>&1
  echo "ep$ep: $(grep -aE 'baseline p\(correct\)' $O/map_guard_ep$ep.log | sed 's/^.*offset: //')"
  grep -aE "^\[.*\] \| (baseline|opp shield|stage mirrored)" $O/map_guard_ep$ep.log | sed "s/^\[[0-9:]*\] /ep$ep /"
done
echo "=== 3. stand floor k=4, rejected vs accepted ($(date +%H:%M))"
GATE_REACTION=4 EPOCHS="33 34 36 38" bash scripts/gate_sweep.sh $N $O/stand_guard 2>&1 | grep -aE "^ep[0-9]+:|ARGMAX"
echo "=== 4. CPU gate k=4 ($(date +%H:%M))"
bash eval_runs/0911_g24_dagger/gate_cpu.sh $N $O/cpu_guard "33 34 38" 2>&1 | tail -6
echo "=== DONE $(date +%H:%M)"
