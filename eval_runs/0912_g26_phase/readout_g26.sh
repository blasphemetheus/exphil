#!/usr/bin/env bash
# g25 readout, in the prereg order: (1) coverage map (offline) on the final
# and a mid snapshot vs ep57's map; (2) stand floor at k=4; (3) CPU gate at k=4.
set -uo pipefail
cd "$(dirname "$0")/../.."
export XLA_TARGET_EVAL=cuda12 EXPHIL_GPU_MEMORY_FRACTION=0.25
OUT=eval_runs/0912_g26_phase; N=checkpoints/ms_g26a
[ -f "$N.bin" ] || exit 1
sleep 20
echo "=== MAP final ($(date +%H:%M))"
mix run scripts/probe_ms_coverage_map.exs --policy $N.bin --delay-id 4 --temperature 1.0 --out $OUT/map_final.json > $OUT/map_final.log 2>&1
grep -aE "^\[.*\] \| (baseline|opp_action|opp_dx|mirror)" $OUT/map_final.log | sed 's/^\[[0-9:]*\] //' | head -30
for ep in 30 45; do
  [ -f "${N}_ep$ep.bin" ] || continue
  echo "=== MAP ep$ep"
  mix run scripts/probe_ms_coverage_map.exs --policy ${N}_ep$ep.bin --delay-id 4 --temperature 1.0 --out $OUT/map_ep$ep.json > $OUT/map_ep$ep.log 2>&1
  grep -aE "^\[.*\] \| (baseline|mirror)" $OUT/map_ep$ep.log | sed 's/^\[[0-9:]*\] //' | head -3
done
echo "=== STAND k=4 (gate_sweep, sampled) ($(date +%H:%M))"
GATE_REACTION=4 EPOCHS="40 50 60" bash scripts/gate_sweep.sh $N $OUT/stand_k4 2>&1 | tail -8
echo "=== CPU k=4 ($(date +%H:%M))"
bash eval_runs/0911_g24_dagger/gate_cpu.sh $N $OUT/cpu_k4 "45 60" 2>&1 | tail -6
echo "=== DONE $(date +%H:%M)"
