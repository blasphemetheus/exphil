#!/usr/bin/env bash
# v1.3 chain part 2 (09-01 evening) — refit + evals on CLEAN captures.
# Part 1 (unit v13-portfix) trained both arms on the repaired streaming
# loader and was deliberately stopped at the IND-end seam (Bradley's
# stop-and-fix directive) before the refit stage, which would have run on
# E1c-contaminated captures (capture_replay embedded swapped perspective on
# non-port-1 files). The E1c fix is now in (capture remap + :r2 capture
# cache keys + own_port stamp/assert); this runs the remaining stages.
#
#   systemd-run --user --unit=v13-part2 --working-directory=$PWD --collect \
#     -p StandardOutput=append:$PWD/logs/v13_part2.log \
#     -p StandardError=append:$PWD/logs/v13_part2.log \
#     devenv shell -- bash scripts/v13_part2.sh
#
# DO NOT edit lib/*.ex while this runs (multi-stage, L7).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

AR_SRC=checkpoints/fox_gen_v1.3_AR_20260901_194518_policy.bin
AR_OUT=checkpoints/fox_gen_v1.3_ARrefit_policy.bin
IND_OUT=checkpoints/fox_gen_v1.3_INDrefit_policy.bin
OUT=eval_runs/0901_v13_portfix
mkdir -p "$OUT" logs

if [ ! -f "$AR_SRC" ]; then echo "!!! missing $AR_SRC"; exit 1; fi

# GOTCHA #106: clean capture rebuilds non-p1 entries under :r2 keys (~44%
# of ~19 GB) on top of part 1's usage. Refuse under 50 GB free.
if [ "$(df --output=avail / | tail -1)" -lt 50000000 ]; then
  echo "!!! under 50 GB free on / — not starting (GOTCHA #106)"; exit 1
fi

while pgrep -x beam.smp >/dev/null 2>&1; do
  echo "[$(date -Is)] waiting for a live beam to finish..."
  sleep 60
done

echo "=== v1.3 part2 start $(date -Is) src=$AR_SRC (E1c-clean captures)" | tee -a "$OUT/TABLE.md"

mix run scripts/train_ar_head.exs \
  --policy "$AR_SRC" \
  --replays 'replays/erickfm_ranked/FOX/extracted/*.slp' \
  --char-id 2 --limit-files 1000 --head both \
  --out-ar "$AR_OUT" --out-ind "$IND_OUT" || { echo "!!! refit FAILED" | tee -a "$OUT/TABLE.md"; exit 1; }

echo "=== refit done $(date -Is); coincidence probe" | tee -a "$OUT/TABLE.md"
mix run scripts/coincidence_probe.exs \
  --set v13_ARrefit="$AR_OUT" --set v13_INDrefit="$IND_OUT" --set v13_AR="$AR_SRC" \
  --replays 'replays/erickfm_ranked/FOX/extracted/*.slp' \
  --limit-files 8 --out "$OUT/coincidence_probe.md" || echo "!!! probe FAILED" | tee -a "$OUT/TABLE.md"

while pgrep -x beam.smp >/dev/null 2>&1; do sleep 30; done
echo "=== arhead_score $(date -Is)" | tee -a "$OUT/TABLE.md"
OUT="$OUT/score" AR_POLICY="$AR_OUT" IND_POLICY="$IND_OUT" \
  bash scripts/arhead_score.sh || echo "!!! score FAILED" | tee -a "$OUT/TABLE.md"

while pgrep -x beam.smp >/dev/null 2>&1; do sleep 30; done
echo "=== position probe sweep $(date -Is)" | tee -a "$OUT/TABLE.md"
mix run scripts/probe_position_dependence.exs \
  --set v13_ARrefit="$AR_OUT" --set v13_INDrefit="$IND_OUT" \
  --replays 'eval_runs/0831_livelook_v12ar/2026-08-Mainline/*.slp' \
  --player-port 1 --situation neutral --sweep "15,25,40,60,90,130" \
  --out "$OUT/position_sweep.md" || echo "!!! position probe FAILED" | tee -a "$OUT/TABLE.md"

while pgrep -x beam.smp >/dev/null 2>&1; do sleep 30; done
echo "=== action-sensitivity rerun (fixed embed-space actions) $(date -Is)" | tee -a "$OUT/TABLE.md"
mix run scripts/dynamics_action_sensitivity.exs \
  --dynamics checkpoints/dynamics_fox_v11AR.bin \
  --policy "$AR_OUT" \
  --replay 'replays/erickfm_ranked/FOX/extracted/*.slp' --k 10 \
  --out "$OUT/action_sensitivity_fixed.md" || echo "!!! sensitivity FAILED" | tee -a "$OUT/TABLE.md"

echo "=== V1.3 PART2 DONE $(date -Is) — read: TABLE.md, coincidence_probe.md (L_cond ARrefit ~2.5+), score/ (died% ~28 class), position_sweep.md (approach_delta vs v1.2's -0.20..-0.26 = pre-registered PRIMARY), action_sensitivity_fixed.md" | tee -a "$OUT/TABLE.md"
