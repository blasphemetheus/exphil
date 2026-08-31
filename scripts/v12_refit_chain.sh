#!/usr/bin/env bash
# v1.2 refit — head-only refit on v1.1-AR's FROZEN trunk (the L_cond
# restoration, 0831_coincidence_probe verdict): the unfreeze atrophied the
# AR conditioning wire (2.67 -> 1.14); a frozen-trunk fit forces it back
# (8a's mechanism), on top of the best trunk we have.
#
#   AR arm   = v1.2-ARrefit  (the fix under test)
#   IND arm  = v1.2-INDrefit (symmetric control, same capture)
#
# Then verifies in the same unit: coincidence probe (did L_cond come back?)
# and arhead_score (did live recovery keep v1.1's gains?).
#
#   systemd-run --user --unit=v12-refit --working-directory=$PWD --collect \
#     -p StandardOutput=append:$PWD/logs/v12_refit.log \
#     -p StandardError=append:$PWD/logs/v12_refit.log \
#     devenv shell -- bash scripts/v12_refit_chain.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

SRC=checkpoints/fox_gen_v1.1_AR_20260831_080100_policy.bin
AR_OUT=checkpoints/fox_gen_v1.2_ARrefit_policy.bin
IND_OUT=checkpoints/fox_gen_v1.2_INDrefit_policy.bin
PROBE_OUT=eval_runs/0831_v12_refit_probe/RESULTS.md
SCORE_OUT=eval_runs/0831_v12_refit_score
mkdir -p logs

# GOTCHA #106: state the budget. Capture ~19 GB to cache/ar_head; refuse
# under 60 GB free.
if [ "$(df --output=avail / | tail -1)" -lt 60000000 ]; then
  echo "!!! under 60 GB free on / — not starting (GOTCHA #106)"; exit 1
fi

while pgrep -x beam.smp >/dev/null 2>&1; do
  echo "[$(date -Is)] waiting for a live beam (critic-ar?) to finish..."
  sleep 120
done

echo "=== v1.2 refit start $(date -Is) src=$SRC"

mix run scripts/train_ar_head.exs \
  --policy "$SRC" \
  --replays 'replays/erickfm_ranked/FOX/extracted/*.slp' \
  --char-id 2 --limit-files 1000 --head both \
  --out-ar "$AR_OUT" --out-ind "$IND_OUT" || { echo "!!! refit FAILED"; exit 1; }

echo "=== refit done $(date -Is); probing L_cond"

mix run scripts/coincidence_probe.exs \
  --set v12_ARrefit="$AR_OUT" --set v12_INDrefit="$IND_OUT" --set v11_AR="$SRC" \
  --replays 'replays/erickfm_ranked/FOX/extracted/*.slp' \
  --limit-files 8 --out "$PROBE_OUT" || echo "!!! probe FAILED"

echo "=== probe done $(date -Is); live scoring"

OUT="$SCORE_OUT" AR_POLICY="$AR_OUT" IND_POLICY="$IND_OUT" \
  bash scripts/arhead_score.sh || echo "!!! score FAILED"

echo "=== V1.2 REFIT CHAIN DONE $(date -Is) — read: probe L_cond (want ~2.5+ on ARrefit, ~1 on INDrefit and v11_AR) BEFORE the score; live gate: keep v1.1's recovery gains (died% <= v1.1-AR's 28 +/- floor) with the restored routes. Then Bradley's look."
