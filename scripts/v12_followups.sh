#!/usr/bin/env bash
# Post-refit follow-ups (HANDOFF_2026-08-31 tasks 2+3), sequential:
#   1. critic ladder on v1.2-ARrefit (coherent candidates with a REAL wire)
#   2. G3b dynamics spike (first run — may need Axon.Loop API debugging)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs eval_runs/0831_critic_refit cache/critic

while pgrep -x beam.smp >/dev/null 2>&1; do sleep 60; done

POLICY=checkpoints/fox_gen_v1.2_ARrefit_policy.bin
DATA=cache/critic/v12arrefit_erickfm40_k16.nx
CRIT=eval_runs/0831_critic_refit

echo "=== critic ladder on ARrefit start $(date -Is)"
mix run scripts/critic_extract.exs --policy "$POLICY" \
  --replays 'replays/erickfm_ranked/FOX/extracted/*.slp' \
  --limit-files 40 --k 16 --stride 2 --out "$DATA" \
&& mix run scripts/critic_train.exs --data "$DATA" --out "$CRIT/critic.nx" --report "$CRIT/train.md" \
&& mix run scripts/interp_bestofn.exs --policy "$POLICY" --critic "$CRIT/critic.nx" \
     --replays 'replays/erickfm_ranked/FOX/extracted/*.slp' --n 16 --limit-files 20 \
     --out "$CRIT/bestofn_indist.md" \
&& mix run scripts/interp_bestofn.exs --policy "$POLICY" --critic "$CRIT/critic.nx" \
     --replays 'replays/fox_il_v1/*.slp' --n 16 --limit-files 20 \
     --out "$CRIT/bestofn_fox_il_v1.md" \
|| echo "!!! critic ladder FAILED"
echo "=== critic ladder done $(date -Is)"

echo "=== dynamics spike start $(date -Is)"
mix run scripts/dynamics_spike.exs \
  --policy checkpoints/fox_gen_v1.1_AR_20260831_080100_policy.bin \
  --replays 'replays/erickfm_ranked/FOX/extracted/*.slp' \
  --limit-files 60 --epochs 2 \
  --out eval_runs/0831_dynamics_spike/RESULTS.md \
|| echo "!!! dynamics spike FAILED (expected candidate: Axon.Loop.run API shape — see HANDOFF task 3)"
echo "=== V12 FOLLOWUPS DONE $(date -Is) — ladder read: selector-over-mode margin vs v1.1-AR's +2.7/+1.4 (wire-live bar >=5); spike gate: R2>0.9 AND cos@10>0.8"
