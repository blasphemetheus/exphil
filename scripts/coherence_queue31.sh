#!/usr/bin/env bash
# Queue 31 (2026-10-08 13:10, after queue 30) — the AIM probe (Q9) on the
# three 3-ep arms: is the missing Firefox aim (stick UP once the jump is
# spent: loop onset 3.4 % vs expert 12.3 % per 3 f) missing teacher-forced
# too (never learned → weighting/label) or only in the loop (distribution
# shift → the sim-DAgger case, Bradley's ok)? No training; ~10 min.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
while systemctl --user is-active --quiet exphil-queue30; do sleep 60; done
echo "== queue 30 finished ($(date +%H:%M))"
# another session's beam may be alive (13:20: a scratch probe); wait for it
# rather than refuse — this queue only reads checkpoints, no training
while pgrep -f '[b]eam.smp' > /dev/null; do sleep 60; done
echo "== beam free ($(date +%H:%M))"
for a in dur8e_e3 on30_e3 on30u_e3; do
  arm=evt2ctx_ck8_off3_dur8e_$a
  echo "== aim probe $arm ($(date +%H:%M))"
  mix run --no-compile scripts/recovery_aim_probe.exs --policy checkpoints/coh_$arm/model_policy.bin --label $arm \
    --out eval_runs/1001_queue/$arm/aim_probe.json 2>&1 | grep -E "RESULT|error|Error|\*\*" | cut -c1-600
done
echo "QUEUE 31 DONE ($(date +%H:%M))"
