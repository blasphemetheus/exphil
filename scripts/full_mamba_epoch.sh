#!/usr/bin/env bash
# One epoch of the full Fox Mamba run + its recovery readout (2026-10-11).
#
#   scripts/launch_unit.sh exphil-fox-mamba-v3-ep1 'scripts/full_mamba_epoch.sh 1'
#   LR=0.0001 scripts/launch_unit.sh exphil-fox-mamba-v3-ep3 'scripts/full_mamba_epoch.sh 3'
#   MIX=data/silent_fall/sim_dagger_expert_m1_split.frames … 'scripts/full_mamba_epoch.sh 2'
#
# Recipe is FROZEN in eval_runs/1011_full/recipe.sh (sourced; never edited
# mid-run). Epoch N trains ONE epoch resumed from epoch N-1's model.axon
# (weights + optimizer; the fox_mamba_v1 pattern, HANDOFF §7g: a resumed
# run finishes the current epoch, the next epoch is a fresh --resume into a
# new --checkpoint dir — stop = systemctl --user stop <unit>, relaunch the
# same N and progress.json skips the trained chunks). Output
# checkpoints/coh_${RUN}_epN/ (model_best_policy.bin + split.json), so the
# readout is the unchanged testbed chain: coherence_experiment.sh skips
# training when model_best_policy.bin exists and runs every closed-loop
# eval into eval_runs/1001_queue/${RUN}_epN/ — the node readers index by
# that name. The readout ends with ONE row appended to
# eval_runs/1011_full/curve.txt (recovery_curve_row.js): val, decided-trip
# return (pass >= 0.75), carried side-B count (<= 5), carried-off died
# (<= 0.60), Firefox -40..-60 row. The curve across epochs is the product:
# death counts falling with val = scale is the lever; plateau while val
# falls = the tail is not in the loss.
#
# Guards (the 10-06/07 night and GOTCHA #136): no other beam; no nx source
# newer than libexla.so; _build/dev not older than lib/ (this runs
# --no-compile). ~8.5 h train (v1's epoch) + ~40 min readout.
set -uo pipefail
cd /home/blewf/git/exphil
N=${1:?usage: full_mamba_epoch.sh EPOCH_N}
source eval_runs/1011_full/recipe.sh
export EDIFICE_LOCAL_NX=1 EDIFICE_FUSED_CUSTOM_CALL=1 EXLA_TARGET=cuda EXPHIL_EXLA_PRECISION=highest
lr=${LR:-$LR_DEFAULT}; seed=${SEED:-$SEED_DEFAULT}
name=${RUN}_ep${N}
ckpt=checkpoints/coh_$name
out=eval_runs/1001_queue/$name
prev=checkpoints/coh_${RUN}_ep$((N - 1))
mkdir -p "$out" eval_runs/1011_full

echo "== $name: epoch $N, lr $lr, seed $seed, mix '${MIX}' ($(date +%H:%M))"
while pgrep -f '[b]eam.smp' > /dev/null; do echo "beam alive, waiting ($(date +%H:%M))"; sleep 60; done
newer=$(find ../nx/nx/lib ../nx/exla/lib ../nx/exla/c_src -newer ../nx/exla/cache/libexla.so \( -name '*.ex' -o -name '*.exs' -o -name '*.cc' -o -name '*.h' \) 2>/dev/null | head -3)
[ -z "$newer" ] || { echo "NX_EDIT_IN_PROGRESS: $newer"; exit 1; }
newest_beam=$(ls -t _build/dev/lib/exphil/ebin/*.beam 2>/dev/null | head -1)
stale=$(find lib -name '*.ex' -newer "${newest_beam:-/nonexistent}" 2>/dev/null | head -3)
[ -z "$stale" ] || { echo "STALE_BUILD (lib newer than _build/dev; run EDIFICE_LOCAL_NX=1 devenv shell -- mix compile): $stale"; exit 1; }
if [ "$N" -gt 1 ]; then
  [ -f "$prev/model.axon" ] || { echo "NO_PREV_EPOCH ($prev/model.axon)"; exit 1; }
  [ -f "$prev/completed.json" ] || { echo "PREV_EPOCH_INCOMPLETE ($prev has no completed.json)"; exit 1; }
fi
mix_args=(); [ -n "$MIX" ] && { [ -f "$MIX" ] || { echo "NO_MIX_SET ($MIX)"; exit 1; }; mix_args=(--mix-frames "$MIX" --mix-oversample 1); }
resume=(); [ "$N" -gt 1 ] && resume=(--resume "$prev/model.axon")
# relaunch of a stopped epoch: progress.json in $ckpt resumes mid-epoch
[ -f "$ckpt/progress.json" ] && [ ! -f "$ckpt/completed.json" ] && resume=(--resume "$ckpt")

if [ ! -f "$ckpt/model_best_policy.bin" ] || [ ! -f "$ckpt/completed.json" ]; then
  echo "== train ($(date +%H:%M)) -> $out/train.log"
  EXPHIL_GPU_MEMORY_FRACTION=0.70 mix run --no-compile --no-deps-check scripts/train_fox_mamba.exs \
    "${model[@]}" "${data[@]}" "${loss[@]}" "${pq[@]}" "${ev[@]}" "${rec0[@]}" "${knob[@]}" \
    --learning-rate "$lr" --epochs 1 --seed "$seed" --name "coh-$name" "${mix_args[@]}" "${resume[@]}" \
    --checkpoint "$ckpt/model.axon" >> "$out/train.log" 2>&1 || { echo "TRAIN_FAILED $name (see $out/train.log)"; exit 1; }
  [ -f "$ckpt/completed.json" ] || { echo "TRAIN_INCOMPLETE $name (no completed.json)"; exit 1; }
fi
grep -o 'val_loss[= ]*[0-9.]*' "$out/train.log" | tail -1

# readout: the testbed's closed-loop chain on this epoch's policy (no
# calibration: its CALIBRATE_ONLY path rebuilds the MinGRU train args)
echo "== readout ($(date +%H:%M))"
EVALS="coherence closed_loop recovery fidelity recovery_means" scripts/coherence_experiment.sh "$name" 2>&1 | grep -E "RESULT|DONE|FAILED" | cut -c1-300
ctl=(evt2ctx_ck8_off3_dur8e_on5u_dag7w_x1_e3 evt2ctx_ck8_off3_dur8e_on5u_dag6w_x1_e3_s906)
prevs=(); for k in $(seq 1 $((N - 1))); do [ -f "eval_runs/1001_queue/${RUN}_ep$k/recovery_means.json" ] && prevs+=("${RUN}_ep$k"); done
node scripts/recovery_death_shape.js "$name"
node scripts/recovery_sideb_trips.js "${ctl[@]}" "${prevs[@]}" "$name"
node scripts/recovery_jump_spend.js "${ctl[@]}" "${prevs[@]}" "$name"
node scripts/recovery_firefox_angle.js "${ctl[@]}" "${prevs[@]}" "$name"
node scripts/recovery_aim_vs_press.js "${ctl[@]}" "${prevs[@]}" "$name"
# the curve
curve=eval_runs/1011_full/curve.txt
[ -s "$curve" ] || { node scripts/recovery_curve_row.js --header > "$curve"; node scripts/recovery_curve_row.js --expert >> "$curve"; node scripts/recovery_curve_row.js "${ctl[@]}" >> "$curve"; }
node scripts/recovery_curve_row.js "$name" | tee -a "$curve"
echo "== lr $lr seed $seed mix '${MIX}'" >> "$curve"
echo "EPOCH $N DONE ($(date +%H:%M))"
