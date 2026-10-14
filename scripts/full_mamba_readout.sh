#!/usr/bin/env bash
# Re-read an already-trained full-run epoch under a NEW name (2026-10-14):
# the readout half of scripts/full_mamba_epoch.sh, no training. Used to
# measure the roll noise of the curve row — the same policy, a second
# closed-loop roll, a second row in eval_runs/1011_full/curve.txt.
#
#   scripts/launch_unit.sh exphil-fox-mamba-v3s906-ep2r \
#     'scripts/full_mamba_readout.sh fox_mamba_v3s906_ep2r fox_mamba_v3s906_ep2'
#
# checkpoints/coh_NEW/ is a symlink view of checkpoints/coh_SRC/ (policy,
# split, config), so coherence_experiment.sh skips training and the node
# readers index the new name. Same guards as the epoch script.
set -uo pipefail
cd /home/blewf/git/exphil
name=${1:?usage: full_mamba_readout.sh NEW_NAME SRC_NAME}
src=${2:?usage: full_mamba_readout.sh NEW_NAME SRC_NAME}
export EDIFICE_LOCAL_NX=1 EDIFICE_FUSED_CUSTOM_CALL=1 EXLA_TARGET=cuda EXPHIL_EXLA_PRECISION=highest
ckpt=checkpoints/coh_$name
srcdir=$(realpath "checkpoints/coh_$src")
[ -f "$srcdir/model_best_policy.bin" ] || { echo "NO_SRC_POLICY ($srcdir)"; exit 1; }
mkdir -p "$ckpt" "eval_runs/1001_queue/$name"
for f in model_best_policy.bin split.json model_config.json completed.json; do
  [ -e "$ckpt/$f" ] || ln -s "$srcdir/$f" "$ckpt/$f"
done

echo "== $name: re-readout of $src ($(date +%H:%M))"
while pgrep -f '[b]eam.smp' > /dev/null; do echo "beam alive, waiting ($(date +%H:%M))"; sleep 60; done
newer=$(find ../nx/nx/lib ../nx/exla/lib ../nx/exla/c_src -newer ../nx/exla/cache/libexla.so \( -name '*.ex' -o -name '*.exs' -o -name '*.cc' -o -name '*.h' \) 2>/dev/null | head -3)
[ -z "$newer" ] || { echo "NX_EDIT_IN_PROGRESS: $newer"; exit 1; }

echo "== readout ($(date +%H:%M))"
EVALS="coherence closed_loop recovery fidelity recovery_means" scripts/coherence_experiment.sh "$name" 2>&1 | grep -E "RESULT|DONE|FAILED" | cut -c1-300
ctl=(evt2ctx_ck8_off3_dur8e_on5u_dag7w_x1_e3 evt2ctx_ck8_off3_dur8e_on5u_dag6w_x1_e3_s906)
node scripts/recovery_death_shape.js "$name"
node scripts/recovery_sideb_trips.js "${ctl[@]}" "$src" "$name"
node scripts/recovery_jump_spend.js "${ctl[@]}" "$src" "$name"
node scripts/recovery_firefox_angle.js "${ctl[@]}" "$src" "$name"
node scripts/recovery_aim_vs_press.js "${ctl[@]}" "$src" "$name"
curve=eval_runs/1011_full/curve.txt
# val column: recovery_curve_row.js reads train.log in the eval dir
[ -e "eval_runs/1001_queue/$name/train.log" ] || ln -s "$(realpath "eval_runs/1001_queue/$src/train.log")" "eval_runs/1001_queue/$name/train.log"
node scripts/recovery_curve_row.js "$name" | tee -a "$curve"
echo "== re-readout of $src (same policy, second roll)" >> "$curve"
echo "READOUT $name DONE ($(date +%H:%M))"
