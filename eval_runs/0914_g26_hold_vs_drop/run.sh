#!/usr/bin/env bash
# Equal-budget HOLD vs DROP comparison on the g26 pool (RECOVERY_LABEL_
# CONFIRMATION "next", HANDOFF_2026-09-13d decision 2). Everything is the
# g26a recipe (eval_runs/0912_g26_phase/run_g26.sh, arm a: g24a pool +
# snippets, independent head + SS, rungs 3/4/5, queue 4, physical ids) except
# ONE flag: --off-loop-labels hold (the g26a rule: off-loop expert frames get
# the recovery input held at every k — measured wrong 18/21 at shift 4) vs
# drop (the 09-13 default: the expert abstains off the loop; recovery
# supervision at a delayed rung must be a recording). NOTE the drop arm has
# NO recorded recoveries at rungs 3/4/5 (none exist yet): it tests "wrong
# recovery labels" vs "no recovery labels", not the full fixed recipe.
# Budget: 40 epochs each (~1m50s/epoch), snapshot-all. Readout per arm:
# coverage map at delay-id 4 (final), stand floor at reaction 4 (ep 30/40),
# CPU gate at reaction 4 (ep 30/40). g26a's numbers: stand floor 67/min c4
# (worst); ep57 296/min.
set -uo pipefail
cd "$(dirname "$0")/../.."
if pgrep -x beam.smp >/dev/null; then echo "A BEAM is live; wait." >&2; exit 1; fi
OUT=eval_runs/0914_g26_hold_vs_drop
mkdir -p "$OUT"
P=$OUT/progress.log
say() { echo "[$(date +%T)] $*" | tee -a "$P"; }
export EXLA_TARGET=cuda
SNIP=eval_runs/0910_snippets_human_causal_d1/snippets.frames
ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp,eval_runs/0802_d2pool/r*.slp,eval_runs/0911_g23a_live/2026-09-Mainline/*.slp,eval_runs/0911_g23a_live/async_d3_cpu/r*.slp,eval_runs/0911_g23a_live/cpu_rollouts/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'
EPOCHS=${EPOCHS:-40}
git rev-parse HEAD > "$OUT/git_head.txt"

for ARM in ${ARMS:-hold drop}; do
  NAME=ms_g27$ARM
  if [[ -f checkpoints/$NAME.bin ]]; then say "$ARM: checkpoints/$NAME.bin exists, skipping train"; continue; fi
  say "=== $ARM: pool label audit at shifts 3,4,5 (--off-loop $ARM)"
  mix run scripts/audit_ms_pool_labels.exs --rollouts "$ROLL" --openers "$OPEN" --snippets "$SNIP" \
    --shifts "3,4,5" --off-loop $ARM > "$OUT/audit_$ARM.log" 2>&1
  say "  audit exit $? ; $(grep -cE 'CONFLICT' $OUT/audit_$ARM.log) CONFLICT rows; dropped: $(grep -E 'frames labeled' $OUT/audit_$ARM.log | tr '\n' ';' | cut -c1-400)"
  say "=== $ARM: TRAIN $EPOCHS epochs -> checkpoints/$NAME.bin"
  EXPHIL_GPU_MEMORY_FRACTION=0.75 mix run scripts/dagger_drill.exs \
    --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
    --rollouts "$ROLL" --opening-replays "$OPEN" --snippet-frames "$SNIP" \
    --opp-scramble-frames 12000 \
    --max-epochs "$EPOCHS" --prev-action-dropout 0.6 --transition-weight 2.0 \
    --action-delay 1 --multi-delay "3,4,5" --queue-depth 4 --with-delay-id \
    --scheduled-sampling 0.5 --ss-ramp 10 --awbc \
    --head independent --clean-loss --action-frame-buckets 24 \
    --off-loop-labels $ARM \
    --lr 2.0e-4 --target-loss 1.0e-9 --snapshot-all \
    --out "checkpoints/$NAME.bin" > "$OUT/train_$ARM.log" 2>&1
  say "  train exit $?; $(grep -oE 'epoch [0-9]+/[0-9]+: loss=[0-9.e-]+' $OUT/train_$ARM.log | tail -1)"
done

for ARM in ${ARMS:-hold drop}; do
  N=checkpoints/ms_g27$ARM
  [[ -f $N.bin ]] || { say "$ARM: no checkpoint, skipping readout"; continue; }
  say "=== $ARM: coverage map (delay-id 4, final)"
  EXPHIL_GPU_MEMORY_FRACTION=0.25 mix run scripts/probe_ms_coverage_map.exs --policy $N.bin --delay-id 4 --temperature 1.0 \
    --out "$OUT/map_$ARM.json" > "$OUT/map_$ARM.log" 2>&1
  grep -aE "^\[.*\] \| (baseline|mirror)" "$OUT/map_$ARM.log" | sed 's/^\[[0-9:]*\] //' | head -3 | tee -a "$P"
  say "=== $ARM: stand floor at reaction 4 (ep 30 $EPOCHS)"
  GATE_REACTION=4 EPOCHS="30 $EPOCHS" bash scripts/gate_sweep.sh $N "$OUT/stand_$ARM" > "$OUT/stand_$ARM.log" 2>&1
  tail -6 "$OUT/stand_$ARM.log" | tee -a "$P"
  say "=== $ARM: CPU gate at reaction 4 (ep 30 $EPOCHS)"
  bash eval_runs/0911_g24_dagger/gate_cpu.sh $N "$OUT/cpu_$ARM" "30 $EPOCHS" > "$OUT/cpu_$ARM.log" 2>&1
  tail -4 "$OUT/cpu_$ARM.log" | tee -a "$P"
done
say "done"
