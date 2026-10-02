#!/usr/bin/env bash
# Interp readout for the coherence program (2026-10-02): Q1/Q2/Q5 probes on
# base vs channel vs change-weighted testbed models, then recovery-drill
# traces + prev-action counterfactuals (Q3). Results: eval_runs/1002_interp/.
set -uo pipefail
out=eval_runs/1002_interp; mkdir -p "$out"
run="mix run --no-compile --no-deps-check"
export EDIFICE_LOCAL_NX=1 EDIFICE_FUSED_CUSTOM_CALL=1 EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.45 EXPHIL_EXLA_PRECISION=highest

for name in base prev_q prev_q_tw4; do
  p=checkpoints/coh_$name/model_best_policy.bin
  echo "== probe $name"
  $run scripts/interp_coherence_probe.exs --policy "$p" --label "$name" --games 16 --stride 4 --out "$out/probe_$name.json" > "$out/probe_$name.log" 2>&1 \
    || echo "FAILED probe $name"
  grep RESULT "$out/probe_$name.log" | sed 's/^\[[0-9:]*\] //'
done

for name in base prev_q; do
  p=checkpoints/coh_$name/model_best_policy.bin
  echo "== trace $name"
  $run scripts/recovery_drill.exs --policy "$p" --label "${name}_trace" --trace "$out/trace_$name.json" > "$out/trace_$name.log" 2>&1 || echo "FAILED trace $name"
  grep RESULT "$out/trace_$name.log" | sed 's/^\[[0-9:]*\] //'
  for ov in up:3 upb:3 jump:3; do
    tag=${ov/:/}
    echo "== counterfactual $name $ov"
    $run scripts/recovery_drill.exs --policy "$p" --label "${name}_cf_$tag" --warm-override "$ov" --trace "$out/trace_${name}_cf_$tag.json" > "$out/cf_${name}_$tag.log" 2>&1 || echo "FAILED cf $name $ov"
    grep RESULT "$out/cf_${name}_$tag.log" | sed 's/^\[[0-9:]*\] //'
  done
done
echo "INTERP QUEUE DONE"
