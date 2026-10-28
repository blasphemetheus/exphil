#!/usr/bin/env bash
# 10-27 night session (first GPU use on Axon 0.9 / Polaris 0.2):
#  1. fused-handler check: one closed-loop readout of the candidate with EDIFICE_FUSED_CUSTOM_CALL=1;
#     if it crashes on crypto again, every later readout runs with the handlers off.
#  2. press-window arms on the testbed, ALL retrained on the new build (base, base906, pw2, pw5, pw10;
#     MAX_FILES 1500, agreement on 3 games) + walk-off origins per arm.
#  3. master_ep6 + null9o closed-loop / recovery / recovery_means readouts (the ones the crash ate).
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
R=eval_runs/1027_session/REPORT.md; mkdir -p eval_runs/1027_session eval_runs/walkoff
log() { echo "$*" | tee -a "$R"; }
waitbeam() { while pgrep -f '[b-b]eam.smp' >/dev/null; do sleep 60; done; }
run="mix run --no-compile --no-deps-check"
log ""; log "# 10-27 session — $(date '+%F %H:%M') (Axon 0.9 / Polaris 0.2 build)"
log "## 1. fused handler check"
waitbeam
if EDIFICE_FUSED_CUSTOM_CALL=1 EXLA_TARGET=cuda EXPHIL_EXLA_PRECISION=highest EXPHIL_GPU_MEMORY_FRACTION=0.45 $run scripts/sim_closed_loop.exs --policy checkpoints/coh_fox_mamba_v3s906_ep2/model_best_policy.bin --label fusedcheck --envs 32 --frames 600 --out eval_runs/1027_session/fusedcheck.json > eval_runs/1027_session/fusedcheck.log 2>&1 && grep -q "RESULT fusedcheck vs idle" eval_runs/1027_session/fusedcheck.log; then
  FUSED=1; log "fused handlers OK on this build"
else
  FUSED=0; log "fused handlers still crash (see fusedcheck.log) — readouts run with EDIFICE_FUSED_CUSTOM_CALL=0"; grep -m1 "crypto\|\*\*" eval_runs/1027_session/fusedcheck.log | cut -c1-160 | tee -a "$R"
fi
export EDIFICE_FUSED_CUSTOM_CALL=$FUSED
base=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize --button-events --stick-events --event-context --chunk-horizon 8 --offstage-weight 3 --stick-duration 8 --onset-weight 5 --action-oversample 1.0 --neutral-weight 1.0)
agree() {
  local n=$1 d=eval_runs/1001_queue/$1
  $run scripts/agreement_eval.exs --policy "checkpoints/coh_$n/model_best_policy.bin" --split "checkpoints/coh_$n/split.json" --games 3 --stride 2 --out "$d/agreement.json" > "$d/agreement.log" 2>&1
  grep -E "RESULT (CHANGE|frames with|press calibration (jump|attack|special)|false press)" "$d/agreement.log" | sed 's/^\[[0-9:]*\] //' | cut -c1-330 | tee -a "$R"
  [ -f "$d/agreement.json" ] && node -e 'const j=JSON.parse(require("fs").readFileSync(process.argv[1]));const s=j.by_action_change_frames.find(a=>a.action==="STANDING");const e=j.press_edges;console.log("RESULT STANDING change frames: n "+(s?s.n:0)+" change recall "+(s?s.change_recall:"?")+"%  | press edges same-family: "+e.map(x=>x.family+" "+x.model_same_press+"%").join(" "))' "$d/agreement.json" | tee -a "$R"
}
arm() {
  local name=$1; shift
  log ""; log "### arm pw_$name — $* ($(date +%H:%M))"
  waitbeam
  rm -rf "checkpoints/coh_pw_$name" "eval_runs/1001_queue/pw_$name"     # retrain on the new build
  MAX_FILES=1500 EVALS="coherence closed_loop recovery" scripts/pq_experiment.sh "pw_$name" "${base[@]}" "$@" 2>&1 | grep -E "RESULT|FAILED|val_loss" | sed 's/^\[[0-9:]*\] //' | cut -c1-300 | tee -a "$R"
  if [ -f "checkpoints/coh_pw_$name/model_best_policy.bin" ]; then
    agree "pw_$name"
    EXLA_TARGET=cuda EXPHIL_EXLA_PRECISION=highest EXPHIL_GPU_MEMORY_FRACTION=0.45 $run scripts/walkoff_origins.exs --policy "checkpoints/coh_pw_$name/model_best_policy.bin" --label "pw_$name" --envs 32 --frames 1800 --out "eval_runs/walkoff/pw_$name.json" 2>&1 | grep -E "RESULT .*(origins|walk_off|jump_off|aerial_drift|special_off)" | sed 's/^\[[0-9:]*\] //' | cut -c1-250 | tee -a "$R"
  fi
}
log "## 2. press-window arms"
arm base
SEED=906 arm base906
arm pw2  --press-window-weight 2
arm pw5  --press-window-weight 5
arm pw10 --press-window-weight 10
log "## 3. master_ep6 / null9o readouts ($(date +%H:%M))"
for name in fox_mamba_master_ep6 fox_mamba_v3s906_ep3_null9o; do
  log "### $name"; waitbeam
  EVALS="closed_loop recovery recovery_means" scripts/coherence_experiment.sh "$name" 2>&1 | grep -E "RESULT|FAILED" | sed 's/^\[[0-9:]*\] //' | cut -c1-300 | tee -a "$R"
  node scripts/recovery_curve_row.js "$name" 2>/dev/null | tee -a "$R" eval_runs/1011_full/curve.txt
done
log ""; log "SESSION DONE ($(date +%H:%M)) — table: node scripts/pq_digest.js pw_base pw_base906 pw_pw2 pw_pw5 pw_pw10"
