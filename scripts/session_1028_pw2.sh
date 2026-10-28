#!/usr/bin/env bash
# 10-28 early: press-window dose and slice follow-up (the 10-27 session saw the first commitment
# signal at dose 10 on the 1500-file slice). Arms: pw20, pw40 at 1500 files; base3k, pw10_3k, pw20_3k
# at the 3,000-file slice (the standard testbed). Walk-off origins + agreement per arm. Fused OFF.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1 EDIFICE_FUSED_CUSTOM_CALL=0
R=eval_runs/1027_session/REPORT.md
log() { echo "$*" | tee -a "$R"; }
waitbeam() { while pgrep -f '[b-b]eam.smp' >/dev/null; do sleep 60; done; }
run="mix run --no-compile --no-deps-check"
base=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize --button-events --stick-events --event-context --chunk-horizon 8 --offstage-weight 3 --stick-duration 8 --onset-weight 5 --action-oversample 1.0 --neutral-weight 1.0)
agree() {
  local n=$1 d=eval_runs/1001_queue/$1
  $run scripts/agreement_eval.exs --policy "checkpoints/coh_$n/model_best_policy.bin" --split "checkpoints/coh_$n/split.json" --games 3 --stride 2 --out "$d/agreement.json" > "$d/agreement.log" 2>&1
  grep -E "RESULT (CHANGE|frames with|press calibration (jump|attack|special)|false press)" "$d/agreement.log" | sed 's/^\[[0-9:]*\] //' | cut -c1-330 | tee -a "$R"
  [ -f "$d/agreement.json" ] && node -e 'const j=JSON.parse(require("fs").readFileSync(process.argv[1]));const s=j.by_action_change_frames.find(a=>a.action==="STANDING");const e=j.press_edges;console.log("RESULT STANDING change frames: n "+(s?s.n:0)+" change recall "+(s?s.change_recall:"?")+"%  | press edges same-family: "+e.map(x=>x.family+" "+x.model_same_press+"%").join(" "))' "$d/agreement.json" | tee -a "$R"
}
arm() {  # NAME MAX_FILES flags...
  local name=$1 mf=$2; shift 2
  log ""; log "### arm pw_$name (files $mf) — $* ($(date +%H:%M))"
  waitbeam
  rm -rf "checkpoints/coh_pw_$name" "eval_runs/1001_queue/pw_$name"
  MAX_FILES=$mf EVALS="coherence closed_loop recovery" scripts/pq_experiment.sh "pw_$name" "${base[@]}" "$@" 2>&1 | grep -E "RESULT|FAILED|val_loss" | sed 's/^\[[0-9:]*\] //' | cut -c1-300 | tee -a "$R"
  if [ -f "checkpoints/coh_pw_$name/model_best_policy.bin" ]; then
    agree "pw_$name"
    EXLA_TARGET=cuda EXPHIL_EXLA_PRECISION=highest EXPHIL_GPU_MEMORY_FRACTION=0.45 $run scripts/walkoff_origins.exs --policy "checkpoints/coh_pw_$name/model_best_policy.bin" --label "pw_$name" --envs 32 --frames 1800 --out "eval_runs/walkoff/pw_$name.json" 2>&1 | grep -E "RESULT .*(origins|jump_off|special_off)" | sed 's/^\[[0-9:]*\] //' | cut -c1-250 | tee -a "$R"
  fi
}
log ""; log "# 10-28 follow-up: press-window dose + slice ($(date '+%F %H:%M'))"
arm pw20 1500 --press-window-weight 20
arm pw40 1500 --press-window-weight 40
arm base3k 3000
arm pw10_3k 3000 --press-window-weight 10
arm pw20_3k 3000 --press-window-weight 20
log ""; log "SESSION 2 DONE ($(date +%H:%M)) — table: node scripts/pq_digest.js pw_base pw_base906 pw_pw2 pw_pw5 pw_pw10 pw_pw20 pw_pw40 pw_base3k pw_pw10_3k pw_pw20_3k"
