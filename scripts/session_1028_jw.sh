#!/usr/bin/env bash
# 10-28 afternoon: jump-family press window on the 3,000-game testbed (new stack: XLA 0.11, Axon 0.9).
# Arms: base3k (fresh), jw5, jw10, jw20, pw10_jw10. Agreement (3 games) + walk-off origins per arm.
# PASS (vs base3k, floor ±0.4 fam / ±0.6 joint / ±0.03 AUC): jump press same-family agreement > 2 %
# (base 0-1 %), AUC jump not down, false press <= 1 %, SD/min not up, jump-off trips not up.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
R=eval_runs/1028_jw/REPORT.md; mkdir -p eval_runs/1028_jw eval_runs/walkoff
log() { echo "$*" | tee -a "$R"; }
waitbeam() { while pgrep -f '[b-b]eam.smp' >/dev/null; do sleep 60; done; }
run="mix run --no-compile --no-deps-check"
base=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize --button-events --stick-events --event-context --chunk-horizon 8 --offstage-weight 3 --stick-duration 8 --onset-weight 5 --action-oversample 1.0 --neutral-weight 1.0)
agree() {
  local n=$1 d=eval_runs/1001_queue/$1
  $run scripts/agreement_eval.exs --policy "checkpoints/coh_$n/model_best_policy.bin" --split "checkpoints/coh_$n/split.json" --games 3 --stride 2 --out "$d/agreement.json" > "$d/agreement.log" 2>&1
  grep -E "RESULT (CHANGE|frames with|press calibration (jump|attack)|false press)" "$d/agreement.log" | sed 's/^\[[0-9:]*\] //' | cut -c1-330 | tee -a "$R"
  [ -f "$d/agreement.json" ] && node -e 'const j=JSON.parse(require("fs").readFileSync(process.argv[1]));const s=j.by_action_change_frames.find(a=>a.action==="STANDING");const e=j.press_edges;console.log("RESULT STANDING change frames: n "+(s?s.n:0)+" change recall "+(s?s.change_recall:"?")+"%  | press edges same-family: "+e.map(x=>x.family+" "+x.model_same_press+"%").join(" "))' "$d/agreement.json" | tee -a "$R"
}
arm() {
  local name=$1; shift
  log ""; log "### arm jw_$name — $* ($(date +%H:%M))"
  waitbeam
  rm -rf "checkpoints/coh_jw_$name" "eval_runs/1001_queue/jw_$name"
  MAX_FILES=3000 EVALS="coherence closed_loop recovery" scripts/pq_experiment.sh "jw_$name" "${base[@]}" "$@" 2>&1 | grep -E "RESULT|FAILED|val_loss" | sed 's/^\[[0-9:]*\] //' | cut -c1-300 | tee -a "$R"
  if [ -f "checkpoints/coh_jw_$name/model_best_policy.bin" ]; then
    agree "jw_$name"
    EXLA_TARGET=cuda EXPHIL_EXLA_PRECISION=highest EXPHIL_GPU_MEMORY_FRACTION=0.45 $run scripts/walkoff_origins.exs --policy "checkpoints/coh_jw_$name/model_best_policy.bin" --label "jw_$name" --envs 32 --frames 1800 --out "eval_runs/walkoff/jw_$name.json" 2>&1 | grep -E "RESULT .*(origins|jump_off|special_off)" | sed 's/^\[[0-9:]*\] //' | cut -c1-250 | tee -a "$R"
  fi
}
log ""; log "# jump-window testbed queue ($(date '+%F %H:%M'))"
arm base3k
arm jw5       --jump-window-weight 5
arm jw10      --jump-window-weight 10
arm jw20      --jump-window-weight 20
arm pw10_jw10 --press-window-weight 10 --jump-window-weight 10
log ""; log "JW QUEUE DONE ($(date +%H:%M)) — table: node scripts/pq_digest.js jw_base3k jw_jw5 jw_jw10 jw_jw20 jw_pw10_jw10"
