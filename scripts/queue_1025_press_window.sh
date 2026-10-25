#!/usr/bin/env bash
# Queue 2 (2026-10-25): the two objective changes from the 10-24 testbed read,
# on the MinGRU testbed. Waits for the overnight unit to finish (ALL STAGES
# DONE in eval_runs/1024_overnight/REPORT.md), then merges the feature branch,
# compiles once (beam free, lib quiet), and runs the arms.
#
# Arms (base = frozen recipe heads: events, chunk 8, stick duration 8, onset 5,
# neutral 1.0; MAX_FILES 1500 to halve the re-embedding cost; agreement on 3 games):
#   base, base906 (seed floor)
#   pw2 / pw5          --press-window-weight 2 / 5     (press commitment)
#   (the scoped neutral-weight arms were dropped 10-25 10:20 — Bradley: the walk-offs are the
#    problem to look at, not a scoped patch; see scripts/overnight walk-off readout)
# PASS (vs base, beyond the 905/906 spread): press-edge same-family agreement
# (jump/attack/special) > 5 % (base 0-2 %) with calibration AUC not down and
# false press <= 1 %; STANDING change recall up from ~32 % with closed-loop
# SD/min NOT up (the blanket neutral 0.5 tripled it). A winner goes to a
# full-scale epoch from s906 ep2 next.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
R=eval_runs/1025_queue2/REPORT.md
mkdir -p eval_runs/1025_queue2
log() { echo "$*" | tee -a "$R"; }
log ""; log "# Queue 2 (press window / grounded neutral) — waiting for the overnight unit ($(date '+%F %H:%M'))"
until grep -q "ALL STAGES DONE" eval_runs/1024_overnight/REPORT.md 2>/dev/null; do sleep 300; done
log "overnight unit done ($(date +%H:%M)); waiting for a free beam and a quiet lib"
while :; do
  pgrep -f '[b]eam.smp' >/dev/null && { sleep 60; continue; }
  recent=$(find lib -name '*.ex' -mmin -10 | head -1); [ -n "$recent" ] && { sleep 120; continue; }
  break
done
newer=$(find ../nx/nx/lib ../nx/exla/lib ../nx/exla/c_src -newer ../nx/exla/cache/libexla.so \( -name '*.ex' -o -name '*.exs' -o -name '*.cc' -o -name '*.h' \) 2>/dev/null | head -3)
[ -z "$newer" ] || { log "NX_EDIT_IN_PROGRESS: $newer"; exit 1; }
log "== merge feature/press-window-1025 + compile ($(date +%H:%M))"
if ! git -c merge.autoStash=false merge --no-edit feature/press-window-1025 2>&1 | tee -a "$R"; then
  log "MERGE_FAILED — no training arms launched"
  exit 1
fi
if ! mix compile 2>&1 | tee -a "$R"; then
  log "COMPILE_FAILED"
  exit 1
fi
if ! mix test test/exphil/training/imitation/press_window_loss_test.exs 2>&1 | tee -a "$R"; then
  log "TEST_FAILED"
  exit 1
fi

base=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize --button-events --stick-events --event-context --chunk-horizon 8 --offstage-weight 3 --stick-duration 8 --onset-weight 5 --action-oversample 1.0)
agree() {
  local n=$1 d=eval_runs/1001_queue/$1
  mix run --no-compile --no-deps-check scripts/agreement_eval.exs --policy "checkpoints/coh_$n/model_best_policy.bin" --split "checkpoints/coh_$n/split.json" --games 3 --stride 2 --out "$d/agreement.json" > "$d/agreement.log" 2>&1
  grep -E "RESULT (CHANGE|frames with|press calibration (jump|attack|special)|false press)" "$d/agreement.log" | sed 's/^\[[0-9:]*\] //' | cut -c1-330 | tee -a "$R"
  [ -f "$d/agreement.json" ] && node -e 'const j=JSON.parse(require("fs").readFileSync(process.argv[1]));const s=j.by_action_change_frames.find(a=>a.action==="STANDING");const e=j.press_edges;console.log("RESULT STANDING change frames: n "+(s?s.n:0)+" change recall "+(s?s.change_recall:"?")+"%  families "+(s?s.families:"?")+"%  | press edges same-family: "+e.map(x=>x.family+" "+x.model_same_press+"%").join(" "))' "$d/agreement.json" | tee -a "$R"
}
arm() {
  local name=$1; shift
  log ""; log "### arm pw_$name — $* ($(date +%H:%M))"
  while pgrep -f '[b]eam.smp' >/dev/null; do sleep 60; done
  MAX_FILES=1500 EVALS="coherence closed_loop recovery" scripts/pq_experiment.sh "pw_$name" "${base[@]}" "$@" 2>&1 | grep -E "RESULT|FAILED|val_loss" | sed 's/^\[[0-9:]*\] //' | cut -c1-300 | tee -a "$R"
  [ -f "checkpoints/coh_pw_$name/model_best_policy.bin" ] && agree "pw_$name"
}
arm base      --neutral-weight 1.0
SEED=906 arm base906 --neutral-weight 1.0
arm pw2       --neutral-weight 1.0 --press-window-weight 2
arm pw5       --neutral-weight 1.0 --press-window-weight 5
arm pw10      --neutral-weight 1.0 --press-window-weight 10
log ""; log "QUEUE 2 DONE ($(date +%H:%M)) — table: node scripts/pq_digest.js pw_base pw_base906 pw_pw2 pw_pw5 pw_pw10"
