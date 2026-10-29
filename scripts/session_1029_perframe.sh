#!/usr/bin/env bash
# PER-FRAME CONTROL arm (2026-10-29). Bradley: our representation diverged from slippi-ai (every-frame
# controller prediction with the previous controller fed back) into the input-coherence recipe (button/stick
# EVENTS as inputs, a stick head that predicts direction + hold length, chunk futures, onset + offstage
# weights) — crowned on OUR instruments (MinGRU testbed band/aim/jump rows), never against the plain
# version in play. This arm = the master recipe with those additions removed (eval_runs/1029_perframe/
# recipe.sh), same data / model / seed / LR / held-out games. 3 epochs, agreement on the master split,
# league vs master_ep3 (equal epochs), master_ep7_plain, s906_ep2.
# READ: if the plain per-frame arm is level or better in the league and on change-frame agreement, the
# coherence additions are invalidated as a representation (keep whichever is simpler = plain); if it is
# clearly worse in play (SD/min, league Elo) the additions earn their keep. Either way next = slippi-ai yardstick.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
R=eval_runs/1029_perframe/REPORT.md
log() { echo "$*" | tee -a "$R"; }
run="mix run --no-compile --no-deps-check"
msplit=checkpoints/coh_fox_mamba_master_ep1/split.json
log ""; log "# Per-frame control arm — 3 epochs ($(date '+%F %H:%M')); corpus $(ls replays/erickfm_master | wc -l) files"
for N in 1 2 3; do
  lr=0.0002; [ "$N" -ge 3 ] && lr=0.0001
  log ""; log "## epoch $N (lr $lr) ($(date +%H:%M))"
  env RECIPE=eval_runs/1029_perframe/recipe.sh READOUT=light AGREE_GAMES=8 LR=$lr scripts/full_epoch_recipe.sh "$N" 2>&1 | grep -E "RESULT|DONE|FAILED|val_loss|INCOMPLETE|NO_PREV|STALE|\*\* \(|Parameters" | sed 's/^\[[0-9:]*\] //' | cut -c1-330 | tee -a "$R"
  ck=checkpoints/coh_fox_mamba_perframe_ep$N
  [ -f "$ck/model_best_policy.bin" ] || { log "EPOCH $N FAILED — stopping the arm"; break; }
  # the comparable readout: the MASTER run's held-out games (not this run's mixed split)
  log "### agreement on the master run's 16 held-out games (8 used) ($(date +%H:%M))"
  while scripts/beam_busy.sh; do sleep 60; done
  $run scripts/agreement_eval.exs --policy "$ck/model_best_policy.bin" --split "$msplit" --games 8 --stride 2 --out "eval_runs/1001_queue/fox_mamba_perframe_ep$N/agreement_mastersplit.json" > "eval_runs/1001_queue/fox_mamba_perframe_ep$N/agreement_mastersplit.log" 2>&1
  grep -E "RESULT (ALL|CHANGE|DECISION|frames with|press calibration (jump|attack|special))" "eval_runs/1001_queue/fox_mamba_perframe_ep$N/agreement_mastersplit.log" | sed 's/^\[[0-9:]*\] //' | cut -c1-330 | tee -a "$R"
done
best=""; for N in 3 2 1; do [ -f "checkpoints/coh_fox_mamba_perframe_ep$N/model_best_policy.bin" ] && { best=$N; break; }; done
[ -n "$best" ] || { log "no perframe checkpoint; league skipped"; log "PERFRAME DONE ($(date +%H:%M))"; exit 0; }
cat > eval_runs/1029_perframe/roster.json <<EOR
[
 {"name":"s906_ep2","policy":"checkpoints/coh_fox_mamba_v3s906_ep2/model_best_policy.bin","character":"fox","line":"candidate"},
 {"name":"master_ep3","policy":"checkpoints/coh_fox_mamba_master_ep3/model_best_policy.bin","character":"fox","line":"master 512, ep3 (equal epochs)"},
 {"name":"master_ep7_plain","policy":"checkpoints/coh_fox_mamba_master_ep7_plain/model_best_policy.bin","character":"fox","line":"master 512, ep7 (strongest so far)"},
 {"name":"perframe_ep${best}","policy":"checkpoints/coh_fox_mamba_perframe_ep${best}/model_best_policy.bin","character":"fox","line":"per-frame control, ep${best}"}
]
EOR
while scripts/beam_busy.sh; do sleep 60; done
log ""; log "## walk-off origins perframe ep${best} ($(date +%H:%M))"
EXLA_TARGET=cuda EXPHIL_EXLA_PRECISION=highest EXPHIL_GPU_MEMORY_FRACTION=0.45 $run scripts/walkoff_origins.exs --policy "checkpoints/coh_fox_mamba_perframe_ep${best}/model_best_policy.bin" --label "perframe_ep${best}" --envs 32 --frames 1800 --out "eval_runs/walkoff/perframe_ep${best}.json" 2>&1 | grep -E "RESULT .*(origins|jump_off|special_off|walk_off)" | sed 's/^\[[0-9:]*\] //' | cut -c1-250 | tee -a "$R"
log ""; log "## league ($(date +%H:%M))"
$run scripts/league.exs --roster eval_runs/1029_perframe/roster.json --envs 16 --frames 3600 --out eval_runs/1029_perframe/league 2>&1 | grep -E "RESULT|unsupported|rror" | sed 's/^\[[0-9:]*\] //' | cut -c1-300 | tee -a "$R"
node -e 'const j=require("./eval_runs/1029_perframe/league/league.json");j.standings.forEach(s=>console.log("RESULT standing",s.elo,s.name,"win",s.win_rate));const n=j.standings.map(s=>s.name);n.forEach(a=>console.log("RESULT matrix",a,n.map(b=>a===b?"-":j.win_matrix[a][b]).join(" ")))' 2>/dev/null | tee -a "$R"
log ""; log "MLIKE DONE ($(date +%H:%M))"
