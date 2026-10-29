#!/usr/bin/env bash
# Master-like data arm (2026-10-29). The master-only line plateaued at fam ~53 by epoch 3 and width did
# not move it (eval_runs/1028_w768), so the open lever is MORE master-grade data. The untiered 56k
# files carry no rank; scripts/corpus_skill_score.exs scored every Fox game from play and
# scripts/skill_select.js calibrated a master-vs-untiered score (AUC 0.74: L-cancel rate, dash density,
# APM carry it; damage/kills/deaths do not separate). 3,986 untiered Fox games score at/above the
# master median -> replays/erickfm_masterlike = 7,837 master games (the master run's 16 held-out val
# games EXCLUDED) + those 3,986 (+43 % minutes). Same recipe/model/seed/LR schedule as the master arm.
#
# PASS: agreement on the master run's OWN 16 held-out games (--split checkpoints/coh_fox_mamba_master_ep1/
# split.json, 8 games, identical frames to every master/w768 readout) at epoch 3 above the master arm's
# ep3 (CHANGE fam 53.4 / joint 25.5, DECISION fam 64.3) by more than the ~2-point seed floor, and league
# Elo not below master_ep7_plain's neighbourhood. Below the floor = the selected games are not master-
# grade in the way that matters (execution stats are not decision quality), and data selection by play
# features is closed as a lever. The run's own val_loss is on 16 MIXED games and is NOT comparable.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
R=eval_runs/1029_masterlike/REPORT.md
log() { echo "$*" | tee -a "$R"; }
run="mix run --no-compile --no-deps-check"
msplit=checkpoints/coh_fox_mamba_master_ep1/split.json
log ""; log "# Master-like data arm — RESUMED at epoch 2 ($(date '+%F %H:%M')); corpus $(ls replays/erickfm_masterlike | wc -l) files"
for N in 2 3; do   # RESUME variant (10-29 15:20): epoch 1 done; epoch 2 resumes from progress.json (GOTCHA #147 fix-up applied)
  lr=0.0002; [ "$N" -ge 3 ] && lr=0.0001
  log ""; log "## epoch $N (lr $lr) ($(date +%H:%M))"
  env RECIPE=eval_runs/1029_masterlike/recipe.sh READOUT=light AGREE_GAMES=8 LR=$lr scripts/full_epoch_recipe.sh "$N" 2>&1 | grep -E "RESULT|DONE|FAILED|val_loss|INCOMPLETE|NO_PREV|STALE|\*\* \(|Parameters" | sed 's/^\[[0-9:]*\] //' | cut -c1-330 | tee -a "$R"
  ck=checkpoints/coh_fox_mamba_mlike_ep$N
  [ -f "$ck/model_best_policy.bin" ] || { log "EPOCH $N FAILED — stopping the arm"; break; }
  # the comparable readout: the MASTER run's held-out games (not this run's mixed split)
  log "### agreement on the master run's 16 held-out games (8 used) ($(date +%H:%M))"
  while scripts/beam_busy.sh; do sleep 60; done
  $run scripts/agreement_eval.exs --policy "$ck/model_best_policy.bin" --split "$msplit" --games 8 --stride 2 --out "eval_runs/1001_queue/fox_mamba_mlike_ep$N/agreement_mastersplit.json" > "eval_runs/1001_queue/fox_mamba_mlike_ep$N/agreement_mastersplit.log" 2>&1
  grep -E "RESULT (ALL|CHANGE|DECISION|frames with|press calibration (jump|attack|special))" "eval_runs/1001_queue/fox_mamba_mlike_ep$N/agreement_mastersplit.log" | sed 's/^\[[0-9:]*\] //' | cut -c1-330 | tee -a "$R"
done
best=""; for N in 3 2 1; do [ -f "checkpoints/coh_fox_mamba_mlike_ep$N/model_best_policy.bin" ] && { best=$N; break; }; done
[ -n "$best" ] || { log "no mlike checkpoint; league skipped"; log "MLIKE DONE ($(date +%H:%M))"; exit 0; }
cat > eval_runs/1029_masterlike/roster.json <<EOR
[
 {"name":"s906_ep2","policy":"checkpoints/coh_fox_mamba_v3s906_ep2/model_best_policy.bin","character":"fox","line":"candidate"},
 {"name":"master_ep3","policy":"checkpoints/coh_fox_mamba_master_ep3/model_best_policy.bin","character":"fox","line":"master 512, ep3 (equal epochs)"},
 {"name":"master_ep7_plain","policy":"checkpoints/coh_fox_mamba_master_ep7_plain/model_best_policy.bin","character":"fox","line":"master 512, ep7 (strongest so far)"},
 {"name":"mlike_ep${best}","policy":"checkpoints/coh_fox_mamba_mlike_ep${best}/model_best_policy.bin","character":"fox","line":"master + master-like untiered, ep${best}"}
]
EOR
while scripts/beam_busy.sh; do sleep 60; done
log ""; log "## walk-off origins mlike ep${best} ($(date +%H:%M))"
EXLA_TARGET=cuda EXPHIL_EXLA_PRECISION=highest EXPHIL_GPU_MEMORY_FRACTION=0.45 $run scripts/walkoff_origins.exs --policy "checkpoints/coh_fox_mamba_mlike_ep${best}/model_best_policy.bin" --label "mlike_ep${best}" --envs 32 --frames 1800 --out "eval_runs/walkoff/mlike_ep${best}.json" 2>&1 | grep -E "RESULT .*(origins|jump_off|special_off|walk_off)" | sed 's/^\[[0-9:]*\] //' | cut -c1-250 | tee -a "$R"
log ""; log "## league ($(date +%H:%M))"
$run scripts/league.exs --roster eval_runs/1029_masterlike/roster.json --envs 16 --frames 3600 --out eval_runs/1029_masterlike/league 2>&1 | grep -E "RESULT|unsupported|rror" | sed 's/^\[[0-9:]*\] //' | cut -c1-300 | tee -a "$R"
node -e 'const j=require("./eval_runs/1029_masterlike/league/league.json");j.standings.forEach(s=>console.log("RESULT standing",s.elo,s.name,"win",s.win_rate));const n=j.standings.map(s=>s.name);n.forEach(a=>console.log("RESULT matrix",a,n.map(b=>a===b?"-":j.win_matrix[a][b]).join(" ")))' 2>/dev/null | tee -a "$R"
log ""; log "MLIKE DONE ($(date +%H:%M))"
