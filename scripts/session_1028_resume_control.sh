#!/usr/bin/env bash
# Resume the A/B control (master ep7 plain) from chunk 49/123 after the 08:03 stop (Bradley: stop,
# do the Edifice/fork work, resume). Same conditions as the pw10 arm: fused OFF, batch 64. Then the
# plain arm's walk-off origins and the 4-bot league.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1 EDIFICE_FUSED_CUSTOM_CALL=0
R=eval_runs/1028_fullscale_pw/REPORT.md
log() { echo "$*" | tee -a "$R"; }
waitbeam() { while pgrep -f '[b-b]eam.smp' >/dev/null; do sleep 60; done; }
run="mix run --no-compile --no-deps-check"
log ""; log "## master ep7 plain — RESUMED from chunk 49/123 ($(date '+%F %H:%M'))"
waitbeam
env RECIPE=eval_runs/1024_master/recipe.sh READOUT=light AGREE_GAMES=8 LR=0.0001 BATCH=64 TAG=plain scripts/full_epoch_recipe.sh 7 2>&1 | grep -E "RESULT|DONE|FAILED|val_loss|INCOMPLETE|NO_PREV|STALE|Resuming|\*\* \(" | sed 's/^\[[0-9:]*\] //' | cut -c1-330 | tee -a "$R"
name=fox_mamba_master_ep7_plain
if [ -f "checkpoints/coh_$name/model_best_policy.bin" ]; then
  waitbeam
  EXLA_TARGET=cuda EXPHIL_EXLA_PRECISION=highest EXPHIL_GPU_MEMORY_FRACTION=0.45 $run scripts/walkoff_origins.exs --policy "checkpoints/coh_$name/model_best_policy.bin" --label "$name" --envs 32 --frames 1800 --out "eval_runs/walkoff/$name.json" 2>&1 | grep -E "RESULT .*(origins|jump_off|special_off|walk_off)" | sed 's/^\[[0-9:]*\] //' | cut -c1-250 | tee -a "$R"
fi
waitbeam
log ""; log "## mini league ($(date +%H:%M))"
$run scripts/league.exs --roster eval_runs/1028_fullscale_pw/roster.json --envs 16 --frames 3600 --out eval_runs/1028_fullscale_pw/league 2>&1 | grep -E "RESULT|unsupported|rror" | sed 's/^\[[0-9:]*\] //' | cut -c1-300 | tee -a "$R"
node -e 'const j=require("./eval_runs/1028_fullscale_pw/league/league.json");j.standings.forEach(s=>console.log("RESULT standing",s.elo,s.name,"win",s.win_rate));const n=j.standings.map(s=>s.name);n.forEach(a=>console.log("RESULT matrix",a,n.map(b=>a===b?"-":j.win_matrix[a][b]).join(" ")))' 2>/dev/null | tee -a "$R"
log ""; log "FULLSCALE A/B DONE ($(date +%H:%M))"
