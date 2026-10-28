#!/usr/bin/env bash
# 10-28: full-scale A/B of the press-window objective on the master-only arm (the better data):
#   ep7_pw10 = master ep6 + one epoch with --press-window-weight 10   (lr 1e-4, seed 905)
#   ep7      = master ep6 + one plain epoch                           (same resume / lr / seed)
# Fused custom call OFF (EXLA operand_layouts on the new stack). Light readout + agreement each,
# then walk-off origins, then a 3-bot mini league (candidate, master_ep6, both ep7s).
# PASS (pw10 vs plain, master held-out split): press-edge same-family agreement up (attack/special from
# ~1-2 %), families up beyond ±0.4, false press <= 1 %, SD/min not up, league not worse.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1 EDIFICE_FUSED_CUSTOM_CALL=0
R=eval_runs/1028_fullscale_pw/REPORT.md; mkdir -p eval_runs/1028_fullscale_pw eval_runs/walkoff
log() { echo "$*" | tee -a "$R"; }
waitbeam() { while pgrep -f '[b-b]eam.smp' >/dev/null; do sleep 60; done; }
run="mix run --no-compile --no-deps-check"
log ""; log "# full-scale press-window A/B from master_ep6 ($(date '+%F %H:%M'))"
for tag in pw10 plain; do
  waitbeam
  extra=""; [ "$tag" = pw10 ] && extra="--press-window-weight 10"
  log ""; log "## master ep7 ${tag} ($(date +%H:%M))"
  env RECIPE=eval_runs/1024_master/recipe.sh READOUT=light AGREE_GAMES=8 LR=0.0001 TAG=$tag EXTRA_FLAGS="$extra" scripts/full_epoch_recipe.sh 7 2>&1 | grep -E "RESULT|DONE|FAILED|val_loss|INCOMPLETE|NO_PREV|STALE|\*\* \(" | sed 's/^\[[0-9:]*\] //' | cut -c1-330 | tee -a "$R"
  name=fox_mamba_master_ep7_$tag
  if [ -f "checkpoints/coh_$name/model_best_policy.bin" ]; then
    waitbeam
    EXLA_TARGET=cuda EXPHIL_EXLA_PRECISION=highest EXPHIL_GPU_MEMORY_FRACTION=0.45 $run scripts/walkoff_origins.exs --policy "checkpoints/coh_$name/model_best_policy.bin" --label "$name" --envs 32 --frames 1800 --out "eval_runs/walkoff/$name.json" 2>&1 | grep -E "RESULT .*(origins|jump_off|special_off|walk_off)" | sed 's/^\[[0-9:]*\] //' | cut -c1-250 | tee -a "$R"
  fi
done
cat > eval_runs/1028_fullscale_pw/roster.json <<EOR
[
 {"name":"s906_ep2","policy":"checkpoints/coh_fox_mamba_v3s906_ep2/model_best_policy.bin","character":"fox","line":"candidate"},
 {"name":"master_ep6","policy":"checkpoints/coh_fox_mamba_master_ep6/model_best_policy.bin","character":"fox","line":"master-only arm"},
 {"name":"master_ep7_plain","policy":"checkpoints/coh_fox_mamba_master_ep7_plain/model_best_policy.bin","character":"fox","line":"master ep7 control"},
 {"name":"master_ep7_pw10","policy":"checkpoints/coh_fox_mamba_master_ep7_pw10/model_best_policy.bin","character":"fox","line":"master ep7 + press-window 10"}
]
EOR
waitbeam
log ""; log "## mini league ($(date +%H:%M))"
$run scripts/league.exs --roster eval_runs/1028_fullscale_pw/roster.json --envs 16 --frames 3600 --out eval_runs/1028_fullscale_pw/league 2>&1 | grep -E "RESULT|unsupported|rror" | sed 's/^\[[0-9:]*\] //' | cut -c1-300 | tee -a "$R"
node -e 'const j=require("./eval_runs/1028_fullscale_pw/league/league.json");j.standings.forEach(s=>console.log("RESULT standing",s.elo,s.name,"win",s.win_rate));const n=j.standings.map(s=>s.name);n.forEach(a=>console.log("RESULT matrix",a,n.map(b=>a===b?"-":j.win_matrix[a][b]).join(" ")))' 2>/dev/null | tee -a "$R"
log ""; log "FULLSCALE A/B DONE ($(date +%H:%M))"
