#!/usr/bin/env bash
# After the width arm: league w768 ep3 vs master ep7 plain vs master ep6 vs the candidate, plus the
# walk-off readout for w768 ep3. Waits for W768 DONE and a free exphil beam.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
R=eval_runs/1028_w768/REPORT.md
log() { echo "$*" | tee -a "$R"; }
until grep -q "W768 DONE" "$R" 2>/dev/null; do sleep 120; done
while scripts/beam_busy.sh; do sleep 60; done
run="mix run --no-compile --no-deps-check"
best=""; for N in 3 2 1; do [ -f "checkpoints/coh_fox_mamba_master_w768_ep$N/model_best_policy.bin" ] && { best=$N; break; }; done
[ -n "$best" ] || { log "no w768 checkpoint; league skipped"; exit 0; }
cat > eval_runs/1028_w768/roster.json <<EOR
[
 {"name":"s906_ep2","policy":"checkpoints/coh_fox_mamba_v3s906_ep2/model_best_policy.bin","character":"fox","line":"candidate"},
 {"name":"master_ep6","policy":"checkpoints/coh_fox_mamba_master_ep6/model_best_policy.bin","character":"fox","line":"master 512, ep6"},
 {"name":"master_ep7_plain","policy":"checkpoints/coh_fox_mamba_master_ep7_plain/model_best_policy.bin","character":"fox","line":"master 512, ep7 (strongest so far)"},
 {"name":"w768_ep${best}","policy":"checkpoints/coh_fox_mamba_master_w768_ep${best}/model_best_policy.bin","character":"fox","line":"master 768 wide, ep${best}"}
]
EOR
log ""; log "## walk-off origins w768 ep${best} ($(date +%H:%M))"
EXLA_TARGET=cuda EXPHIL_EXLA_PRECISION=highest EXPHIL_GPU_MEMORY_FRACTION=0.45 $run scripts/walkoff_origins.exs --policy "checkpoints/coh_fox_mamba_master_w768_ep${best}/model_best_policy.bin" --label "w768_ep${best}" --envs 32 --frames 1800 --out "eval_runs/walkoff/w768_ep${best}.json" 2>&1 | grep -E "RESULT .*(origins|jump_off|special_off|walk_off)" | sed 's/^\[[0-9:]*\] //' | cut -c1-250 | tee -a "$R"
log ""; log "## league ($(date +%H:%M))"
$run scripts/league.exs --roster eval_runs/1028_w768/roster.json --envs 16 --frames 3600 --out eval_runs/1028_w768/league 2>&1 | grep -E "RESULT|unsupported|rror" | sed 's/^\[[0-9:]*\] //' | cut -c1-300 | tee -a "$R"
node -e 'const j=require("./eval_runs/1028_w768/league/league.json");j.standings.forEach(s=>console.log("RESULT standing",s.elo,s.name,"win",s.win_rate));const n=j.standings.map(s=>s.name);n.forEach(a=>console.log("RESULT matrix",a,n.map(b=>a===b?"-":j.win_matrix[a][b]).join(" ")))' 2>/dev/null | tee -a "$R"
log ""; log "W768 LEAGUE DONE ($(date +%H:%M))"
