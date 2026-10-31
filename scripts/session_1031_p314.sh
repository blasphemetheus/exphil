#!/usr/bin/env bash
# PER-PLAYER PILOT (2026-10-31 overnight; relaunched 03:08 with the resume fix). Bradley's product: a bot that plays like ONE player. The corpus has
# in-game nametags; [314] has 324 Fox games (no dittos, so --select-character-port = 314's port). Fine-tune the
# strongest generalist (master_ep7_plain) on 314's games for 3 epochs at LR 1e-4 (recipe unchanged otherwise);
# the trainer holds out 16 of 314's games by hash (split.json).
# MEASURE on 314's 16 held-out games: agreement of the fine-tuned model vs the generalist (control) on the SAME
# frames. PASS = change-frame families/joint up by more than the ~2-point floor on 314's own games = the
# per-player bot is a real thing (plays more like him than the generalist does). Also: the fine-tuned model on the
# MASTER split (what it forgets) and the yardstick (3 headless games vs slippi-ai imitation Fox) for deficiencies.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
R=eval_runs/1031_p314/REPORT.md
log() { echo "$*" | tee -a "$R"; }
run="mix run --no-compile --no-deps-check"
log ""; log "# Per-player pilot [314] — fine-tune master_ep7_plain, 3 epochs ($(date '+%F %H:%M')); corpus $(ls replays/p314 | wc -l) files"
for N in 2 3 4; do   # ep1 = SYMLINK to master_ep7_plain: full_epoch_recipe.sh only passes --resume for N>1 (the 02:17 run trained from scratch — void)
  log ""; log "## epoch $N (lr 0.0001) ($(date +%H:%M))"
  env RECIPE=eval_runs/1031_p314/recipe.sh READOUT=light AGREE_GAMES=16 LR=0.0001 scripts/full_epoch_recipe.sh "$N" 2>&1 | grep -E "RESULT (ALL|CHANGE|DECISION|.*vs idle)|DONE|FAILED|val_loss|INCOMPLETE|NO_PREV|STALE|\*\* \(" | sed 's/^\[[0-9:]*\] //' | cut -c1-330 | tee -a "$R"
  [ -f "checkpoints/coh_fox_mamba_p314_ep$N/model_best_policy.bin" ] || { log "EPOCH $N FAILED — stopping"; break; }
done
best=""; for N in 4 3 2; do [ -f "checkpoints/coh_fox_mamba_p314_ep$N/model_best_policy.bin" ] && { best=$N; break; }; done
[ -n "$best" ] || { log "P314 DONE (no checkpoint) ($(date +%H:%M))"; exit 0; }
split=checkpoints/coh_fox_mamba_p314_ep2/split.json
while scripts/beam_busy.sh; do sleep 60; done
log ""; log "## CONTROL: generalist master_ep7_plain on 314's 16 held-out games ($(date +%H:%M))"
mkdir -p eval_runs/1001_queue/p314_control
$run scripts/agreement_eval.exs --policy checkpoints/coh_fox_mamba_master_ep7_plain/model_best_policy.bin --split "$split" --games 16 --stride 2 --out eval_runs/1001_queue/p314_control/agreement.json > eval_runs/1001_queue/p314_control/agreement.log 2>&1
grep -E "RESULT (ALL|CHANGE|DECISION)" eval_runs/1001_queue/p314_control/agreement.log | sed 's/^\[[0-9:]*\] //' | cut -c1-330 | tee -a "$R"
for N in $(seq 2 $best); do
  log "## fine-tuned ep$N on 314's held-out games (from its epoch readout):"; grep -E "RESULT (CHANGE|DECISION)" eval_runs/1001_queue/fox_mamba_p314_ep$N/agreement.log | sed 's/^\[[0-9:]*\] //' | cut -c1-200 | tee -a "$R"
done
log ""; log "## FORGETTING: fine-tuned ep$best on the MASTER split (master_ep7_plain there: CHANGE fam 53.9 / joint 27.3) ($(date +%H:%M))"
$run scripts/agreement_eval.exs --policy checkpoints/coh_fox_mamba_p314_ep$best/model_best_policy.bin --split checkpoints/coh_fox_mamba_master_ep1/split.json --games 8 --stride 2 --out eval_runs/1001_queue/fox_mamba_p314_ep$best/agreement_mastersplit.json > eval_runs/1001_queue/fox_mamba_p314_ep$best/agreement_mastersplit.log 2>&1
grep -E "RESULT (CHANGE|DECISION)" eval_runs/1001_queue/fox_mamba_p314_ep$best/agreement_mastersplit.log | sed 's/^\[[0-9:]*\] //' | cut -c1-330 | tee -a "$R"
log ""; log "## YARDSTICK: p314 ep$best vs slippi-ai imitation Fox d18, 3 headless games ($(date +%H:%M))"
mkdir -p eval_runs/1031_p314/vs_slippi
EXLA_TARGET=cuda EXPHIL_EXLA_PRECISION=highest EXPHIL_GPU_MEMORY_FRACTION=0.45 timeout 900 $run scripts/play_vs_slippi_ai.exs --policy checkpoints/coh_fox_mamba_p314_ep$best/model_best_policy.bin --character fox --stage final_destination --dolphin ~/.local/share/slippi/exi-ai/dolphin-emu-headless --iso "$HOME/isos/melee.iso" --headless --stateful-step --reaction-delay 1 --num-games 3 --slippi-ai-model ~/git/slippi-ai/models/fox_d18_imitation_v3 --replay-dir eval_runs/1031_p314/vs_slippi/ --results eval_runs/1031_p314/vs_slippi/results.json 2>&1 | grep -E "\[Match\] game|exphil [0-9] – " | tee -a "$R"
log ""; log "P314 DONE ($(date +%H:%M))"
