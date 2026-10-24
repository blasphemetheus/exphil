#!/usr/bin/env bash
# Overnight 2026-10-24 (Bradley: "queue all of these, testbed first, then
# keep doing experiments; I'll check in the morning"). One beam at a time.
# Everything appends to eval_runs/1024_overnight/REPORT.md.
#
# STAGE 0  null-label DAgger sets (CPU): the five r9o/live1014 windows with
#          label := the bot's own press (scripts/dagger_set_null.exs).
# STAGE 1  TESTBED QUEUE on press commitment + passivity (MinGRU, 3,000 games,
#          ~9 min per arm incl. the agreement eval). Base = the frozen recipe's
#          heads (events, chunk 8, stick duration 8, onset 5, neutral 1.0).
#          Press commitment arms: onset 10, transition 3, transition 3 + onset
#          10, chunk weight 2, button weight 4. Passivity arms: neutral 0.5,
#          neutral 0.25, silent-fall 3, action oversample 2. Combo: onset 10 +
#          transition 3 + neutral 0.5. Seed control: base at seed 906.
#          PASS (vs base, beyond the 905/906 spread): press-edge same-family
#          agreement (jump/attack/special) UP from the ~2-8 % level and press
#          calibration AUC not down; false press <= 1 %; STANDING change
#          recall UP from ~49 %; change recall up with false change <= 10 %;
#          closed-loop SD/min not up. A winner goes to full scale next.
# STAGE 2  MASTER-ONLY full arm: the frozen Mamba recipe on the 7,853
#          master-master games, epochs 1-6 (lr 2e-4 then 1e-4 from epoch 3),
#          light readout + agreement eval per epoch; the candidate scored on
#          the same master held-out split for the comparison.
#          PASS: agreement on master held-out above the candidate's on the
#          same split (change-frame families ~47 %), closed-loop SD/min <=
#          the candidate's, and >= even vs the candidate in the league.
# STAGE 3  DOSE-MATCHED NULL ARM: epoch 3 from s906 ep2 with the null sets as
#          MIX (same windows, dose, lr 1e-4, seed 906 as dag9o), the full
#          curve-row readout, then a mini league: s906_ep2 vs dag9o vs null9o
#          vs master. PASS/READ: null regresses like dag9o -> the mixing
#          mechanics are the harm; null holds -> the labels were.
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
R=eval_runs/1024_overnight/REPORT.md
mkdir -p eval_runs/1024_overnight
log() { echo "$*" | tee -a "$R"; }
waitbeam() { while pgrep -f '[b]eam.smp' > /dev/null; do sleep 60; done; }
log ""; log "# Overnight 2026-10-24 — started $(date '+%F %H:%M')"
newer=$(find ../nx/nx/lib ../nx/exla/lib ../nx/exla/c_src -newer ../nx/exla/cache/libexla.so \( -name '*.ex' -o -name '*.exs' -o -name '*.cc' -o -name '*.h' \) 2>/dev/null | head -3)
[ -z "$newer" ] || { log "NX_EDIT_IN_PROGRESS: $newer"; exit 1; }
newest_beam=$(ls -t _build/dev/lib/exphil/ebin/*.beam 2>/dev/null | head -1)
stale=$(find lib -name '*.ex' -newer "${newest_beam:-/nonexistent}" 2>/dev/null | head -3)
[ -z "$stale" ] || { log "STALE_BUILD: $stale"; exit 1; }
waitbeam

# ---------------------------------------------------------------- STAGE 0
log ""; log "## Stage 0 — null-label sets ($(date +%H:%M))"
parts=(sim_dagger_expert_r9o_2091 sim_dagger_expert_r9o_2094 sim_dagger_expert_r9o_2097 sim_dagger_expert_r9o_2100 slp_dagger_expert_live1014)
nullmix=""
for part in "${parts[@]}"; do
  outf=data/silent_fall/${part}_null_lean.frames
  if [ ! -f "$outf" ]; then
    elixir -pa '_build/dev/lib/*/ebin' scripts/dagger_set_null.exs "data/silent_fall/${part}_split.frames" "$outf" 2>&1 | grep -E "RESULT|\*\*|rror" | cut -c1-300 | tee -a "$R"
  fi
  [ -f "$outf" ] && nullmix="${nullmix:+$nullmix,}$outf"
done
log "null mix: $nullmix"

# ---------------------------------------------------------------- STAGE 1
log ""; log "## Stage 1 — testbed queue: press commitment + passivity ($(date +%H:%M))"
base=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize --button-events --stick-events --event-context --chunk-horizon 8 --offstage-weight 3 --stick-duration 8)
agree() {  # NAME: teacher-forced agreement on the arm's own held-out split
  local n=$1 d=eval_runs/1001_queue/$1
  mix run --no-compile --no-deps-check scripts/agreement_eval.exs --policy "checkpoints/coh_$n/model_best_policy.bin" --split "checkpoints/coh_$n/split.json" --games 6 --stride 2 --out "$d/agreement.json" > "$d/agreement.log" 2>&1
  grep -E "RESULT (CHANGE|frames with|press calibration (jump|attack|special)|false press)" "$d/agreement.log" | sed 's/^\[[0-9:]*\] //' | cut -c1-330 | tee -a "$R"
  [ -f "$d/agreement.json" ] && node -e 'const j=JSON.parse(require("fs").readFileSync(process.argv[1]));const s=j.by_action_change_frames.find(a=>a.action==="STANDING");const e=j.press_edges;console.log("RESULT STANDING change frames: n "+(s?s.n:0)+" change recall "+(s?s.change_recall:"?")+"%  families "+(s?s.families:"?")+"%  | press edges same-family: "+e.map(x=>x.family+" "+x.model_same_press+"%").join(" "))' "$d/agreement.json" | tee -a "$R"
}
arm() {  # NAME flags...
  local name=$1; shift
  log ""; log "### arm pq_$name — $* ($(date +%H:%M))"
  waitbeam
  EVALS="coherence closed_loop recovery" scripts/pq_experiment.sh "pq_$name" "${base[@]}" "$@" 2>&1 | grep -E "RESULT|FAILED|val_loss" | sed 's/^\[[0-9:]*\] //' | cut -c1-300 | tee -a "$R"
  [ -f "checkpoints/coh_pq_$name/model_best_policy.bin" ] && agree "pq_$name"
}
arm base      --onset-weight 5  --neutral-weight 1.0  --action-oversample 1.0
SEED=906 arm base906 --onset-weight 5  --neutral-weight 1.0  --action-oversample 1.0
arm on10      --onset-weight 10 --neutral-weight 1.0  --action-oversample 1.0
arm tw3       --onset-weight 5  --neutral-weight 1.0  --action-oversample 1.0 --transition-weight 3
arm tw3on10   --onset-weight 10 --neutral-weight 1.0  --action-oversample 1.0 --transition-weight 3
arm cw2       --onset-weight 5  --neutral-weight 1.0  --action-oversample 1.0 --chunk-weight 2
arm bw4       --onset-weight 5  --neutral-weight 1.0  --action-oversample 1.0 --button-weight 4
arm nw05      --onset-weight 5  --neutral-weight 0.5  --action-oversample 1.0
arm nw025     --onset-weight 5  --neutral-weight 0.25 --action-oversample 1.0
arm sf3       --onset-weight 5  --neutral-weight 1.0  --action-oversample 1.0 --silent-fall-weight 3
arm ao2       --onset-weight 5  --neutral-weight 1.0  --action-oversample 2.0
arm combo     --onset-weight 10 --neutral-weight 0.5  --action-oversample 1.0 --transition-weight 3
log ""; log "STAGE 1 DONE ($(date +%H:%M))"

# ---------------------------------------------------------------- STAGE 2
log ""; log "## Stage 2 — master-only full arm, epochs 1-6 ($(date +%H:%M))"
for N in 1 2 3 4 5 6; do
  lr=0.0002; [ "$N" -ge 3 ] && lr=0.0001
  waitbeam
  log ""; log "### master epoch $N (lr $lr) ($(date +%H:%M))"
  env RECIPE=eval_runs/1024_master/recipe.sh READOUT=light AGREE_GAMES=8 LR=$lr scripts/full_epoch_recipe.sh "$N" 2>&1 | grep -E "RESULT|DONE|FAILED|val_loss|INCOMPLETE|NO_PREV|STALE" | sed 's/^\[[0-9:]*\] //' | cut -c1-330 | tee -a "$R"
  if [ "$N" = 1 ] && [ -f checkpoints/coh_fox_mamba_master_ep1/split.json ]; then
    waitbeam
    log "### candidate s906_ep2 on the master held-out split (the comparison)"
    mix run --no-compile --no-deps-check scripts/agreement_eval.exs --policy checkpoints/coh_fox_mamba_v3s906_ep2/model_best_policy.bin --split checkpoints/coh_fox_mamba_master_ep1/split.json --games 8 --stride 2 --out eval_runs/1024_overnight/candidate_on_master.json > eval_runs/1024_overnight/candidate_on_master.log 2>&1
    grep -E "RESULT (ALL|CHANGE|frames with|press calibration (jump|attack))" eval_runs/1024_overnight/candidate_on_master.log | sed 's/^\[[0-9:]*\] //' | cut -c1-330 | tee -a "$R"
  fi
done
log ""; log "STAGE 2 DONE ($(date +%H:%M))"

# ---------------------------------------------------------------- STAGE 3
log ""; log "## Stage 3 — dose-matched null arm (epoch 3 from s906 ep2, null mix) ($(date +%H:%M))"
if [ -n "$nullmix" ]; then
  waitbeam
  env RUN=fox_mamba_v3s906 SEED=906 LR=0.0001 TAG=null9o MIX="$nullmix" scripts/full_mamba_epoch.sh 3 2>&1 | grep -E "RESULT|DONE|FAILED|val_loss|INCOMPLETE|NO_|STALE|^fox_mamba|^expert|^evt2" | sed 's/^\[[0-9:]*\] //' | cut -c1-330 | tee -a "$R"
  log "curve rows (eval_runs/1011_full/curve.txt tail):"; tail -6 eval_runs/1011_full/curve.txt | cut -c1-300 | tee -a "$R"
else
  log "no null mix built; stage 3 skipped"
fi
# mini league: candidate vs dag9o vs null vs master
best_master=""; for N in 6 5 4 3 2 1; do [ -f "checkpoints/coh_fox_mamba_master_ep$N/model_best_policy.bin" ] && { best_master="checkpoints/coh_fox_mamba_master_ep$N/model_best_policy.bin"; break; }; done
cat > eval_runs/1024_overnight/roster.json <<EOF
[
 {"name":"s906_ep2","policy":"checkpoints/coh_fox_mamba_v3s906_ep2/model_best_policy.bin","character":"fox","line":"candidate"},
 {"name":"s906_dag9o","policy":"checkpoints/coh_fox_mamba_v3s906_ep3_dag9o/model_best_policy.bin","character":"fox","line":"expert-label DAgger epoch (10-16)"},
 {"name":"s906_null9o","policy":"checkpoints/coh_fox_mamba_v3s906_ep3_null9o/model_best_policy.bin","character":"fox","line":"dose-matched null epoch (10-24)"},
 {"name":"master_last","policy":"${best_master:-missing}","character":"fox","line":"master-only arm, last epoch"},
 {"name":"V3.1_ep3","policy":"checkpoints/fox_v3_1_20260919_022008_ep3/model_best_policy.bin","character":"fox","line":"GRU generalist (crowned 09-20)"}
]
EOF
waitbeam
log ""; log "## Mini league ($(date +%H:%M))"
mix run --no-compile --no-deps-check scripts/league.exs --roster eval_runs/1024_overnight/roster.json --envs 16 --frames 3600 --out eval_runs/1024_overnight/league 2>&1 | grep -E "RESULT|unsupported|rror" | sed 's/^\[[0-9:]*\] //' | cut -c1-300 | tee -a "$R"
log ""; log "ALL STAGES DONE ($(date '+%F %H:%M'))"
