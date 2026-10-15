#!/usr/bin/env bash
# DAgger round r9o (2026-10-15, Bradley: "both offstage and onstage") from
# the full-run candidate s906 ep2 — the states the live look (HANDOFF §8e
# "INSTRUMENT READS") found deficient: unforced deaths 44 % (recovery), OOS
# drop 8 vs 27 %, tech rate 0.26 vs 0.51, in-threat smashes punished 2x.
# Labeler v6 = window `:onstage` (scripts/lib/expert_recovery_labeler.exs):
# air ∪ shield ∪ knockdown ∪ in-threat, with opponent / shield-stun /
# knockdown / percent dims. Chain = queue 45's, plus the 31 live games vs
# Bradley relabelled the same way (scripts/slp_dagger_expert.exs):
#   1. index v6 (400 FD Fox games from the candidate's TRAIN split) +
#      self-check on held-out games (recovery keys AND the onstage cells
#      within x0.5–x2 of the expert) + coverage gate (held-out q95)
#   2. r9o rollout + relabel from s906 ep2 (12 seeds, 4 beams, --keep-actual,
#      --max-run 90) and the live session relabel
#   3. concat / hazards / split_gated / onstage-cell dose + label content
# STOPS after the dose ("ROUND READY"): the mixed epoch 3 is launched by hand
# after the dose is read (feedback 10-10: count the dose before launching),
#   env MIX=data/silent_fall/sim_dagger_expert_r9o_split.frames RUN=fox_mamba_v3s906 SEED=906 scripts/full_mamba_epoch.sh 3
# (move checkpoints/coh_fox_mamba_v3s906_ep3 — the dead plain epoch 3 — aside first).
set -uo pipefail
cd /home/blewf/git/exphil
export EDIFICE_LOCAL_NX=1
while pgrep -f '[b]eam.smp' > /dev/null; do sleep 60; done
echo "== beam free ($(date +%H:%M))"
newer=$(find ../nx/nx/lib ../nx/exla/lib ../nx/exla/c_src -newer ../nx/exla/cache/libexla.so \( -name '*.ex' -o -name '*.exs' -o -name '*.cc' -o -name '*.h' \) 2>/dev/null | head -3)
if [ -n "$newer" ]; then echo "NX_EDIT_IN_PROGRESS: $newer"; exit 1; fi
newest_beam=$(ls -t _build/dev/lib/exphil/ebin/*.beam 2>/dev/null | head -1)
stale=$(find lib -name '*.ex' -newer "${newest_beam:-/nonexistent}" 2>/dev/null | head -3)
[ -z "$stale" ] || { echo "STALE_BUILD (lib newer than _build/dev): $stale"; exit 1; }

pol=checkpoints/coh_fox_mamba_v3s906_ep2/model_best_policy.bin
split=checkpoints/coh_fox_mamba_v3s906_ep2/split.json
held=data/silent_fall/heldout_fd_fox_split.json
idx=data/silent_fall/expert_recovery_index_v6.bin
live='eval_runs/fox_mamba_live_v3s906ep2_20261014_131550/2026-10-Mainline/*.slp'
q=eval_runs/1001_queue
[ -s "$pol" ] || { echo "NO_POLICY $pol"; exit 1; }
mf() { elixir -pa '_build/dev/lib/*/ebin' "$@"; }
# GPU-side labeler reads need EXLA as the default backend = the mix config (queue 37 lesson)
mx() { mix run --no-compile "$@"; }

# 1. index v6 (onstage window) + self-check + gates
echo "== index v6, onstage window ($(date +%H:%M))"
mx scripts/build_expert_recovery_index.exs --split $split --games 400 --window onstage --out $idx 2>&1 | grep -E "RESULT|error|\*\*" | cut -c1-600
[ -s $idx ] || { echo "LABELER_FAILED (no v6 index)"; exit 1; }
echo "== self-check v6 ($(date +%H:%M))"
mx scripts/expert_labeler_selfcheck.exs --split $held --index $idx --games 24 --out $q/labeler_selfcheck_v6.json 2>&1 | grep -E "RESULT|error|Error|\*\*" | cut -c1-600
ok=$(node -e '
try {
  const r = JSON.parse(require("fs").readFileSync("eval_runs/1001_queue/labeler_selfcheck_v6.json"));
  const w = (a, b, n) => b * n < 20 || (b > 0 && a / b >= 0.5 && a / b <= 2.0);
  let ok = r.n >= 10000 && w(r.label_up, r.actual_up, r.n_spent) && w(r.label_jump, r.actual_jump, r.n_in_hand) && w(r.label_b, r.actual_b, r.n_spent);
  const bad = [];
  for (const [cell, c] of Object.entries(r.cells || {})) for (const [e, v] of Object.entries(c.events)) if (!w(v.label, v.actual, c.n)) bad.push(cell + "." + e + " " + v.label.toFixed(4) + "/" + v.actual.toFixed(4));
  if (bad.length) console.error("outside x0.5-x2: " + bad.join("  "));
  console.log(ok && bad.length === 0 ? "yes" : "no");
} catch (e) { console.log("no"); }')
echo "== labeler v6 within x0.5-x2 of the expert (recovery keys + onstage cells): $ok"
[ "$ok" = yes ] || { echo "LABELER_FAILED"; exit 1; }
mx scripts/expert_labeler_distance.exs --split $held --index $idx --games 24 --out $q/labeler_distance_v6_heldout.json 2>&1 | grep -E "RESULT|\*\*" | cut -c1-600
gate=$(node -e 'const r=JSON.parse(require("fs").readFileSync("eval_runs/1001_queue/labeler_distance_v6_heldout.json"));console.log(r.gate_q95?r.gate_q95.toFixed(3):"")')
[ -n "$gate" ] || { echo "LABELER_FAILED (no gate)"; exit 1; }
echo "== v6 coverage gate d2 <= $gate (held-out q95)"

# 2. r9o: gated v6 round from s906 ep2 (12 seeds, 4 beams) + the live session
r9=data/silent_fall/sim_dagger_expert_r9o.frames
r9s=data/silent_fall/sim_dagger_expert_r9o_split.frames
echo "== gated v6 rollout + relabel from s906 ep2 (d2 <= $gate), 12 seeds in 4 beams, --keep-actual --max-run 90 ($(date +%H:%M))"
parts=()
for grp in 2091,2092,2093 2094,2095,2096 2097,2098,2099 2100,2101,2102; do
  part=data/silent_fall/sim_dagger_expert_r9o_${grp%%,*}.frames
  mix run --no-compile scripts/sim_recovery_dagger_expert.exs --policy $pol --index $idx --max-d2 $gate --keep-actual --max-run 90 \
    --seeds $grp --label-seed 12 --out $part --report $q/sim_dagger_expert_r9o_${grp%%,*}.json \
    > logs/dagger_r9o_${grp%%,*}.log 2>&1
  grep -E "RESULT|seed |RESOURCE_EXHAUSTED|\*\*" logs/dagger_r9o_${grp%%,*}.log | cut -c1-600
  [ -s $q/sim_dagger_expert_r9o_${grp%%,*}.json ] || { echo "DAGGER_FAILED (seeds $grp, see logs/dagger_r9o_${grp%%,*}.log)"; exit 1; }
  parts+=("$part")
done
livep=data/silent_fall/slp_dagger_expert_live1014.frames
echo "== live session relabel (31 games vs Bradley, bot port 1) ($(date +%H:%M))"
mix run --no-compile scripts/slp_dagger_expert.exs --index $idx --set "$live" --port 1 --max-d2 $gate --keep-actual --max-run 90 \
  --label-seed 12 --out $livep --report $q/slp_dagger_expert_live1014.json > logs/dagger_r9o_live.log 2>&1
grep -E "RESULT|skip|\*\*" logs/dagger_r9o_live.log | cut -c1-600
[ -s $q/slp_dagger_expert_live1014.json ] || { echo "DAGGER_FAILED (live relabel, see logs/dagger_r9o_live.log)"; exit 1; }
parts+=("$livep")

# 3. hazards / split / dose — PER PART (10-15 14:50: the concatenated set is
# > 4 GB uncompressed and term_to_binary's compressed header wraps; the
# loader takes a comma-separated glob, so MIX = the split parts)
splits=()
for part in "${parts[@]}"; do
  sp=${part%.frames}_split.frames
  echo "== $(basename $part) ($(date +%H:%M))"
  mf scripts/dagger_set_label_hazards.exs $part 2>&1 | grep -E "RESULT|rror" | cut -c1-400
  mf scripts/dagger_set_split_gated.exs $part $sp 2>&1 | grep -E "RESULT|rror" | cut -c1-300
  [ -s $sp ] || { echo "DAGGER_FAILED (no split for $part)"; exit 1; }
  splits+=("$sp")
done
echo "== dose + label content, onstage cells (unsplit parts pooled) ($(date +%H:%M))"
mf scripts/dagger_set_onstage_cells.exs "${parts[@]}" 2>&1 | cut -c1-600
mf scripts/dagger_set_drift_cell.exs "${parts[0]}" 2>&1 | grep -E "RESULT|rror" | cut -c1-300
mix_spec=$(IFS=,; echo "${splits[*]}")
echo "ROUND READY: MIX=$mix_spec ($(date +%H:%M))"
