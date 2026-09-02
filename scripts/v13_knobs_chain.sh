#!/usr/bin/env bash
# 09-01 night — v1.3 follow-ups after the primary passed:
#   1. F4 combo-depth probe (Bradley's "how do we show it the deeper
#      layers" — knowledge vs state-visitation fork)
#   2. Full decode-knob ladder RERUN on v1.3-ARrefit with a critic
#      retrained on CLEAN extracts (every previous ladder number was
#      measured on corrupted trunks and/or E1c-contaminated features):
#      extract -> critic train -> Best-of-N in-dist + fresh corpus.
#
#   systemd-run --user --unit=v13-knobs --working-directory=$PWD --collect \
#     -p StandardOutput=append:$PWD/logs/v13_knobs.log \
#     -p StandardError=append:$PWD/logs/v13_knobs.log \
#     devenv shell -- bash scripts/v13_knobs_chain.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

POLICY=checkpoints/fox_gen_v1.3_ARrefit_policy.bin
DATA=cache/critic/v13arrefit_erickfm40_k16_r2.nx
CRIT=eval_runs/0901_critic_v13
CRITIC="$CRIT/critic.nx"
mkdir -p "$CRIT" eval_runs/0901_combo_depth logs cache/critic

while pgrep -x beam.smp >/dev/null 2>&1; do
  echo "[$(date -Is)] waiting for a live beam..."; sleep 60
done

stage() {
  local name="$1"; shift
  echo "=== $name start $(date -Is)"
  if "$@"; then echo "=== $name ok $(date -Is)"; return 0
  else echo "!!! $name FAILED $(date -Is)"; return 1; fi
}

stage f4_combo_depth \
  mix run scripts/combo_depth_probe.exs --policy "$POLICY" \
    --replays 'replays/erickfm_ranked/FOX/extracted/*.slp' \
    --limit-files 24 --k 16 --out eval_runs/0901_combo_depth/RESULTS.md \
  || echo "(continuing to ladder)"

stage critic_extract \
  mix run scripts/critic_extract.exs --policy "$POLICY" \
    --replays 'replays/erickfm_ranked/FOX/extracted/*.slp' \
    --limit-files 40 --k 16 --stride 2 --out "$DATA" || exit 1

stage critic_train \
  mix run scripts/critic_train.exs --data "$DATA" --out "$CRITIC" \
    --report "$CRIT/train.md" || exit 1

stage bestofn_indist \
  mix run scripts/interp_bestofn.exs --policy "$POLICY" --critic "$CRITIC" \
    --replays 'replays/erickfm_ranked/FOX/extracted/*.slp' --n 16 --limit-files 20 \
    --out "$CRIT/bestofn_indist.md" || exit 1

stage bestofn_fresh \
  mix run scripts/interp_bestofn.exs --policy "$POLICY" --critic "$CRITIC" \
    --replays 'replays/fox_il_v1/*.slp' --n 16 --limit-files 20 \
    --out "$CRIT/bestofn_fox_il_v1.md" || exit 1

echo "=== V13 KNOBS CHAIN DONE $(date -Is) — F4: flat-vs-collapse across depth decides drills-vs-curation; ladder: selector margin over mode-of-N vs the >=5 wire-live bar (v1.2 dirty-instrument read was +3.6/+1.0); any winner owes the live gate."
