#!/usr/bin/env bash
# (a) of the 08-31 critic plan: offline critic/Best-of-N ladder on v1.1-AR
# with COHERENT candidates (sample_autoregressive_kn — the 08-29 ladder
# ranked independent-head chimeras). k=16 so the ladder reads against
# mode-of-16's 22.9 offline pass@1 and the Leg S AR numbers.
#
#   systemd-run --user --unit=critic-ar --working-directory=$PWD --collect \
#     -p StandardOutput=append:$PWD/logs/critic_ar.log \
#     -p StandardError=append:$PWD/logs/critic_ar.log \
#     devenv shell -- bash scripts/critic_ar_chain.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

POLICY=checkpoints/fox_gen_v1.1_AR_20260831_080100_policy.bin
DATA=cache/critic/v11ar_erickfm40_k16.nx
CRIT=eval_runs/0831_critic_ar
CRITIC="$CRIT/critic.nx"
mkdir -p "$CRIT" logs cache/critic

stage() {
  local name="$1"; shift
  echo "=== $name start $(date -Is)"
  if "$@"; then echo "=== $name ok $(date -Is)"; return 0
  else echo "!!! $name FAILED $(date -Is)"; return 1; fi
}

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

echo "=== CRITIC-AR CHAIN DONE $(date -Is) — read against: mode-of-16 offline 22.9 (ep10, independent candidates), Leg S AR pass@16 38.6/headroom +30.1; any winner still owes the LIVE gate (frozen-input <= 0.20, 7/8 to cap) before belief."
