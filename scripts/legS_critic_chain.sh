#!/usr/bin/env bash
# Leg S calibration (task 12) + D2 critic shakedown (task 14), sequential on
# one GPU — HANDOFF_2026-08-29 §6 steps 1 and 3, verbatim commands.
#
# Order: cheap-and-informative first.
#   1. Leg S ep1/ep5/ep10 on erickfm FOX (calibration: pass@1 must be roughly
#      FLAT across epochs, else the +29 pts is an artifact of the instrument)
#   2. Leg S ep10 on fox_il_v1 (second corpus, auto-port)
#   3. critic_extract -> critic_train -> interp_bestofn (runbook in
#      docs/planning/CRITIC_D2_DESIGN.md; first run = its own shakedown)
#
# Each stage is a fresh `mix run`; nothing here edits lib/. Every stage's
# rc is appended to $OUT/CHAIN.md; a failed stage does NOT stop the chain
# (later stages are independent) except critic_train/bestofn, which need
# the previous stage's artifact.
#
#   systemd-run --user --unit=legS-critic --collect \
#     --working-directory=/home/blewf/git/exphil \
#     -p StandardOutput=append:/home/blewf/git/exphil/logs/legS_critic_chain.log \
#     -p StandardError=append:/home/blewf/git/exphil/logs/legS_critic_chain.log \
#     devenv shell -- bash scripts/legS_critic_chain.sh
set -uo pipefail
cd "$(dirname "$0")/.."

CK=checkpoints/fox_gen_v1_20260825_210355
LEGS=eval_runs/0828_legS
CRIT=eval_runs/0829_critic
OUT=eval_runs/0829_legS_critic_chain
mkdir -p "$LEGS" "$CRIT" "$OUT" cache/critic logs
CHAIN="$OUT/CHAIN.md"
DONE_MARKER="$OUT/.chain_done"
rm -f "$DONE_MARKER"

if pgrep -x beam.smp >/dev/null; then
  echo "a beam is already live — refusing (second-EXLA-client law)" >&2; exit 3
fi

stage() {  # stage <name> <logfile> <cmd...>
  local name="$1" log="$2"; shift 2
  echo "=== $name start $(date -Is)" | tee -a "$CHAIN"
  "$@" > "$log" 2>&1
  local rc=$?
  echo "=== $name end $(date -Is) rc=$rc log=$log" | tee -a "$CHAIN"
  return $rc
}

echo "chain started $(date -Is)" | tee -a "$CHAIN"

# 1. Leg S calibration — erickfm FOX, port 1, n=16, T=0.5, same seed for all
for ep in 1 5 10; do
  stage "legS_ep$ep" "logs/legS_cal_ep$ep.log" \
    mix run scripts/interp_passk.exs --policy "${CK}_ep$ep.bin" \
      --replays 'replays/erickfm_ranked/FOX/extracted/*.slp' --port 1 --n 16 --seed 829 \
      --limit-files 20 --limit-frames 2000 --out "$LEGS/ep$ep.md"
done

# 2. Second corpus (auto-port)
stage "legS_ep10_fox_il_v1" "logs/legS_cal_ep10_fox_il_v1.log" \
  mix run scripts/interp_passk.exs --policy "${CK}_ep10.bin" \
    --replays 'replays/fox_il_v1/*.slp' --n 16 --seed 829 \
    --limit-files 20 --limit-frames 2000 --out "$LEGS/ep10_fox_il_v1.md"

# 3. Critic shakedown
DATA=cache/critic/fox_gen_v1_ep10_erickfm40.nx
CRITIC=checkpoints/critic_fox_gen_v1_ep10.bin
if stage "critic_extract" "logs/critic_extract.log" \
     mix run scripts/critic_extract.exs --policy "${CK}_ep10.bin" \
       --replays 'replays/erickfm_ranked/FOX/extracted/*.slp' --port 1 --limit-files 40 \
       --out "$DATA"; then
  if stage "critic_train" "logs/critic_train.log" \
       mix run scripts/critic_train.exs --data "$DATA" --out "$CRITIC" --report "$CRIT/train.md"; then
    stage "bestofn_fox_il_v1" "logs/critic_bestofn.log" \
      mix run scripts/interp_bestofn.exs --policy "${CK}_ep10.bin" --critic "$CRITIC" \
        --replays 'replays/fox_il_v1/*.slp' --n 16 --out "$CRIT/bestofn_fox_il_v1.md"
  else
    echo "critic_train failed — bestofn skipped" | tee -a "$CHAIN"
  fi
else
  echo "critic_extract failed — train/bestofn skipped" | tee -a "$CHAIN"
fi

echo "CHAIN DONE $(date -Is)" | tee -a "$CHAIN"
touch "$DONE_MARKER"
