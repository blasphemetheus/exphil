#!/usr/bin/env bash
# AWBC arms B1 / B2 / B3 on fox_gen_v1 — the value-model (D2) entry rung.
#
# Pre-registered in eval_runs/0828_awbc_arms/PREREG.md. Three resumes from
# the SAME checkpoint (ep10), SAME data, SAME seed, SAME epoch budget; the
# only difference is the loss weighting:
#
#   B1  baseline   plain BC continuation            (no --awbc)
#   B2  awbc       outcome-weighted BC              (--awbc --awbc-reward standard)
#   B3  control    the B2 weights, permuted         (--awbc --awbc-reward standard --awbc-shuffle)
#
# B3 exists so "B2 differs from B1" cannot be explained by the weight
# DISTRIBUTION alone (non-uniform weights change the effective batch); a
# real signal needs B2 to beat both B1 and B3.
#
# Runs SEQUENTIALLY (one GPU) inside a single systemd user unit:
#   systemd-run --user --unit=exphil-awbc-arms --working-directory=$PWD \
#     --collect -p StandardOutput=append:$PWD/logs/awbc_arms.log \
#     -p StandardError=append:$PWD/logs/awbc_arms.log \
#     bash scripts/awbc_arms.sh
#
# Waits for any exphil-legS-* unit to finish first (the second-EXLA-client
# law: never start a mix beam next to a live one). DO NOT edit lib/*.ex
# while this chain runs — each arm recompiles.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

RESUME=checkpoints/fox_gen_v1_20260825_210355_epoch10.axon
EPOCHS="${EPOCHS:-3}"
SEED="${SEED:-828}"
OUT=eval_runs/0828_awbc_arms
mkdir -p "$OUT" logs

# fox_gen_v1's non-default recipe (diffed against Config.defaults/0 on
# 2026-08-28: everything else in its _config.json is a default).
COMMON=(
  --backbone gru --temporal --stage-internals
  --hidden-sizes 512,512,256
  --batch-size 256
  --dropout 0.1
  --stream-chunk-size 100
  --replays replays/erickfm_ranked/FOX/extracted
  --resume "$RESUME"
  --epochs "$EPOCHS"
  --seed "$SEED"
)

wait_for_legS() {
  while systemctl --user list-units --type=service --state=active --no-legend 2>/dev/null \
        | grep -q 'exphil-legS-'; do
    echo "[$(date -Is)] waiting for Leg S to finish..."
    sleep 60
  done
}

run_arm() {
  local name="$1"; shift
  local log="logs/awbc_${name}.log"
  echo "=== $name start $(date -Is)" | tee -a "$OUT/TABLE.md"
  devenv shell -- mix run scripts/train.exs "${COMMON[@]}" --name "fox_gen_v1_${name}" "$@" \
    > "$log" 2>&1
  local rc=$?
  echo "=== $name end $(date -Is) rc=$rc" | tee -a "$OUT/TABLE.md"
  # Knob assertion (guard #6 class): read the SAVED config, not the log.
  # train.exs never prints the word "awbc" (the 04:27 B2 "no awbc text" alarm
  # was false); the checkpoint's _config.json is the ground truth.
  local cfg want_awbc want_shuffle
  cfg=$(ls -t checkpoints/fox_gen_v1_${name}_*_config.json 2>/dev/null | head -1)
  case "$name" in B1) want_awbc=false; want_shuffle=false ;; B2) want_awbc=true; want_shuffle=false ;; B3) want_awbc=true; want_shuffle=true ;; esac
  if [ -z "$cfg" ]; then
    echo "!!! $name: no _config.json saved — arm did not reach save" | tee -a "$OUT/TABLE.md"
  elif ! grep -qE "\"awbc\": *\"$want_awbc\"" "$cfg" || ! grep -qE "\"awbc_shuffle\": *\"$want_shuffle\"" "$cfg" \
       || { [ "$want_awbc" = true ] && ! grep -qE "\"awbc_reward\": *\"standard\"" "$cfg"; }; then
    echo "!!! $name: KNOB MISMATCH in $cfg (want awbc=$want_awbc shuffle=$want_shuffle reward=standard): $(grep -oE "\"awbc[a-z_]*\": *\"[^\"]*\"" "$cfg" | tr "\n" " ")" | tee -a "$OUT/TABLE.md"
  else
    echo "    $name knobs OK: $(grep -oE "\"awbc[a-z_]*\": *\"[^\"]*\"" "$cfg" | tr "\n" " ")" | tee -a "$OUT/TABLE.md"
  fi
  return $rc
}

wait_for_legS
echo "arms started $(date -Is) resume=$RESUME epochs=$EPOCHS seed=$SEED" | tee -a "$OUT/TABLE.md"

run_arm B1
run_arm B2 --awbc --awbc-reward standard
run_arm B3 --awbc --awbc-reward standard --awbc-shuffle

echo "=== all arms done $(date -Is)" | tee -a "$OUT/TABLE.md"
ls -la checkpoints/ | grep -E 'fox_gen_v1_B[123]' | tee -a "$OUT/TABLE.md"
