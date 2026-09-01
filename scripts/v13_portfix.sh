#!/usr/bin/env bash
# v1.3 port-fix retrain (09-01) — rerun of the v1.1 unfreeze recipe on the
# REPAIRED corpus. The 09-01 discovery: --select-character-port resolved the
# imitated port but the streaming loader never passed opponent_port — on the
# 3,429 non-port-1 fox files (~44%) the real opponent was DROPPED from every
# game_state (distance 0) and the embedding saw an all-zero self with the
# imitated fox in the OPPONENT slot. v1.1/v1.2 trained on that. Fixed by
# opponent_port_for + :remap_ports in Streaming.parse_chunk (+ :r2 cache-key
# generation so the corrupted entries are unreachable).
#
# Stages (single unit, sequential):
#   1. v1.3-AR    joint train, AR head, ep10 trunk transplant  (v1.1 recipe)
#   2. v1.3-IND   joint train, IND head re-init                (control)
#   3. head refit on v1.3-AR's frozen trunk -> v1.3_{AR,IND}refit
#      (the v1.2 law: joint training atrophies the AR wire)
#   4. coincidence probe (L_cond — want ARrefit ~2.5+, controls ~1)
#   5. arhead_score (live gate: died%, routes)
#   6. position probe --sweep (approach_delta by distance — the F3 metric
#      this retrain targets: v1.1/v1.2 sat at -0.20..-0.26)
#
# Pre-registered readings (vs v1.2-ARrefit's card):
#   - PRIMARY: approach_delta becomes materially less negative (the
#     port-2 44% no longer teaches scrambled-slot movement).
#   - GUARD: live recovery kept (died% ~28 class), L_cond restored after
#     refit, val loss reported per arm.
#   - The v1.1->v1.3 delta isolates the LOADER fix (same recipe, corpus,
#     seed, epochs).
#
#   systemd-run --user --unit=v13-portfix --working-directory=$PWD --collect \
#     -p StandardOutput=append:$PWD/logs/v13_portfix.log \
#     -p StandardError=append:$PWD/logs/v13_portfix.log \
#     devenv shell -- bash scripts/v13_portfix.sh
#
# DO NOT edit lib/*.ex while this runs (multi-stage, L7).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

RESUME=checkpoints/fox_gen_v1_20260825_210355_epoch10.axon
EPOCHS="${EPOCHS:-3}"
SEED="${SEED:-828}"
OUT=eval_runs/0901_v13_portfix
mkdir -p "$OUT" logs

# GOTCHA #106 budget: ~40 GB fresh :r2 cache for non-p1 files + ~19 GB refit
# capture. Refuse under 75 GB free.
if [ "$(df --output=avail / | tail -1)" -lt 75000000 ]; then
  echo "!!! under 75 GB free on / — not starting (GOTCHA #106)"; exit 1
fi

COMMON=(
  --backbone gru --temporal --stage-internals
  --hidden-sizes 512,512,256
  --batch-size 256
  --dropout 0.1
  --stream-chunk-size 100
  --replays replays/erickfm_ranked/FOX/extracted
  --train-character fox --select-character-port
  --resume "$RESUME"
  --epochs "$EPOCHS"
  --seed "$SEED"
)

wait_for_beams() {
  while pgrep -x beam.smp >/dev/null 2>&1; do
    echo "[$(date -Is)] waiting for a live beam to finish..."
    sleep 60
  done
}

run_arm() {
  local name="$1"; shift
  local log="logs/v13_${name}.log"
  echo "=== v1.3-$name start $(date -Is)" | tee -a "$OUT/TABLE.md"
  mix run scripts/train.exs "${COMMON[@]}" --name "fox_gen_v1.3_${name}" "$@" \
    > "$log" 2>&1
  local rc=$?
  echo "=== v1.3-$name end $(date -Is) rc=$rc" | tee -a "$OUT/TABLE.md"

  local cfg
  cfg=$(ls -t "checkpoints/fox_gen_v1.3_${name}"*_config.json 2>/dev/null | head -1)
  if [ -z "$cfg" ]; then
    echo "!!! $name: no _config.json saved — arm did not reach save" | tee -a "$OUT/TABLE.md"
    return 1
  fi
  local want_head
  case "$name" in AR) want_head=autoregressive ;; IND) want_head=independent ;; esac
  if ! grep -qE "\"head\": *\"$want_head\"" "$cfg"; then
    echo "!!! $name: HEAD MISMATCH in $cfg" | tee -a "$OUT/TABLE.md"
  fi
  if ! grep -q "TRUNK TRANSPLANT" "$log"; then
    echo "!!! $name: no TRUNK TRANSPLANT banner in $log" | tee -a "$OUT/TABLE.md"
  fi
  grep -E "val_loss|Validation" "$log" | tail -3 | tee -a "$OUT/TABLE.md"
  return $rc
}

wait_for_beams
echo "v1.3 portfix started $(date -Is) resume=$RESUME epochs=$EPOCHS seed=$SEED" | tee -a "$OUT/TABLE.md"

run_arm AR --head autoregressive || echo "!!! AR arm failed — continuing to IND for the record"
wait_for_beams
run_arm IND --reinit-head || echo "!!! IND arm failed"
wait_for_beams

AR_SRC=$(ls -t checkpoints/fox_gen_v1.3_AR_*_policy.bin 2>/dev/null | head -1)
if [ -z "$AR_SRC" ]; then
  echo "!!! no v1.3_AR policy — stopping before refit" | tee -a "$OUT/TABLE.md"; exit 1
fi

AR_OUT=checkpoints/fox_gen_v1.3_ARrefit_policy.bin
IND_OUT=checkpoints/fox_gen_v1.3_INDrefit_policy.bin

echo "=== refit start $(date -Is) src=$AR_SRC" | tee -a "$OUT/TABLE.md"
mix run scripts/train_ar_head.exs \
  --policy "$AR_SRC" \
  --replays 'replays/erickfm_ranked/FOX/extracted/*.slp' \
  --char-id 2 --limit-files 1000 --head both \
  --out-ar "$AR_OUT" --out-ind "$IND_OUT" || echo "!!! refit FAILED" | tee -a "$OUT/TABLE.md"

wait_for_beams
echo "=== coincidence probe $(date -Is)" | tee -a "$OUT/TABLE.md"
mix run scripts/coincidence_probe.exs \
  --set v13_ARrefit="$AR_OUT" --set v13_INDrefit="$IND_OUT" --set v13_AR="$AR_SRC" \
  --replays 'replays/erickfm_ranked/FOX/extracted/*.slp' \
  --limit-files 8 --out "$OUT/coincidence_probe.md" || echo "!!! probe FAILED" | tee -a "$OUT/TABLE.md"

wait_for_beams
echo "=== arhead_score $(date -Is)" | tee -a "$OUT/TABLE.md"
OUT="$OUT/score" AR_POLICY="$AR_OUT" IND_POLICY="$IND_OUT" \
  bash scripts/arhead_score.sh || echo "!!! score FAILED" | tee -a "$OUT/TABLE.md"

wait_for_beams
echo "=== position probe sweep $(date -Is)" | tee -a "$OUT/TABLE.md"
mix run scripts/probe_position_dependence.exs \
  --set v13_ARrefit="$AR_OUT" --set v13_INDrefit="$IND_OUT" \
  --replays 'eval_runs/0831_livelook_v12ar/2026-08-Mainline/*.slp' \
  --player-port 1 --situation neutral --sweep "15,25,40,60,90,130" \
  --out "$OUT/position_sweep.md" || echo "!!! position probe FAILED" | tee -a "$OUT/TABLE.md"

echo "=== V1.3 PORTFIX CHAIN DONE $(date -Is) — read order: TABLE.md (val losses), coincidence_probe.md (L_cond ARrefit ~2.5+), score/ (died% ~28 class), position_sweep.md (approach_delta vs v1.2's -0.20..-0.26 = THE pre-registered primary)" | tee -a "$OUT/TABLE.md"
