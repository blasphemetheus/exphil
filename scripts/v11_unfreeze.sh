#!/usr/bin/env bash
# v1.1 unfreeze — AUTOREGRESSIVE_HEAD_PLAN items 8/9, on the FIXED fox corpus
# (Bradley's corpus decision, 2026-08-30; E1 fix via --select-character-port).
#
#   v1.1-AR   resume ep10 trunk, AR head from scratch      (--head autoregressive)
#   v1.1-IND  resume ep10 trunk, IND head re-initialised   (--reinit-head)
#
# Both arms: B1's recipe (3 epochs, seed 828) + --train-character fox
# --select-character-port. The trunk-transplant resume (commit f92aa34) loads
# matching non-head params, keeps head + optimizer fresh, keeps --head.
# The AR-vs-IND pair is the pre-registered comparison; the v1->v1.1 delta
# deliberately confounds head+corpus (recorded on the status board).
#
# Launch (single unit, arms sequential):
#   systemd-run --user --unit=exphil-v11-unfreeze --working-directory=$PWD \
#     --collect -p StandardOutput=append:$PWD/logs/v11_unfreeze.log \
#     -p StandardError=append:$PWD/logs/v11_unfreeze.log \
#     bash scripts/v11_unfreeze.sh
#
# DO NOT edit lib/*.ex while this runs — each arm recompiles (L7).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

RESUME=checkpoints/fox_gen_v1_20260825_210355_epoch10.axon
EPOCHS="${EPOCHS:-3}"
SEED="${SEED:-828}"
OUT=eval_runs/0831_v11_unfreeze
mkdir -p "$OUT" logs

# fox_gen_v1's non-default recipe (B1's COMMON block, awbc_arms.sh) + the
# corpus fix. stream-chunk-size 100 = v1's chunking; the port-aware cache
# keys mean this builds a FRESH embedding cache on the first arm and the
# second arm rides it.
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
  local log="logs/v11_${name}.log"
  echo "=== v1.1-$name start $(date -Is)" | tee -a "$OUT/TABLE.md"
  devenv shell -- mix run scripts/train.exs "${COMMON[@]}" --name "fox_gen_v1.1_${name}" "$@" \
    > "$log" 2>&1
  local rc=$?
  echo "=== v1.1-$name end $(date -Is) rc=$rc" | tee -a "$OUT/TABLE.md"

  # Knob assertions (guard #6 class): the SAVED _config.json is ground truth.
  local cfg
  cfg=$(ls -t "checkpoints/fox_gen_v1.1_${name}"*_config.json 2>/dev/null | head -1)
  if [ -z "$cfg" ]; then
    echo "!!! $name: no _config.json saved — arm did not reach save" | tee -a "$OUT/TABLE.md"
    return 1
  fi
  local want_head
  case "$name" in AR) want_head=autoregressive ;; IND) want_head=independent ;; esac
  if ! grep -qE "\"head\": *\"$want_head\"" "$cfg"; then
    echo "!!! $name: HEAD MISMATCH in $cfg (want $want_head): $(grep -oE '"head": *"[^"]*"' "$cfg")" | tee -a "$OUT/TABLE.md"
  elif ! grep -qE "\"select_character_port\": *\"?true\"?" "$cfg"; then
    echo "!!! $name: select_character_port NOT true in $cfg" | tee -a "$OUT/TABLE.md"
  else
    echo "    $name knobs OK: $(grep -oE '"(head|select_character_port|train_character|reinit_head)": *"[^"]*"' "$cfg" | tr '\n' ' ')" | tee -a "$OUT/TABLE.md"
  fi
  # Transplant assertion: the run log must show the trunk transplant banner
  # (the resume path that silently full-loads would train the WRONG head).
  if ! grep -q "TRUNK TRANSPLANT" "$log"; then
    echo "!!! $name: no TRUNK TRANSPLANT banner in $log — resume took the full-load path?" | tee -a "$OUT/TABLE.md"
  fi
  return $rc
}

wait_for_beams
echo "v1.1 unfreeze started $(date -Is) resume=$RESUME epochs=$EPOCHS seed=$SEED" | tee -a "$OUT/TABLE.md"

run_arm AR --head autoregressive
wait_for_beams
run_arm IND --reinit-head

echo "=== v1.1 unfreeze COMPLETE $(date -Is)" | tee -a "$OUT/TABLE.md"
