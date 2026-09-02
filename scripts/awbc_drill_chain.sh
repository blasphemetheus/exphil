#!/usr/bin/env bash
# AWBC drill arm (DRILL_HITCONFIRM training half + HANDOFF task 4, 09-02).
# One knob: the v1.3 recipe + --awbc standard + the sliced Drill 1 windows
# (drills/drill1_hitconfirm.frames, 538 windows / 129,658 frames — NEVER
# the raw drill .slp files; see drill.json training block). Then the
# pre-registered re-measure: fresh drill episodes with the retrained
# policy, scored against the same expert cell.
#
# Gate for "the drill works" (DRILL_HITCONFIRM, pre-registered): the
# hitstun-linked chain-length distribution on FRESH drill episodes shifts
# toward the expert's (baseline: 2.9 mean / 30% deep / 16.4 dmg at 0-19;
# expert 3.9 / 87 / 27.0). Then a live-look transfer check (Bradley).
#
#   systemd-run --user --unit=awbc-drill --working-directory=$PWD --collect \
#     -p StandardOutput=append:$PWD/logs/awbc_drill.log \
#     -p StandardError=append:$PWD/logs/awbc_drill.log \
#     devenv shell -- bash scripts/awbc_drill_chain.sh
#
# DO NOT edit lib/*.ex while this runs (multi-stage, L7).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

RESUME="${RESUME:-checkpoints/fox_gen_v1.3_AR_20260901_194518_epoch3.axon}"
EPOCHS="${EPOCHS:-2}"
SEED="${SEED:-902}"
NAME="${NAME:-fox_gen_v1.3_AWBCdrill}"
MIX="${MIX:-drills/drill1_hitconfirm.frames}"
OVERSAMPLE="${OVERSAMPLE:-20}"
OUT=eval_runs/0902_awbc_drill
mkdir -p "$OUT" logs

if [ "$(df --output=avail / | tail -1)" -lt 30000000 ]; then
  echo "!!! under 30 GB free on / — not starting (GOTCHA #106; caches should HIT)"; exit 1
fi

echo "=== AWBC drill arm start $(date -Is) resume=$RESUME epochs=$EPOCHS mix=$MIX x$OVERSAMPLE" | tee -a "$OUT/TABLE.md"

mix run scripts/train.exs \
  --backbone gru --temporal --stage-internals \
  --hidden-sizes 512,512,256 \
  --batch-size 256 \
  --dropout 0.1 \
  --stream-chunk-size 100 \
  --replays replays/erickfm_ranked/FOX/extracted \
  --train-character fox --select-character-port \
  --resume "$RESUME" \
  --epochs "$EPOCHS" \
  --seed "$SEED" \
  --head autoregressive \
  --awbc --awbc-reward standard \
  --mix-frames "$MIX" --mix-oversample "$OVERSAMPLE" \
  --name "$NAME" \
  > "logs/train_${NAME}.log" 2>&1
rc=$?
echo "=== train end $(date -Is) rc=$rc" | tee -a "$OUT/TABLE.md"

# Assert the mix actually entered the stream (flag-drop guard #6)
if ! grep -qa "curriculum mix: 129" "logs/train_${NAME}.log"; then
  echo "!!! MIX ASSERTION FAILED: streaming curriculum-mix line absent — the drill windows never entered training" | tee -a "$OUT/TABLE.md"
fi

POLICY=$(ls -t "checkpoints/${NAME}"*_policy.bin 2>/dev/null | head -1)
if [ -z "$POLICY" ]; then
  echo "!!! no policy artifact — stopping before re-measure" | tee -a "$OUT/TABLE.md"; exit 1
fi
echo "policy: $POLICY" | tee -a "$OUT/TABLE.md"

# Pre-registered re-measure: fresh drill episodes, same cell, same scorer.
echo "=== drill re-measure start $(date -Is)" | tee -a "$OUT/TABLE.md"
mix run scripts/drill_episode.exs \
  --policy "$POLICY" \
  --episodes 150 --out "$OUT/drill_remeasure" \
  >> "$OUT/TABLE.md" 2>&1
echo "=== drill re-measure end $(date -Is)" | tee -a "$OUT/TABLE.md"

mix run scripts/drill_score.exs --bank "$OUT/drill_remeasure" \
  --out "$OUT/drill_remeasure/RESULTS.md" >> "$OUT/TABLE.md" 2>&1

echo "AWBC DRILL CHAIN DONE $(date -Is) — compare $OUT/drill_remeasure/RESULTS.md vs the 538-ep baseline (2.9/30/16.4), then live-look transfer (g6 rule)" | tee -a "$OUT/TABLE.md"
