#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if pgrep -x beam.smp >/dev/null; then echo "A BEAM is live; wait." >&2; exit 1; fi
OUT=eval_runs/0913_epoch_metric_validation
MODE="${1:?normal, forensics, omitted, or disabled}"
NAME="${2:-$MODE}"
case "$MODE" in normal) EXTRA=(--snapshot-all);; forensics) EXTRA=(--snapshot-all --nan-forensics);; omitted) EXTRA=();; disabled) EXTRA=(--no-snapshot-all);; *) exit 1;; esac
test ! -e "$OUT/$NAME.log"
export EXPHIL_SKIP_NIF_COMPILE=1 EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.15
mix run --no-compile --no-deps-check \
  -r lib/exphil/training/labels.ex -r lib/exphil/training/data.ex \
  -r lib/exphil/training/probe_regularizer.ex -r lib/exphil/training/epoch_loss.ex \
  -r lib/exphil/training/recorded_frames.ex -r lib/exphil/training/epoch_health.ex \
  scripts/dagger_drill.exs --expert multishine \
  --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
  --recorded-frames 'eval_runs/0913_teacher_ingestion/validated/*.frames' \
  --init-from eval_runs/0913_tiny_overfit/initial.bin \
  --hidden-size 64 --window 16 --action-delay 2 --multi-delay 2 \
  --with-delay-id --queue-depth 3 --prev-action --head autoregressive \
  --clean-loss --max-epochs 1 --target-loss 0.0 "${EXTRA[@]}" \
  --out "$OUT/$NAME.bin" > "$OUT/$NAME.log" 2>&1
test -s "$OUT/$NAME.bin"
case "$MODE" in
  omitted|disabled) test ! -e "$OUT/${NAME}_ep1.bin";;
  normal|forensics) test -s "$OUT/${NAME}_ep1.bin";;
esac
