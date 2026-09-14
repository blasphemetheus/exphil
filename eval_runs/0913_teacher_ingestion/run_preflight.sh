#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if pgrep -x beam.smp >/dev/null; then
  echo "A BEAM is live; wait before preflight." >&2
  exit 1
fi
NAME="${1:?Pass a fresh output name}"
OUT=eval_runs/0913_teacher_ingestion
test ! -e "$OUT/$NAME.log"
export EXPHIL_SKIP_NIF_COMPILE=1 EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.15
mix run --no-compile --no-deps-check \
  -r lib/exphil/training/labels.ex -r lib/exphil/training/data.ex \
  -r lib/exphil/training/epoch_loss.ex \
  -r lib/exphil/training/recorded_frames.ex -r lib/exphil/training/epoch_health.ex \
  scripts/dagger_drill.exs --expert multishine \
  --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
  --recorded-frames "$OUT/validated/*.frames" \
  --hidden-size 64 --window 16 --action-delay 2 --multi-delay 2 \
  --with-delay-id --queue-depth 3 --prev-action --head autoregressive \
  --clean-loss --max-epochs 1 --preflight --out "$OUT/${NAME}_policy.bin" \
  > "$OUT/$NAME.log" 2>&1
