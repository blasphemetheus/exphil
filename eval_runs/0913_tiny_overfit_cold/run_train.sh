#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if pgrep -x beam.smp >/dev/null; then echo "A BEAM is live; wait." >&2; exit 1; fi
OUT=eval_runs/0913_tiny_overfit_cold
test ! -e "$OUT/train.log"
test ! -e "$OUT/candidate.bin"
export EXPHIL_SKIP_NIF_COMPILE=1 EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.15
mix run --no-compile --no-deps-check \
  -r lib/exphil/training/labels.ex -r lib/exphil/training/data.ex \
  -r lib/exphil/training/epoch_loss.ex \
  -r lib/exphil/training/probe_regularizer.ex \
  -r lib/exphil/training/recorded_frames.ex -r lib/exphil/training/epoch_health.ex \
  scripts/dagger_drill.exs --expert multishine \
  --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
  --recorded-frames 'eval_runs/0913_teacher_ingestion/validated/*.frames' \
  --init-from eval_runs/0913_tiny_overfit/initial.bin --initial-out "$OUT/initial.bin" \
  --hidden-size 64 --window 16 --action-delay 2 --multi-delay 2 \
  --with-delay-id --queue-depth 3 --prev-action --head autoregressive \
  --clean-loss --max-epochs 40 --target-loss 0.0 --snapshot-all \
  --out "$OUT/candidate.bin" > "$OUT/train.log" 2>&1
