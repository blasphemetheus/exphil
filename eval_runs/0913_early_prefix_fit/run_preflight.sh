#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if pgrep -x beam.smp >/dev/null; then echo "A BEAM is live; wait." >&2; exit 1; fi
OUT=eval_runs/0913_early_prefix_fit
test ! -e "$OUT/preflight.log"
export EXPHIL_SKIP_NIF_COMPILE=1 EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.15
mix run --no-compile --no-deps-check \
  -r lib/exphil/networks/policy/execution_contract.ex \
  -r lib/exphil/training/labels.ex -r lib/exphil/training/data.ex \
  -r lib/exphil/training/recorded_frames.ex -r lib/exphil/training/recorded_prefix_sampling.ex \
  -r lib/exphil/training/epoch_loss.ex -r lib/exphil/training/epoch_health.ex \
  -r lib/exphil/training/probe_regularizer.ex \
  scripts/dagger_drill.exs --expert multishine \
  --recurrent-state legacy-random --precision bf16 \
  --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
  --recorded-frames 'eval_runs/0913_teacher_ingestion/validated/*.frames' \
  --recorded-prefix-weight 64 --recorded-prefix-frames 18 \
  --init-from eval_runs/0913_tiny_overfit/initial.bin \
  --hidden-size 64 --window 16 --action-delay 2 --multi-delay 2 \
  --with-delay-id --queue-depth 3 --prev-action --head autoregressive \
  --clean-loss --max-epochs 40 --target-loss 0.0 --preflight \
  --out "$OUT/preflight.bin" > "$OUT/preflight.log" 2>&1
