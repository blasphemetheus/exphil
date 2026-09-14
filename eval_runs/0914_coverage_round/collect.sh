#!/usr/bin/env bash
# Coverage round, stage 1: ROLLOUTS that contain held-out starts, positions,
# facings and opponents. The delay-4 proof candidate plays at reaction 4 on
# the async runner vs a level-1 CPU of each roster character, 2 x 90 s each.
# These replays are handoff SOURCES only (the teacher re-executes every
# response from the handoff on; nothing here becomes a label directly).
set -uo pipefail
cd "$(dirname "$0")/../.."
if pgrep -x beam.smp >/dev/null; then echo "A BEAM is live; wait." >&2; exit 1; fi
OUT=eval_runs/0914_coverage_round/rollouts
POLICY=eval_runs/0914_delay4_proof/round21/candidate.bin
mkdir -p "$OUT"
P=eval_runs/0914_coverage_round/progress.log
say() { echo "[$(date +%T)] $*" | tee -a "$P"; }
say "=== stage 1: rollouts vs CPU roster (reaction 4, T=1.0, 2 x 90 s each)"
for char in ${ROSTER:-marth falco peach samus fox}; do
  [[ -d $OUT/$char ]] && { say "  $char: exists, skipping"; continue; }
  EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh "$POLICY" "$OUT/$char" \
    --runs 2 --seconds 90 --dummy cpu --runner async --temperature 1.0 \
    -- --reaction-delay 4 --headless --emulation-speed 0 --blocking-input --slippi-port 51442 \
       --dummy-character "$char" > "$OUT/$char.log" 2>&1
  say "  $char: exit $? ; $(grep -aE '^\[[0-9:]+\] r[12] ' "$OUT/$char.log" | sed 's/^\[[0-9:]*\] //' | tr -s ' ' | tr '\n' ';')"
done
say "stage 1 done"
