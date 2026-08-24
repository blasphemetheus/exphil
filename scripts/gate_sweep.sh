#!/usr/bin/env bash
# Behavioral gate-sweep over per-epoch snapshots (2026-08-20 recipe).
# Usage: gate_sweep.sh <snapshot-prefix> <out-dir> [--confirm]
#   e.g. gate_sweep.sh checkpoints/ms_g19 eval_runs/0820_g19_gatesweep/sweep
# Gates every <prefix>_ep*.bin with ONE sync stand-fox d3 run (~30s;
# deterministic on FD), prints a per-epoch table, names the argmax.
# With --confirm: re-gates the argmax x3 fox + x1 mewtwo.
# Run AFTER training only (spawns beams; NO-MIX discipline).
set -euo pipefail
cd "$(dirname "$0")/.."
PREFIX="${1:?usage: gate_sweep.sh <snapshot-prefix> <out-dir> [--confirm]}"
OUTDIR="${2:?usage: gate_sweep.sh <snapshot-prefix> <out-dir> [--confirm]}"
CONFIRM="${3:-}"

export DOLPHIN_DIR="${DOLPHIN_DIR:-$HOME/.local/share/slippi/exi-ai/dolphin-emu-headless}"
export ISO="${ISO:-$HOME/isos/melee.iso}"
mkdir -p "$OUTDIR"
TABLE="$OUTDIR/sweep_table.txt"
: > "$TABLE"

snaps=$(ls "${PREFIX}"_ep*.bin 2>/dev/null | sort -t_ -k2 -V) || true
[ -n "$snaps" ] || { echo "no snapshots at ${PREFIX}_ep*.bin" >&2; exit 1; }
echo "sweeping $(echo "$snaps" | wc -l) snapshots"

best_rate=-1; best_snap=""
for snap in $snaps; do
  ep=$(basename "$snap" | grep -oE "ep[0-9]+" | tr -d 'ep')

  # Resumable (2026-08-21): skip epochs already scored in the table, so
  # a re-run after an interruption only gates what's missing.
  if grep -q "^ep${ep}:" "$TABLE" 2>/dev/null; then
    line=$(grep "^ep${ep}:" "$TABLE" | tail -1)
    rate=$(echo "$line" | sed -E 's/^ep[0-9]+: ([0-9.]+)\/min.*/\1/')
    better=$(awk -v a="$rate" -v b="$best_rate" 'BEGIN{print (a>b)?1:0}')
    if [ "$better" = "1" ]; then best_rate="$rate"; best_snap="$snap"; fi
    continue
  fi

  dir="$OUTDIR/ep${ep}"
  EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
    "$snap" "$dir" --runs 1 --dummy stand --runner sync \
    -- --frame-delay 3 --delay-id-override 3 --headless --emulation-speed 0 --blocking-input --slippi-port 51442 \
    > "$dir.log" 2>&1 || { echo "ep${ep} GATE FAILED" | tee -a "$TABLE"; continue; }
  # `|| true` everywhere: under set -e a no-match grep in a $() assignment
  # killed the whole sweep at ep32 of the f3_a2 run (2026-08-21) — a
  # missing score line must cost one epoch, never the sweep.
  line=$(grep -aE "^\[[0-9:]+\] r1 " "$dir.log" | tail -1 || true)
  [ -n "$line" ] || { echo "ep${ep} GATE FAILED (no score line)" | tee -a "$TABLE"; continue; }
  rate=$(echo "$line" | awk '{print $(NF-1)}')
  chain=$(echo "$line" | awk '{print $NF}')
  echo "ep${ep}: ${rate}/min chain ${chain}" | tee -a "$TABLE"
  better=$(awk -v a="$rate" -v b="$best_rate" 'BEGIN{print (a>b)?1:0}')
  if [ "$better" = "1" ]; then best_rate="$rate"; best_snap="$snap"; fi
done

echo "=== ARGMAX: $best_snap at ${best_rate}/min" | tee -a "$TABLE"

if [ "$CONFIRM" = "--confirm" ] && [ -n "$best_snap" ]; then
  echo "=== confirming argmax x3 fox + mewtwo"
  EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
    "$best_snap" "$OUTDIR/argmax_fox" --runs 3 --dummy stand --runner sync \
    -- --frame-delay 3 --delay-id-override 3 --headless --emulation-speed 0 --blocking-input --slippi-port 51442 \
    2>&1 | grep -aE "r[123] " | tail -3 | tee -a "$TABLE"
  EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
    "$best_snap" "$OUTDIR/argmax_mewtwo" --runs 1 --dummy stand --runner sync \
    -- --frame-delay 3 --delay-id-override 3 --dummy-character mewtwo --headless --emulation-speed 0 --blocking-input --slippi-port 51442 \
    2>&1 | grep -aE "r1 " | tail -1 | tee -a "$TABLE"
fi
echo "=== SWEEP DONE. Table: $TABLE"