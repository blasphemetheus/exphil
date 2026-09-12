#!/usr/bin/env bash
# Behavioral gate-sweep over per-epoch snapshots (2026-08-20 recipe).
# Usage: gate_sweep.sh <snapshot-prefix> <out-dir> [--confirm]
#   e.g. gate_sweep.sh checkpoints/ms_g19 eval_runs/0820_g19_gatesweep/sweep
# Gates every <prefix>_ep*.bin with ONE sync stand-fox run (~30s; deterministic
# on FD) at GATE_DELAY (default 3), prints a per-epoch table, names the argmax.
# Env (2026-09-10): GATE_DELAY=N sets --frame-delay N; GATE_ID=N adds
# --delay-id-override N (default: NONE — the Agent derives the id from the
# checkpoint's label convention, INVARIANTS item 1; pass GATE_ID=3 for the
# legacy g19-era behavior). EPOCHS="4 5 6" restricts the sweep (chunking).
# With --confirm: re-gates the argmax x3 fox + x1 mewtwo.
# Run AFTER training only (spawns beams; NO-MIX discipline).
set -euo pipefail
cd "$(dirname "$0")/.."
PREFIX="${1:?usage: gate_sweep.sh <snapshot-prefix> <out-dir> [--confirm]}"
OUTDIR="${2:?usage: gate_sweep.sh <snapshot-prefix> <out-dir> [--confirm]}"
CONFIRM="${3:-}"

# 2026-09-10: FORCE the exi-ai headless build. DOLPHIN_DIR is exported globally
# in the login shell (fish) pointing at the NETPLAY AppImage, which needs a
# DISPLAY even with --headless ("Unable to initialize GTK+") — every gate
# from 11:27 failed with a console-connect timeout and no Dolphin process.
# Override deliberately with GATE_DOLPHIN_DIR.
export DOLPHIN_DIR="${GATE_DOLPHIN_DIR:-$HOME/.local/share/slippi/exi-ai/dolphin-emu-headless}"
export ISO="${ISO:-$HOME/isos/melee.iso}"
# INVARIANTS item 12: ONE knob. GATE_REACTION=k plays physical reaction delay k
# (default 4 = the local rung: drill ids 2..5 physical, or 0..3 + the retired
# pipeline offset 2 for pre-09-12 checkpoints). GATE_DELAY=N (deprecated
# --frame-delay alias) still works: k = N + 1.
GATE_REACTION="${GATE_REACTION:-$(( ${GATE_DELAY:-3} + 1 ))}"
# 2026-09-10 (Bradley): argmax gating was the wrong instrument — sampling at
# T=1.0 is the decode the policies are trained for (the fox_gen argmax-collapse
# lesson; the AR head especially). GATE_TEMP=0 restores --deterministic.
GATE_TEMP="${GATE_TEMP:-1.0}"
if [ "$GATE_TEMP" = "0" ]; then DECODE_ARGS=""; else DECODE_ARGS="--temperature $GATE_TEMP"; fi
GATE_ID_ARGS=""
[ -n "${GATE_ID:-}" ] && GATE_ID_ARGS="--delay-id-override $GATE_ID"
mkdir -p "$OUTDIR"
TABLE="$OUTDIR/sweep_table.txt"
: > "$TABLE"

snaps=$(ls "${PREFIX}"_ep*.bin 2>/dev/null | sort -t_ -k2 -V) || true
[ -n "$snaps" ] || { echo "no snapshots at ${PREFIX}_ep*.bin" >&2; exit 1; }
echo "sweeping $(echo "$snaps" | wc -l) snapshots"

best_rate=-1; best_snap=""
for snap in $snaps; do
  ep=$(basename "$snap" | grep -oE "ep[0-9]+" | tr -d 'ep')
  if [ -n "${EPOCHS:-}" ] && ! echo " $EPOCHS " | grep -q " $ep "; then continue; fi

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
    "$snap" "$dir" --runs 1 --dummy stand --runner sync $DECODE_ARGS \
    -- --reaction-delay "$GATE_REACTION" $GATE_ID_ARGS --headless --emulation-speed 0 --blocking-input --slippi-port 51442 \
    > "$dir.log" 2>&1 || {
      echo "ep${ep} GATE FAILED" | tee -a "$TABLE"
      # GUARDS_BACKLOG #3: N consecutive failures = infrastructure,
      # not policy (0824: 60x GATE FAILED -> "argmax -1" marched on).
      CONSEC_FAIL=$(( ${CONSEC_FAIL:-0} + 1 ))
      if [ "$CONSEC_FAIL" -ge 5 ]; then
        echo "=== ABORT: $CONSEC_FAIL consecutive gate failures — INFRASTRUCTURE? See $dir.log" | tee -a "$TABLE" >&2
        exit 7
      fi
      continue
    }
    CONSEC_FAIL=0
  # orphan guard (2026-09-10): the exi-ai headless build ignores TERM; a
  # survivor holds UDP 51442 and fails every later gate with a connect timeout.
  pkill -9 -f "squashfs-root/usr/bin/dolphin-emu" 2>/dev/null || true
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
    "$best_snap" "$OUTDIR/argmax_fox" --runs 3 --dummy stand --runner sync $DECODE_ARGS \
    -- --reaction-delay "$GATE_REACTION" $GATE_ID_ARGS --headless --emulation-speed 0 --blocking-input --slippi-port 51442 \
    2>&1 | grep -aE "r[123] " | tail -3 | tee -a "$TABLE"
  EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
    "$best_snap" "$OUTDIR/argmax_mewtwo" --runs 1 --dummy stand --runner sync $DECODE_ARGS \
    -- --reaction-delay 4 --dummy-character mewtwo --headless --emulation-speed 0 --blocking-input --slippi-port 51442 \
    2>&1 | grep -aE "r1 " | tail -1 | tee -a "$TABLE"
fi
echo "=== SWEEP DONE. Table: $TABLE"