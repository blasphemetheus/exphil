#!/usr/bin/env bash
# Wait for the machine to be QUIET (user's Dolphin closed + no permuter/beam +
# loadavg < 2), then run the buttons-temperature sweep. Launched via
# systemd-run --user so it survives the agent's ~10-min SIGTERM. A live eval
# under load or beside a second Dolphin is invalid (staleness / port clash).
set -uo pipefail
cd "$(dirname "$0")/.."

OUT="eval_runs/0828_buttons_temp"
mkdir -p "$OUT"
LOG="$OUT/wait_then_sweep.log"
MARKER="$OUT/.sweep_done"

quiet() {
  local d p b load
  d=$(pgrep -fc "dolphin|Slippi" 2>/dev/null)
  p=$(pgrep -fc "permuter\.py" 2>/dev/null)
  b=$(pgrep -fc "beam\.smp" 2>/dev/null)
  load=$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)
  [ "$d" -eq 0 ] && [ "$p" -eq 0 ] && [ "$b" -eq 0 ] && [ "$load" -lt 2 ]
}

echo "[$(date '+%F %T')] wait_then_sweep: waiting for quiet (dolphin/permuter/beam gone + 1-min loadavg < 2)" | tee -a "$LOG"

MAX_WAIT_MIN=360
waited=0
while ! quiet; do
  sleep 60
  waited=$((waited + 1))

  if [ $((waited % 30)) -eq 0 ]; then
    echo "[$(date '+%F %T')] still waiting (${waited}m) — loadavg=$(cut -d' ' -f1 /proc/loadavg) dolphin=$(pgrep -fc 'dolphin|Slippi' 2>/dev/null) permuter=$(pgrep -fc 'permuter\.py' 2>/dev/null)" | tee -a "$LOG"
  fi

  if [ "$waited" -ge "$MAX_WAIT_MIN" ]; then
    echo "[$(date '+%F %T')] NEVER QUIET after ${MAX_WAIT_MIN}m — ABORTING sweep" | tee -a "$LOG"
    echo "aborted:never_quiet $(date '+%F %T')" > "$MARKER"
    exit 2
  fi
done

echo "[$(date '+%F %T')] MACHINE QUIET (loadavg=$(cut -d' ' -f1 /proc/loadavg)) — starting buttons-temperature sweep" | tee -a "$LOG"
bash scripts/buttons_temp_sweep.sh >> "$LOG" 2>&1
rc=$?
echo "[$(date '+%F %T')] SWEEP DONE (exit $rc) — table: $OUT/sweep_table.txt" | tee -a "$LOG"
echo "done:$rc $(date '+%F %T')" > "$MARKER"
exit "$rc"
