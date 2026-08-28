#!/usr/bin/env bash
# Wait for the machine to go QUIET (permuter done + 1-min loadavg low), then
# run the per-head temperature bracket. Launched via systemd-run --user so it
# survives the agent Bash tool's ~10-min SIGTERM (HANDOFF_2026-08-26 §5).
#
# The bracket is a LIVE eval — its numbers are invalid under CPU load (the
# starvation law), so we block until the decomp-permuter job is gone AND the
# 1-min loadavg is < 2. Idle on this box is ~0.2; the permuter drives it to
# 25+. Safety cap: if never quiet after 6h, abort rather than produce garbage.
set -uo pipefail
cd "$(dirname "$0")/.."

OUT="eval_runs/0826_gen_v1_sweep/per_head_temp"
mkdir -p "$OUT"
LOG="$OUT/wait_then_bracket.log"
MARKER="$OUT/.bracket_done"

quiet() {
  local p load
  p=$(pgrep -fc "permuter\.py" 2>/dev/null)
  load=$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)
  [ "$p" -eq 0 ] && [ "$load" -lt 2 ]
}

echo "[$(date '+%F %T')] wait_then_bracket: waiting for quiet (permuter gone + 1-min loadavg < 2)" | tee -a "$LOG"

MAX_WAIT_MIN=360  # 6 hours
waited=0
while ! quiet; do
  sleep 60
  waited=$((waited + 1))

  # progress line every 30 min so a log reader can see it's still alive
  if [ $((waited % 30)) -eq 0 ]; then
    echo "[$(date '+%F %T')] still waiting (${waited}m) — loadavg=$(cut -d' ' -f1 /proc/loadavg) permuter=$(pgrep -fc 'permuter\.py|permuter-tui' 2>/dev/null)" | tee -a "$LOG"
  fi

  if [ "$waited" -ge "$MAX_WAIT_MIN" ]; then
    echo "[$(date '+%F %T')] NEVER QUIET after ${MAX_WAIT_MIN}m — ABORTING bracket (permuter=$(pgrep -fc 'permuter\.py|permuter-tui' 2>/dev/null), loadavg=$(cat /proc/loadavg)). Re-run manually when the core is free." | tee -a "$LOG"
    echo "aborted:never_quiet $(date '+%F %T')" > "$MARKER"
    exit 2
  fi
done

echo "[$(date '+%F %T')] MACHINE QUIET (loadavg=$(cut -d' ' -f1 /proc/loadavg)) — starting bracket" | tee -a "$LOG"
bash scripts/per_head_temp_bracket.sh >> "$LOG" 2>&1
rc=$?
echo "[$(date '+%F %T')] BRACKET DONE (exit $rc) — table: $OUT/bracket_table.txt" | tee -a "$LOG"
echo "done:$rc $(date '+%F %T')" > "$MARKER"
exit "$rc"
