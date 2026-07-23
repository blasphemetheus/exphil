#!/usr/bin/env bash
# Training-run watchdog — the r16 lesson operationalized (2026-07-22).
#
# r16 died at 03:05 and went undetected until 10:00 because the monitor
# grepped the drill log for error STRINGS (FATAL/diverged/NaN) — and the
# actual death (an ArgumentError, and in general any SIGKILL/OOM) writes
# no line the grep matched. The log just went quiet, and quiet read as OK.
#
# Correct liveness for a training run is NOT a log-grep. It is:
#   pid alive  AND  (log advancing  OR  GPU still computing)
# The GPU clause matters because the drill is legitimately SILENT for up
# to 5 epochs at a time (it logs per-epoch now, but older runs every 5th),
# so ~10-min mamba_2 epochs can leave the log untouched for ~40 min during
# perfectly healthy training. GPU utilisation distinguishes "slow but
# alive" from "hung".
#
# DEATH (process gone): watchdog prints a report and EXITS non-zero — if
# launched in the background from a Claude session, that exit is the
# notification. Also fires a desktop notify-send if available.
# HANG (pid alive, log stale AND GPU idle for K consecutive checks):
# WARNs (notify + report) but keeps watching, because a legitimate
# post-training eval/fan-out phase can idle the GPU while the launcher
# lives.
#
# Usage:
#   scripts/train_watchdog.sh --pid <launcher_pid> --log <logfile> \
#     [--checkpoint <path>] [--interval 60] [--stale-min 25] \
#     [--gpu-idle-checks 5] [--gpu-busy-pct 10]
#
# Watch the TOP-LEVEL launcher pid (overnight_newera8.sh), not a beam:
# the launcher survives the per-round `mix run` beam cycling, so its exit
# means the whole run ended (then --checkpoint decides success vs fail).

set -uo pipefail

PID=""
LOG=""
CKPT=""
INTERVAL=60
STALE_MIN=25
GPU_IDLE_CHECKS=5
GPU_BUSY_PCT=10

while [ $# -gt 0 ]; do
  case "$1" in
    --pid) PID="$2"; shift 2 ;;
    --log) LOG="$2"; shift 2 ;;
    --checkpoint) CKPT="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --stale-min) STALE_MIN="$2"; shift 2 ;;
    --gpu-idle-checks) GPU_IDLE_CHECKS="$2"; shift 2 ;;
    --gpu-busy-pct) GPU_BUSY_PCT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 64 ;;
  esac
done

[ -n "$PID" ] || { echo "watchdog: --pid required" >&2; exit 64; }
[ -n "$LOG" ] || { echo "watchdog: --log required" >&2; exit 64; }

notify() {
  # Desktop toast if a session bus is reachable; never fatal if not.
  command -v notify-send >/dev/null 2>&1 && notify-send -u critical "$1" "$2" 2>/dev/null || true
}

gpu_util() {
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'
}

log_age_min() {
  # Minutes since the log was last written; huge number if it's missing.
  if [ -f "$LOG" ]; then
    echo $(( ( $(date +%s) - $(stat -c %Y "$LOG") ) / 60 ))
  else
    echo 999999
  fi
}

report() {
  echo "======================================================================"
  echo "[watchdog] $(date '+%F %T')  $1"
  echo "  pid=$PID  log=$LOG  log_age=$(log_age_min)min  gpu=$(gpu_util)%"
  echo "  --- last log lines (non-progress-bar) ---"
  grep -avE "Embedding:|Progress:|\[K" "$LOG" 2>/dev/null | tail -12 | sed 's/^/  | /'
  echo "======================================================================"
}

echo "[watchdog] $(date '+%F %T') watching pid $PID, log $LOG"
echo "[watchdog] healthy = pid alive AND (log <${STALE_MIN}min old OR gpu >=${GPU_BUSY_PCT}%)"
echo "[watchdog] death = pid gone; hang = ${GPU_IDLE_CHECKS} consecutive idle+stale checks"

idle_streak=0
warned_hang=0

while true; do
  if ! kill -0 "$PID" 2>/dev/null; then
    # Process ended — success or failure? Ask the checkpoint + log.
    if [ -n "$CKPT" ] && [ -f "$CKPT" ]; then
      report "RUN ENDED — checkpoint present ($CKPT): looks like SUCCESS"
      notify "Training done ✓" "pid $PID ended; checkpoint written"
      exit 0
    elif grep -qiE "Policy exported|Converged" "$LOG" 2>/dev/null; then
      report "RUN ENDED — log shows export/convergence: SUCCESS"
      notify "Training done ✓" "pid $PID ended; policy exported"
      exit 0
    else
      report "RUN ENDED WITHOUT CHECKPOINT — likely DIED (check exit status / journalctl -k for OOM)"
      notify "Training DIED ✗" "pid $PID gone, no checkpoint — see $LOG"
      exit 1
    fi
  fi

  age=$(log_age_min)
  util=$(gpu_util); util=${util:-0}

  if [ "$age" -lt "$STALE_MIN" ]; then
    idle_streak=0; warned_hang=0            # log fresh → healthy
  elif [ "$util" -ge "$GPU_BUSY_PCT" ]; then
    idle_streak=0; warned_hang=0            # log stale but GPU busy → silent-epoch, healthy
  else
    idle_streak=$((idle_streak + 1))        # stale AND idle → suspect hang
    if [ "$idle_streak" -ge "$GPU_IDLE_CHECKS" ] && [ "$warned_hang" -eq 0 ]; then
      report "POSSIBLE HANG — log stale ${age}min AND gpu idle ${idle_streak} checks (pid still alive; could be eval/fan-out)"
      notify "Training may be HUNG ⚠" "log stale ${age}min, gpu idle — pid $PID still alive"
      warned_hang=1                         # warn once per episode, keep watching
    fi
  fi

  sleep "$INTERVAL"
done
