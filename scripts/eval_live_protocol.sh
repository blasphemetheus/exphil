#!/usr/bin/env bash
# Live-eval protocol runner — EXPOSURE_BIAS.md items 0a/0c operationalized.
#
#   scripts/eval_live_protocol.sh <policy.bin> <outdir> [--runs N] [--seconds S]
#       [--dummy cpu|stand] [-- <extra play_dolphin_async args...>]
#
# Runs N live runs of ONE policy (the variance is in the RUNS, not the
# seeds), collects logs + replays, then scores them all with
# analyze_shine_source.exs and prints per-run staleness so machine-degraded
# runs can be discarded. Defaults: 3 runs, 60s, fox vs level-1 fox CPU,
# deterministic.
#
#   --dummy stand  idle opponent for a CLEAN capability number (the opponent
#                  interferes: walks up to jab mid-multishine, lasers cause
#                  shine lag). CPU remains the default for comparability with
#                  the existing ms_synth_a baseline (n=8).
#
# Refuses to run while training or another heavy mix job is live: under load
# the harness COLLAPSES (~1 fps at stage select, 2026-07-27) and any score
# would measure the machine, not the policy. Override with FORCE=1.
set -uo pipefail

POLICY="${1:?usage: eval_live_protocol.sh <policy.bin> <outdir> [--runs N] [--seconds S] [--dummy cpu|stand] [-- extra args]}"
OUTDIR="${2:?need outdir}"
shift 2

RUNS=3
SECONDS_ARG=60
DUMMY=cpu
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --runs) RUNS="$2"; shift 2 ;;
    --seconds) SECONDS_ARG="$2"; shift 2 ;;
    --dummy) DUMMY="$2"; shift 2 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

DOLPHIN_DIR="${DOLPHIN_DIR:-$HOME/.config/Slippi Launcher/netplay}"
ISO="${ISO:-$HOME/games/melee.iso}"
cd "$(dirname "$0")/.."

# Protocol rule 0c-2: live evals never share the machine with heavy jobs.
if [ "${FORCE:-0}" != "1" ]; then
  busy=$(pgrep -af "dagger_drill.exs|train.exs|train_from_replays" | grep -v $$ || true)
  if [ -n "$busy" ]; then
    echo "REFUSING: heavy mix job running (harness collapses under load, EXPOSURE_BIAS.md 0c):" >&2
    echo "$busy" >&2
    echo "Set FORCE=1 to override (your numbers will be garbage)." >&2
    exit 3
  fi
fi

mkdir -p "$OUTDIR"
echo "policy=$POLICY runs=$RUNS seconds=$SECONDS_ARG dummy=$DUMMY loadavg=$(cut -d' ' -f1-3 /proc/loadavg)" | tee "$OUTDIR/protocol.txt"

DUMMY_ARGS=(--dummy "$DUMMY" --dummy-character fox)
[ "$DUMMY" = "cpu" ] && DUMMY_ARGS+=(--dummy-cpu-level 1)

kill_dolphins() {
  # match the mounted AppImage, never the launcher path, never our own cmdline
  ps -eo pid,args | grep '[A]ppRun.wrapped' | awk '{print $1}' | xargs -r kill -9 2>/dev/null
  sleep 2
  rm -rf /tmp/libmelee_* 2>/dev/null
}

for i in $(seq 1 "$RUNS"); do
  kill_dolphins
  echo "=== run $i/$RUNS start $(date +%H:%M:%S) loadavg=$(cut -d' ' -f1 /proc/loadavg)"
  XLA_TARGET=cpu timeout $((SECONDS_ARG + 420)) mix run scripts/play_dolphin_async.exs \
    --policy "$POLICY" \
    --dolphin "$DOLPHIN_DIR" --iso "$ISO" \
    --character fox "${DUMMY_ARGS[@]}" \
    --on-game-end stop --seconds "$SECONDS_ARG" --deterministic \
    "${EXTRA[@]}" > "$OUTDIR/r$i.log" 2>&1
  kill_dolphins
  newest=$(ls -t "$HOME"/Slippi/*.slp 2>/dev/null | head -1)
  [ -n "$newest" ] && cp "$newest" "$OUTDIR/r$i.slp"
  grep -a "Final stats\|Staleness\|SD FAILED\|replay finalized" "$OUTDIR/r$i.log" | sed "s/^/  r$i /"
done

echo "=== scoring"
SLPS=$(ls "$OUTDIR"/r*.slp 2>/dev/null)
if [ -z "$SLPS" ]; then echo "no replays captured" >&2; exit 4; fi
if [ "$DUMMY" = "cpu" ]; then
  for s in $SLPS; do mix run scripts/check_replay_ports.exs "$s" --expect-cpu 2 2>/dev/null | grep -a "port\|✓\|❌"; done
fi
mix run scripts/analyze_shine_source.exs $SLPS 2>/dev/null | grep -a "replay\|r[0-9]"
echo "=== done. Protocol: report mean AND range; <2x differences are unresolved (EXPOSURE_BIAS.md 0a/0b)."
