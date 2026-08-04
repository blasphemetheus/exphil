#!/usr/bin/env bash
# Live-eval protocol runner — EXPOSURE_BIAS.md items 0a/0c operationalized.
#
#   scripts/eval_live_protocol.sh <policy.bin> <outdir> [--runs N] [--seconds S]
#       [--dummy cpu|stand] [--temperature T] [-- <extra play_dolphin_async args...>]
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
TEMPERATURE=""
RUNNER=async
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --runs) RUNS="$2"; shift 2 ;;
    --seconds) SECONDS_ARG="$2"; shift 2 ;;
    --dummy) DUMMY="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --runner) RUNNER="$2"; shift 2 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# --runner sync: frame-locked loop (play_dolphin.exs) — deterministic
# 1-frame delay, no staleness by construction; per-run health stat is the
# "skipped N (x%)" line instead. The jitter experiment of 2026-07-28.
RUNNER_SCRIPT=scripts/play_dolphin_async.exs
RUNNER_ARGS=(--on-game-end stop)
if [ "$RUNNER" = "sync" ]; then
  RUNNER_SCRIPT=scripts/play_dolphin.exs
  RUNNER_ARGS=()
fi

# --temperature T: stochastic decode (EXPOSURE_BIAS 6-replication follow-up —
# sampling is the only noise that can ESCAPE an absorber; stale-send repeat
# noise cannot, measured 2026-07-27). Replaces --deterministic for the run.
DECODE_ARGS=(--deterministic)
[ -n "$TEMPERATURE" ] && DECODE_ARGS=(--temperature "$TEMPERATURE")

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
echo "policy=$POLICY runner=$RUNNER runs=$RUNS seconds=$SECONDS_ARG dummy=$DUMMY decode=${DECODE_ARGS[*]} loadavg=$(cut -d' ' -f1-3 /proc/loadavg)" | tee "$OUTDIR/protocol.txt"

DUMMY_ARGS=(--dummy "$DUMMY" --dummy-character fox)
[ "$DUMMY" = "cpu" ] && DUMMY_ARGS+=(--dummy-cpu-level 1)

kill_dolphins() {
  # BOT-owned Dolphins only: the bot's wrapper runs via the appimage-run
  # cache (~/.cache/appimage-run/*/AppRun.wrapped); Bradley's launcher
  # sessions extract to /tmp/appimage_extracted_* and MUST survive
  # (2026-07-31: the unscoped kill took out his live game mid-play).
  ps -eo pid,args | grep '[A]ppRun.wrapped' | grep 'appimage-run' \
    | awk '{print $1}' | xargs -r kill -9 2>/dev/null
  sleep 2
  rm -rf /tmp/libmelee_* 2>/dev/null
}

for i in $(seq 1 "$RUNS"); do
  kill_dolphins
  echo "=== run $i/$RUNS start $(date +%H:%M:%S) loadavg=$(cut -d' ' -f1 /proc/loadavg)"
  run_start=$(date +%s)
  XLA_TARGET="${XLA_TARGET_EVAL:-cpu}" timeout $((SECONDS_ARG + 420)) mix run "$RUNNER_SCRIPT" \
    --policy "$POLICY" \
    --dolphin "$DOLPHIN_DIR" --iso "$ISO" \
    --character fox "${DUMMY_ARGS[@]}" \
    --seconds "$SECONDS_ARG" "${RUNNER_ARGS[@]}" "${DECODE_ARGS[@]}" \
    "${EXTRA[@]}" > "$OUTDIR/r$i.log" 2>&1
  kill_dolphins
  # mainline builds write into monthly subdirs (~/Slippi/2026-07/); a replay
  # older than the run start is a FAILED run's leftover, not this run's output
  # (the 2026-07-30 stale-copy artifact: 3 identical "runs" scored from one file)
  newest=$(ls -t "$HOME"/Slippi/*.slp "$HOME"/Slippi/*/*.slp 2>/dev/null | head -1)
  if [ -n "$newest" ] && [ "$(stat -c %Y "$newest")" -ge "$run_start" ]; then
    cp "$newest" "$OUTDIR/r$i.slp"
  else
    echo "  r$i NO FRESH REPLAY (run failed?) — see $OUTDIR/r$i.log" >&2
  fi
  grep -a "Final stats\|Staleness\|skipped\|SD FAILED\|replay finalized" "$OUTDIR/r$i.log" | sed "s/^/  r$i /"
done

# Offline-vs-live discriminator (task #21, 2026-08-03). ~10s, no Dolphin.
# Recorded BEFORE scoring so every block carries the number: high offline
# agreement + poor live numbers => STATE-STREAM / delay problem, not a
# learning failure (the distinction GOTCHA #81 took four escalations to
# reach). Advisory, not fatal — we have no calibrated threshold yet, and a
# hard gate on an uncalibrated number would block good runs.
echo "=== offline fixture agreement"
FIXTURE_AGREE=$(XLA_TARGET="${XLA_TARGET_EVAL:-cpu}" mix run scripts/eval_policy_on_fixture.exs \
  --policy "$POLICY" 2>&1 | grep -a "FIXTURE_AGREEMENT" | tail -1)
echo "  ${FIXTURE_AGREE:-(unavailable)}"
echo "fixture_agreement: ${FIXTURE_AGREE:-unavailable}" >> "$OUTDIR/protocol.txt"

echo "=== scoring"
SLPS=$(ls "$OUTDIR"/r*.slp 2>/dev/null)
if [ -z "$SLPS" ]; then echo "no replays captured" >&2; exit 4; fi
# NB: Output.puts writes to STDERR — do not 2>/dev/null here (first run of
# this script silently dropped the entire scoring section that way).
if [ "$DUMMY" = "cpu" ]; then
  for s in $SLPS; do mix run scripts/check_replay_ports.exs "$s" --expect-cpu 2 2>&1 | grep -a "port \|checks"; done
fi
mix run scripts/analyze_shine_source.exs $SLPS 2>&1 | grep -a "replay  \|r[0-9] "
# Harness-health gate (2026-07-30): offset calibration on r1, every block.
# Sharp single peak >=0.75 = trustworthy timing; a smeared/bimodal cal means
# the numbers above measure the harness, not the policy. Peak POSITION is
# harness-specific — compare concentration only (HANDOFF_2026-07-29b rule 2).
echo "=== calibration (r1)"
XLA_TARGET="${XLA_TARGET_EVAL:-cpu}" mix run scripts/probe_cycle_margins.exs \
  --policies "$POLICY" --replay "$OUTDIR/r1.slp" \
  --out "$OUTDIR/cal_r1.json" 2>&1 | grep -a "cal="
echo "=== done. Protocol: report mean AND range; <2x differences are unresolved (EXPOSURE_BIAS.md 0a/0b)."
