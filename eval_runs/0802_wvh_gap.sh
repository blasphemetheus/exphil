#!/usr/bin/env bash
# Windowed-vs-headless gap forensics (task #9, 2026-08-02).
# Same checkpoint (ms_g2_mdq_ss), same d3 sync protocol: headless scores
# 381/min c367, windowed 105/min c19 (0801 locktest) — and live/netplay
# behaves like the WINDOWED regime, so record numbers overstate the live
# bot. This block collects the evidence to name the mechanism.
#
# Hypotheses (pre-registered):
#   H1 jitter-smear: compositor/vsync adds frame-time variance ->
#      offset-calibration smears (multi-peak / low max) -> chains die.
#      (LATENCY finding: chain capability tracks cal SHARPNESS.)
#      Predicts: windowed cal spread flat/low vs headless single sharp
#      peak; qtrace lag histogram wide vs tight.
#   H2 constant-shift: windowed adds a CONSTANT extra frame (vsync
#      pipeline) -> policy plays one rung off its trained delay.
#      Predicts: sharp cal/lag peak at a DIFFERENT offset than headless;
#      re-eval at --frame-delay 2 or 4 windowed would recover.
#   H3 load-latency: rendering slows inference past the frame budget.
#      Predicts: frame-skip stat > 0 windowed (sync loop is blocked
#      otherwise), qtrace decision gaps.
#
# Protocol: 3 runs each regime, qtrace ON, then analyzers:
#   scripts/analyze_qtrace.exs LOG        (lag curve sharp vs smear)
#   probe_cycle_margins offset-cal on the replays (cal spread)
# H2 follow-up (only if H2 signature): windowed evals at d2/d4.
#
# RUN ONLY ON AN IDLE MACHINE (no training beam — NO-MIX; timing-critical).
# Windowed under locked screen is validated safe (0801_locktest), but the
# 08-01 postmortem crash was a window-map-under-lock RACE in a Hyprland
# plugin — if the compositor dies, this script survives in tmux; restart
# the Claude session with claude --resume.
set -uo pipefail
cd "$(dirname "$0")/.."
export ISO="${ISO:-$HOME/isos/melee.iso}"
export DOLPHIN_DIR="$HOME/.config/Slippi Launcher/netplay-beta-nixos"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_GPU_MEMORY_FRACTION=0.25
export EXPHIL_QUEUE_TRACE=1
# GOTCHA #82: prebuilt NIFs + rustc 1.97 can't rebuild old crate pins
export EXPHIL_SKIP_NIF_COMPILE=1

CKPT=checkpoints/ms_g2_mdq_ss.bin

echo "=== WVH headless block $(date +%H:%M:%S)"
EXLA_TARGET=host bash scripts/eval_live_protocol.sh "$CKPT" \
  eval_runs/0802_wvh_headless --runs 3 --dummy stand --runner sync \
  -- --frame-delay 3 --headless --slippi-port 51442

# Task #10 rider: speed-0 + blocking-input re-measure. The July trust
# ladder measured the speed-0 smear BEFORE the blocking-input fixes
# landed; if blocking now paces gameplay sharply (cal peak >= 0.75),
# the runtime speed-switch lever is unnecessary — speed 0 becomes the
# fast-menus recipe outright.
echo "=== WVH headless speed0+blocking block $(date +%H:%M:%S)"
EXLA_TARGET=host bash scripts/eval_live_protocol.sh "$CKPT" \
  eval_runs/0802_wvh_headless_s0block --runs 3 --dummy stand --runner sync \
  -- --frame-delay 3 --headless --emulation-speed 0 --blocking-input --slippi-port 51442

echo "=== WVH windowed block $(date +%H:%M:%S)"
EXLA_TARGET=host bash scripts/eval_live_protocol.sh "$CKPT" \
  eval_runs/0802_wvh_windowed --runs 3 --dummy stand --runner sync \
  -- --frame-delay 3 --blocking-input --slippi-port 51442

echo "=== WVH qtrace analysis $(date +%H:%M:%S)"
for d in eval_runs/0802_wvh_headless eval_runs/0802_wvh_windowed; do
  for log in "$d"/r*.log; do
    echo "--- $log"
    EXLA_TARGET=host mix run scripts/analyze_qtrace.exs "$log" 2>/dev/null | tail -8
  done
done
echo "=== WVH done $(date +%H:%M:%S)"
