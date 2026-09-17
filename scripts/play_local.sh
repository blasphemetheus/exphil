#!/usr/bin/env bash
# Play a policy against a human on port 2 via the Slippi netplay-beta Dolphin.
# Usage: scripts/play_local.sh [policy.bin] [extra play_dolphin.exs flags...]
# Runs itself inside `devenv shell` if not already there. Exists because this
# command recalled from fish history with \-continuations turns "\<newline>"
# into an escaped space, which env then tries to execute.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -n "${DEVENV_ROOT:-}" ] || exec devenv shell -- "$0" "$@"

POLICY=eval_runs/0915_local_zero/corrected_bootstrap/candidate.bin
if [ $# -gt 0 ] && [[ $1 != --* ]]; then POLICY=$1; shift; fi
export EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.15
exec mix run scripts/play_dolphin.exs \
  --policy "$POLICY" \
  --character fox --stage battlefield \
  --reaction-delay 0 --live-af --temperature 1.0 \
  --port 1 --opponent-port 2 --human-port 2 \
  --blocking-input --frozen-stadium --postgame-delay 15 \
  --dolphin "$HOME/.config/Slippi Launcher/netplay-beta-nixos" \
  --iso "$HOME/isos/melee.iso" \
  --replay-dir "eval_runs/local_recording_temp_1_$(date +%Y%m%d_%H%M%S)" "$@"
