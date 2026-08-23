#!/usr/bin/env bash
# Blind A/B decider launcher (2026-08-22 evening, ep4 vs g15).
# Game N's arm comes from sealed_order.txt — DO NOT cat that file or
# the gN.log Policy line until all games are played (Bradley is blind).
# Usage: devenv shell -- bash eval_runs/0822_ab_decider/launch_game.sh N
set -euo pipefail
N="$1"
cd /home/blewf/git/exphil

# GOTCHA #100 guard: human rungs get an idle GPU, verified mechanically.
if pgrep -f "[d]agger_drill|[r]un_g|[g]ate_sweep|[e]val_live_protocol|[t]rain_from_replays" >/dev/null; then
  echo "REFUSING: training/drill processes live (GOTCHA #100)" >&2
  exit 1
fi

line=$(sed -n "${N}p" eval_runs/0822_ab_decider/sealed_order.txt)
policy=${line%% *}
extra=""
case "$line" in *" "*) extra=${line#* } ;; esac

EXPHIL_NETPLAY_HOME="$HOME/.config/slippi-dolphin-bot" \
EXPHIL_QUEUE_TRACE=1 \
XLA_TARGET=cuda12 \
EXPHIL_GPU_MEMORY_FRACTION=0.25 \
nohup mix run scripts/play_dolphin_async.exs \
  --policy "$policy" \
  --dolphin "$HOME/.config/Slippi Launcher/netplay-beta-nixos" \
  --iso "$HOME/isos/melee.iso" \
  --connect-code "DBTD#411" \
  --slippi-port 51442 \
  --character fox \
  --frame-delay 4 \
  --deterministic \
  $extra \
  > "eval_runs/0822_ab_decider/g${N}.log" 2>&1 &
disown
echo "game $N launched (pid $!)"
