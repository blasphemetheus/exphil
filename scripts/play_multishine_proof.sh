#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export EXPHIL_SKIP_NIF_COMPILE=1 EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.15
exec mix run --no-compile --no-deps-check \
  -r lib/exphil/constants.ex \
  -r ../edifice/lib/edifice/recurrent/recurrent.ex \
  -r lib/exphil/networks/policy/execution_contract.ex \
  -r lib/exphil/networks/policy/backbone.ex \
  -r ../libmelee_ex/lib/melee/controller.ex -r ../libmelee_ex/lib/melee/console.ex \
  -r lib/exphil_bridge/melee_port.ex -r lib/exphil/data/action_frame_convention.ex \
  -r lib/exphil/embeddings/player.ex -r lib/exphil/agents/agent.ex \
  scripts/play_dolphin_async.exs \
  --policy eval_runs/0913_no_dropout_fit/round21/candidate.bin \
  --character fox --stage final_destination --reaction-delay 2 --live-af \
  --temperature 1.0 --port 1 --opponent-port 2 --human-port 2 \
  --blocking-input --postgame-delay 15 \
  --dolphin "$HOME/.config/Slippi Launcher/netplay-beta-nixos/Slippi_Netplay_Mainline-x86_64.AppImage" \
  --iso "$HOME/isos/melee.iso" \
  --replay-dir "eval_runs/local_multishine_$(date +%Y%m%d_%H%M%S)" "$@"
