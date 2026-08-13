#!/usr/bin/env bash
# Dose-response taste test: convC2 vs convC3 vs convC4, blind, one game
# each (HANDOFF_2026-08-12 item 0). Async metrics rank C2 > C4 > C3 —
# if blind feel reproduces that, the async rung is validated as a
# human-experience proxy.
#
# Run from repo root: bash eval_runs/0813_dose_session/run_dose_session.sh
# You take PORT 2. Jot a blind read after each game; at the end, GUESS
# the ranking before unsealing.
set -uo pipefail
cd "$(dirname "$0")/../.."

OUT=eval_runs/0813_dose_session
declare -A POLICIES=(
  [convC2]=checkpoints/fox_il_v2_convC2_20260811_222129_best_policy.bin
  [convC3]=checkpoints/fox_il_v2_convC3_20260812_173500_best_policy.bin
  [convC4]=checkpoints/fox_il_v2_convC4_20260812_193038_best_policy.bin
)

if [ ! -f "$OUT/sealed_order.txt" ]; then
  order=$(printf "convC2\nconvC3\nconvC4\n" | shuf | tr '\n' ' ')
  echo "sealed order: $order" > "$OUT/sealed_order.txt"
  echo "(order sealed — no peeking)"
fi

read -ra ORDER <<< "$(sed 's/^sealed order: //' "$OUT/sealed_order.txt")"

for i in 1 2 3; do
  name="${ORDER[$((i-1))]}"
  echo
  echo "=== GAME $i/3 — press Enter when ready at the CSS ==="
  read -r
  EXPHIL_QUEUE_TRACE=1 XLA_TARGET=cuda12 EXPHIL_GPU_MEMORY_FRACTION=0.25 \
  devenv shell -- mix run scripts/play_dolphin_async.exs \
    --policy "${POLICIES[$name]}" \
    --character fox --temperature 0.3 \
    --dolphin "$HOME/.config/Slippi Launcher/netplay-beta-nixos" \
    --iso "$HOME/isos/melee.iso" \
    --slippi-port 51442 \
    --on-game-end stop \
    --replay-dir "$OUT/g$i" 2>&1 | tee "$OUT/g$i.log" | grep -aE "Game end|error" | tail -3
  echo "game $i done -> $OUT/g$i (blind read: better or worse than the last?)"
done

echo
echo "Guess the ranking (best->worst), THEN: cat $OUT/sealed_order.txt"
echo "Score: devenv shell -- mix run scripts/analyze_behavior.exs --replays $OUT/g1 --port 1  (etc.)"
