#!/usr/bin/env bash
# Generalist crown decider: convC2 vs edgeB, blind A/B vs a human
# (2026-08-12, the 0809 sealed-order protocol adapted for generalists).
#
# Run from the repo root:  bash eval_runs/0812_crown_decider/run_decider.sh
#
# - 6 games, 3 per policy, order sealed before game 1 (sealed_order.txt
#   is written up front — DON'T read it until scoring).
# - You take PORT 2 with the GC adapter; the bot drives its own CSS.
# - Generalist deploy settings: no frame-delay flag (trained at delay 0),
#   temperature 0.3 (matches the async-rung baselines: edgeB 0.5 stocks
#   taken/game, convC2 0.9).
# - Between games: press Enter when you're ready at your setup.
# - After: score with analyze_behavior per game dir, then unblind.
set -uo pipefail
cd "$(dirname "$0")/../.."

OUT=eval_runs/0812_crown_decider
C2=checkpoints/fox_il_v2_convC2_20260811_222129_best_policy.bin
EB=checkpoints/fox_il_v2_edgeB_20260810_060518_best_policy.bin

if [ ! -f "$OUT/sealed_order.txt" ]; then
  # 3 of each, shuffled once, sealed
  order=$(printf "convC2\nconvC2\nconvC2\nedgeB\nedgeB\nedgeB\n" | shuf | tr '\n' ' ')
  echo "sealed order: $order" > "$OUT/sealed_order.txt"
  echo "(order sealed to $OUT/sealed_order.txt — no peeking)"
fi

read -ra ORDER <<< "$(sed 's/^sealed order: //' "$OUT/sealed_order.txt")"

for i in 1 2 3 4 5 6; do
  pol_name="${ORDER[$((i-1))]}"
  case "$pol_name" in
    convC2) POLICY=$C2 ;;
    edgeB)  POLICY=$EB ;;
  esac
  echo
  echo "=== GAME $i/6 — press Enter when ready at the CSS (plug in, sit down) ==="
  read -r
  EXPHIL_QUEUE_TRACE=1 XLA_TARGET=cuda12 EXPHIL_GPU_MEMORY_FRACTION=0.25 \
  devenv shell -- mix run scripts/play_dolphin_async.exs \
    --policy "$POLICY" \
    --character fox --temperature 0.3 \
    --dolphin "$HOME/.config/Slippi Launcher/netplay-beta-nixos" \
    --iso "$HOME/isos/melee.iso" \
    --slippi-port 51442 \
    --on-game-end stop \
    --replay-dir "$OUT/g$i" 2>&1 | tee "$OUT/g$i.log" | grep -aE "Game end|stocks|Frame 0:|error" | tail -5
  echo "game $i recorded -> $OUT/g$i (jot your blind read: which model did that feel like?)"
done

echo
echo "All 6 games done. Score BEFORE unsealing:"
echo "  devenv shell -- mix run scripts/analyze_behavior.exs --replays $OUT/g1 --port 1   (etc.)"
echo "Then: cat $OUT/sealed_order.txt"
