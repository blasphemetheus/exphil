#!/usr/bin/env bash
# AWBC deploy-rung decider: ms_awbc_b1 (plain BC) vs ms_awbc_b2 (AWBC),
# blind, LOCAL d3 — the human rung the offline verdict is gated on
# (eval_runs/0813_awbc_ms/RESULTS.md; offline: B2 +17% shines / +21%
# chain, but rung-0 rose and the g6 lesson says dummy rankings can
# invert against humans — this session decides).
#
# 4 games, sealed shuffled order (2 per arm). You take PORT 2 (auto-
# configured via EXPHIL_HUMAN_PORT — no controller menu needed).
# Pick FINAL DESTINATION at the CSS (chain numbers comparable to the
# stand-rung records). Jot a blind read after each game; GUESS which
# arm was which before unsealing.
#
# Run from repo root: bash eval_runs/0813_awbc_decider/run_awbc_decider.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

OUT=eval_runs/0813_awbc_decider
declare -A POLICIES=(
  [b1]=checkpoints/ms_awbc_b1.bin
  [b2]=checkpoints/ms_awbc_b2.bin
)

if [ ! -f "$OUT/sealed_order.txt" ]; then
  order=$(printf "b1\nb1\nb2\nb2\n" | shuf | tr '\n' ' ')
  echo "sealed order: $order" > "$OUT/sealed_order.txt"
  echo "(order sealed — no peeking)"
fi

read -ra ORDER <<< "$(sed 's/^sealed order: //' "$OUT/sealed_order.txt")"

for i in 1 2 3 4; do
  name="${ORDER[$((i-1))]}"
  echo
  echo "=== GAME $i/4 — press Enter when ready at the CSS (pick FD) ==="
  read -r
  EXPHIL_QUEUE_TRACE=1 EXPHIL_HUMAN_PORT=2 XLA_TARGET=cuda12 EXPHIL_GPU_MEMORY_FRACTION=0.25 \
  devenv shell -- mix run scripts/play_dolphin_async.exs \
    --policy "${POLICIES[$name]}" \
    --character fox --deterministic \
    --frame-delay 3 \
    --dolphin "$HOME/.config/Slippi Launcher/netplay-beta-nixos" \
    --iso "$HOME/isos/melee.iso" \
    --slippi-port 51442 \
    --on-game-end stop \
    --replay-dir "$OUT/g$i" 2>&1 | tee "$OUT/g$i.log" | grep -aE "Game end|error" | tail -3
  echo "game $i done -> $OUT/g$i (blind read: chains? recovery after breaks? which arm?)"
done

echo
echo "Guess which games were which arm, THEN: cat $OUT/sealed_order.txt"
echo "Score: mix run scripts/analyze_shine_source.exs $OUT/g*/**/*.slp --port 1"
