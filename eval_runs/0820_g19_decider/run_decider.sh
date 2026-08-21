#!/usr/bin/env bash
# THE PROMOTE RUNG — 3-ARM VERSION (2026-08-21, Bradley's call after
# the g20 replicate): ms_g19_ep4 (deploy-band profile: fox 437 / d4id3
# 388 / mewtwo 94) vs ms_g20_ep13 (opponent-invariant profile: fox 435
# / MEWTWO 424 / d4id3 dead) vs ms_g15_oppmask_full (champion). Blind,
# LOCAL d3. g6 rule: no crown before this session.
#
# PRE-REGISTERED (2026-08-21, before the session):
#   6 games, sealed shuffled order (2 per arm; n=2 resolves only
#   categorical gaps — epsilon differences go to a rematch).
#   Reads:
#     WIN(x)   an arm chains vs the human while others don't -> that
#              arm is the production candidate (its known rung profile
#              sets the deploy story).
#     INV      both new arms collapse vs human while g15 chains ->
#              peak checkpoints are dummy-specialists; gate-sweep
#              needs a human rung; g15 retains.
#     TIE      comparable chains -> g15 retains on incumbency; best
#              new arm graduates to the netplay rung.
#   HUMAN on in-game
#   PORT 1 (Bradley's request): bot moved OFF its default via
#   --port 2 --opponent-port 1, human declared port 1 via
#   EXPHIL_HUMAN_PORT=1 (no collision once the bot is on 2; the
#   collision was the first attempt's failure). Score at bot PORT 2.
#   Pick FINAL
#   DESTINATION at the CSS (comparable to stand-rung records).
#   Primary metric: bot max chain + shines/min per game, scored from
#   replays (analyze_shine_source, bot port 2). Secondary: Bradley's
#   blind per-game reads + arm guesses before unsealing.
#   Reads:
#     A-WIN  ep4 matches/beats g15 on chains vs a human -> ep4 is the
#            new production candidate (netplay d4-id3 check rides the
#            next remote session before full crown).
#     A-INV  ep4 chains collapse vs human while g15 chains (the g6_sp1
#            inversion pattern) -> early-peak policies are dummy-
#            specialists; gate-sweep needs a human-robustness rung;
#            g15 retains.
#     A-TIE  both chain comparably -> g15 retains on incumbency; ep4
#            graduates to the netplay rung for the tiebreak.
#   Known ep4 profile going in: d3-specialist, mewtwo 93.9 c1, no d2.
#
# Run from repo root: bash eval_runs/0820_g19_decider/run_decider.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

# HARD GUARD (added 2026-08-20 after the invalidated first session): a
# live training/eval beam starves the game loop via GPU contention —
# frames get stale actions (22% observed), the bot plays at an
# untrained lag regime, and BOTH arms flatline (ground shines, no
# chains). Human sessions require an idle GPU, full stop.
if pgrep -f "[d]agger_drill|[r]un_g[0-9]+\.sh|[g]ate_sweep|[e]val_live_protocol" >/dev/null; then
  echo "REFUSING: a training/eval process is live (GPU contention invalidates the session):"
  pgrep -af "[d]agger_drill|[r]un_g[0-9]+\.sh|[g]ate_sweep|[e]val_live_protocol" | head -3
  echo "Wait for it to finish (or ask the agent to stop it), then rerun."
  exit 1
fi

OUT=eval_runs/0820_g19_decider
declare -A POLICIES=(
  [ep4]=checkpoints/ms_g19_ep4.bin
  [ep13]=checkpoints/ms_g20_ep13.bin
  [g15]=checkpoints/ms_g15_oppmask_full.bin
)

if [ ! -f "$OUT/sealed_order.txt" ]; then
  order=$(printf "ep4\nep4\nep13\nep13\ng15\ng15\n" | shuf | tr '\n' ' ')
  echo "sealed order: $order" > "$OUT/sealed_order.txt"
  echo "(order sealed — no peeking)"
fi

read -ra ORDER <<< "$(sed 's/^sealed order: //' "$OUT/sealed_order.txt")"

for i in 1 2 3 4 5 6; do
  name="${ORDER[$((i-1))]}"
  echo
  echo "=== GAME $i/6 — press Enter when ready at the CSS (pick FD) ==="
  read -r
  EXPHIL_QUEUE_TRACE=1 EXPHIL_HUMAN_PORT=1 XLA_TARGET=cuda12 EXPHIL_GPU_MEMORY_FRACTION=0.25 \
  devenv shell -- mix run scripts/play_dolphin_async.exs \
    --policy "${POLICIES[$name]}" \
    --port 2 --opponent-port 1 \
    --character fox --deterministic \
    --frame-delay 3 \
    --dolphin "$HOME/.config/Slippi Launcher/netplay-beta-nixos" \
    --iso "$HOME/isos/melee.iso" \
    --slippi-port 51442 \
    --on-game-end stop \
    --replay-dir "$OUT/g$i" 2>&1 | tee "$OUT/g$i.log" \
    | grep -aE --line-buffered "warmup complete|Game end|error|EXIT"
  # (line-buffered, no tail — the warmup line must appear LIVE: wait for
  # "JIT warmup complete" before starting the game, so game 1 never runs
  # on a half-compiled bot.)
  echo "game $i done -> $OUT/g$i  (blind read: chains? pressure response? which arm?)"
done

echo
echo "Write your 6 arm guesses down, THEN: cat $OUT/sealed_order.txt"
echo "Score: mix run scripts/analyze_shine_source.exs '$OUT/g*/**/*.slp' --port 2"