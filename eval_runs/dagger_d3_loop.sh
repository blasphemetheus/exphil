#!/usr/bin/env bash
# DAgger d3 campaign loop (2026-07-31): rounds 2-4 of the sync-d3 rung climb.
# Round 1 (ms_d3_dagger1_ad3, --action-delay 3): 48/min c1 at d3 — R1-quality.
# Each round: collect 6 rollouts from the CURRENT policy at sync-d3 -> train
# on ALL rollouts so far (fixture-relabeled, farm-5 rule by construction) ->
# eval d3+d4 screening. Stop early if d3 chains look locked (>=50).
set -uo pipefail
cd "$(dirname "$0")/.."
export ISO="${ISO:-$HOME/isos/melee.iso}"
export DOLPHIN_DIR="$HOME/.config/Slippi Launcher/netplay-beta-nixos"
export XLA_TARGET_EVAL=cuda12

POLICY=checkpoints/ms_d3_dagger1_ad3.bin
ROLLOUTS="eval_runs/dagger_d3_round1_collect/r1.slp,eval_runs/dagger_d3_round1_collect/r2.slp,eval_runs/dagger_d3_round1_collect/r3.slp,eval_runs/dagger_d3_round1_collect/r4.slp,eval_runs/dagger_d3_round1_collect/r5.slp,eval_runs/dagger_d3_round1_collect/r6.slp"

for round in 2 3 4; do
  echo "=== D3LOOP round $round: COLLECT from $POLICY $(date +%H:%M:%S)"
  COLLECT=eval_runs/dagger_d3_round${round}_collect
  EXLA_TARGET=host bash scripts/eval_live_protocol.sh "$POLICY" "$COLLECT" \
    --runs 6 --dummy stand --runner sync -- --frame-delay 3 --headless --emulation-speed 0
  for f in "$COLLECT"/r*.slp; do ROLLOUTS="$ROLLOUTS,$f"; done

  OUT=checkpoints/ms_d3_dagger${round}_policy.bin
  echo "=== D3LOOP round $round: TRAIN $(date +%H:%M:%S)"
  mix run scripts/dagger_drill.exs \
    --expert multishine \
    --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
    --rollouts "$ROLLOUTS" \
    --action-delay 3 --prev-action-dropout 0.6 --transition-weight 2.0 \
    --out "$OUT" \
    2>&1 | grep -aE "Converged|exported|corrected|Aggregate|error|\*\*" | tail -12
  [ -f "$OUT" ] || { echo "=== D3LOOP round $round TRAIN FAILED" >&2; exit 1; }
  POLICY="$OUT"

  echo "=== D3LOOP round $round: EVAL $(date +%H:%M:%S)"
  for d in 3 4; do
    EXLA_TARGET=host bash scripts/eval_live_protocol.sh "$POLICY" \
      eval_runs/0731_d3loop_r${round}_d$d --runs 1 --dummy stand --runner sync \
      -- --frame-delay $d --headless --emulation-speed 0
  done
done
echo "=== D3LOOP done $(date +%H:%M:%S)"
