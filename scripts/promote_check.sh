#!/usr/bin/env bash
# Pre-promotion gate: may this checkpoint be called "production"?
#
#   bash scripts/promote_check.sh checkpoints/ms_gX.bin [--rung 3]
#
# WHY THIS EXISTS (2026-08-04): ms_g6_sp1 was crowned on 08-03 for winning
# the stand-dummy comparison (434/min vs 380) and then scored ZERO shines
# in two games against a human, while the policy it displaced shone 20-27
# per game. The stand-dummy ranking was not merely uninformative — it was
# INVERTED. A standing dummy rewards a policy for exploiting a world where
# nothing ever happens, and cannot distinguish "good at multishining" from
# "good at multishining only when unmolested".
#
# The three checks below are the cheapest known discriminators:
#
#   1. DEPLOY-RUNG CHAIN — score at the delay you will actually run at, not
#      the policy's best rung. sp1's strength was d2; deployment is d3.
#   2. OFF-DISTRIBUTION (Yoshi's Story) — the human-free absorber repro
#      (task A): a policy that collapses here collapses under pressure.
#      Expect run-to-run variance; that is the point, so we run 3.
#   3. MOVING OPPONENT — a dummy that never acts is the whole problem.
#      cpu-9 measured FLAT (21-26/min) across every policy and does NOT
#      discriminate, so this uses --p2-policy (policy vs policy).
#   0. OPPONENT-SENSITIVITY (offline, seconds, runs FIRST) — perturb the
#      opponent's state in a fixed replay's frames and measure B/X logit
#      movement (scripts/probe_opponent_dependence.exs). Measured
#      2026-08-04 on the g6/g4/g2 labeled contrast: the score ranks
#      INVERSELY with human performance (g4 1.34 / g2 1.83 / g6 3.84 vs
#      human shines 40 / ~25 / 0). Static overfit is not opponent
#      BLINDNESS — it is overfitting TO the static opponent: the dummy's
#      exact state becomes cycle context, and any perturbation (i.e. a
#      human) destabilizes it. LOW = robust cycle, HIGH = red flag.
#      Three calibration points; advisory like everything else here.
#
# All three are advisory: they print a verdict, they do not block. We have
# one human data point per policy and thresholds are not calibrated — a
# hard gate on an uncalibrated number would be its own failure mode. What
# is NOT advisory is the rule: do not write "production" in the docs
# without running this and recording the numbers.
set -uo pipefail
cd "$(dirname "$0")/.."
CKPT="${1:?usage: promote_check.sh <checkpoint.bin> [--rung N]}"
shift || true
RUNG=3
[ "${1:-}" = "--rung" ] && RUNG="$2"

export ISO="${ISO:-$HOME/isos/melee.iso}"
# All rungs below run --headless, which since the native libmelee_ex
# bridge (2026-08-05) requires the exi-ai HEADLESS build — the netplay
# AppImage ships only the xcb Qt platform and dies on "-platform
# headless" (2026-08-08, the g13 GATE 0 lesson). Env-overridable.
export DOLPHIN_DIR="${DOLPHIN_DIR:-$HOME/.local/share/slippi/exi-ai/dolphin-emu-headless}"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_GPU_MEMORY_FRACTION=0.25
export EXPHIL_SKIP_NIF_COMPILE=1

NAME=$(basename "$CKPT" .bin)
OUT="eval_runs/promote_${NAME}_$(date +%m%d%H%M)"
mkdir -p "$OUT"

echo "=== PROMOTE-CHECK $NAME (deploy rung d$RUNG)"

echo "--- 0/3 opponent-sensitivity (offline; LOW=robust, HIGH=static-overfit flag)"
echo "    reference: g4_d2mix 1.34 (human-best) / g2_mdq_ss 1.83 / g6_sp1 3.84 (human-zero)"
XLA_TARGET=cpu mix run scripts/probe_opponent_dependence.exs \
  --policies "$CKPT" --delay-id "$RUNG" \
  --out "$OUT/opp_dependence.json" 2>&1 | grep -aE "DEPENDENCE|far|neutral" || true

echo "--- 1/3 deploy-rung chain (FD, stand dummy)"
EXLA_TARGET=host bash scripts/eval_live_protocol.sh "$CKPT" "$OUT/rung" \
  --runs 3 --dummy stand --runner sync \
  -- --frame-delay "$RUNG" --headless --emulation-speed 0 --blocking-input \
     --slippi-port 51442

echo "--- 2/3 off-distribution (Yoshi's Story — the human-free absorber repro)"
EXLA_TARGET=host bash scripts/eval_live_protocol.sh "$CKPT" "$OUT/ys" \
  --runs 3 --dummy stand --runner sync \
  -- --frame-delay "$RUNG" --stage yoshis_story --headless \
     --emulation-speed 0 --blocking-input --slippi-port 51442

echo "--- 3/3 moving opponent (policy vs policy)"
EXLA_TARGET=host bash scripts/eval_live_protocol.sh "$CKPT" "$OUT/vs" \
  --runs 3 --runner sync \
  -- --frame-delay "$RUNG" --p2-policy checkpoints/ms_g2_mdq_ss.bin \
     --headless --emulation-speed 0 --blocking-input --slippi-port 51442

echo "=== PROMOTE-CHECK summary (CHAINS — not qtrace press counts)"
for phase in rung ys vs; do
  echo "--- $phase"
  EXLA_TARGET=host mix run scripts/analyze_shine_source.exs "$OUT/$phase"/r*.slp 2>&1 \
    | grep -aE "replay |r[0-9] " || echo "  (no replays)"
done
echo "=== done. Record these numbers next to any 'production' claim."
