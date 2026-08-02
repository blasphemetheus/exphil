#!/usr/bin/env bash
# Delay-id patch probe (2026-08-02, task #1) — interp intervention on the
# delay-id one-hot channel via --delay-id-override (new flag).
# Two mysteries, one instrument:
#   jq d3 collapse: jq scores 0 shines at d3 (4/4 zeros, crouch absorber
#     from frame 104) while d2=89, d4=101. If forcing id=2 or 4 at d3
#     rescues shining, the delay-id CHANNEL conditioning triggers the
#     basin (not the actual delay dynamics).
#   mdq_ss d2 inversion: weakest rung is d2 (139.8 c6, 3-run-identical)
#     vs 381 c367 at d3 — inverted slope. If id=3 at d2 recovers, the
#     id channel drives the inversion; if unchanged, the dynamics do.
# Preregistered readings:
#   P1: jq d3 id={2,4} shines >> 0  => channel-triggered basin.
#   P2: mdq_ss d2 id=3 ~ d3-level  => channel-driven inversion.
#   P3: controls reproduce baselines (jq d3 ~0; mdq_ss d2 ~140 c6),
#       else regime is off and patched runs are uninterpretable.
# Protocol: 1-run screening, sync headless, stand dummy — identical to
# d3_grind1.sh evals (EXLA_TARGET=host, port 51442).
set -uo pipefail
cd "$(dirname "$0")/.."
export ISO="${ISO:-$HOME/isos/melee.iso}"
export DOLPHIN_DIR="$HOME/.config/Slippi Launcher/netplay-beta-nixos"
# NIF is built for cuda12; without this the protocol defaults XLA_TARGET=cpu
# and EXLA.NIF fails to load (learned 2026-08-02, first launch of this script).
export XLA_TARGET_EVAL=cuda12
export EXPHIL_GPU_MEMORY_FRACTION=0.25

run_eval () { # outdir, checkpoint, extra play args...
  local outdir=$1 ckpt=$2; shift 2
  echo "=== DELAYID $outdir $(date +%H:%M:%S)"
  EXLA_TARGET=host bash scripts/eval_live_protocol.sh "checkpoints/$ckpt" \
    "eval_runs/$outdir" --runs 1 --dummy stand --runner sync \
    -- --headless --slippi-port 51442 "$@"
}

# Controls (regime check against 07-31 baselines)
run_eval 0802_jq_d3_ctrl      ms_g1_jq.bin     --frame-delay 3
run_eval 0802_mdqss_d2_ctrl   ms_g2_mdq_ss.bin --frame-delay 2

# jq d3 collapse probes
run_eval 0802_jq_d3_id2       ms_g1_jq.bin     --frame-delay 3 --delay-id-override 2
run_eval 0802_jq_d3_id4       ms_g1_jq.bin     --frame-delay 3 --delay-id-override 4

# mdq_ss inversion probes
run_eval 0802_mdqss_d2_id3    ms_g2_mdq_ss.bin --frame-delay 2 --delay-id-override 3
run_eval 0802_mdqss_d3_id2    ms_g2_mdq_ss.bin --frame-delay 3 --delay-id-override 2

echo "=== DELAYID PROBE COMPLETE $(date +%H:%M:%S)"
