#!/usr/bin/env bash
# Task #27 — architecture bake-off, RE-AIMED (2026-08-04): scored by the
# new instrument suite, not headline dummy chains.
#
# Arms: gru (control — the entire champion lineage) vs mamba_2 (the
# survey's pick for any new line: DRAMA, better + 2-8x faster at exactly
# our scale) on the CURRENT champion recipe (cycle-3b's, human snippets
# included; y-augment intentionally EXCLUDED so this measures the recipe
# both arms have history with — rerun with it if 4a lands P1).
#
# Scoring (per docs/guides/EVAL_PROTOCOL.md successive-halving order):
#   rung 0  opponent-sensitivity (offline; LOW=robust)
#   FD d3   chains, deterministic, 1 run
#   YS      3 runs, run-level collapse bucket + AbsorberEntry landings
#   dashboard platX/shield columns
# Caveat recorded up front: trunk-level interp readouts (OOD, probes) on
# the mamba arm are NOT trustworthy until W5's probing gotchas (conv
# off-by-one, gate-site probing, delta-sinks) land in Activations — this
# bake-off uses behavioral + head-level instruments only.
#
# Prereg:
#   P1 mamba_2 within 10% of gru on FD chain AND no worse on YS bucket
#      AND rung 0 in the robust band => literature prediction (GRU~Mamba
#      at 60f) holds; architecture confirmed not-the-bottleneck; close
#      #27 and stop spending on bake-offs.
#   P2 mamba_2 materially better on the OFF-distribution rungs (YS/rung0)
#      at similar FD => consider a mamba line for the fight-state arc.
#   P3 mamba_2 materially worse => GRU stays, note against the survey's
#      MaIL small-data prediction.
set -uo pipefail
cd "$(dirname "$0")/.."
export ISO="${ISO:-$HOME/isos/melee.iso}"
export DOLPHIN_DIR="$HOME/.config/Slippi Launcher/netplay-beta-nixos"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_SKIP_NIF_COMPILE=1

ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp,eval_runs/0802_d2pool/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

run_arm () { # backbone out_name
  local bb=$1 name=$2
  echo "=== BAKEOFF ARM $bb -> $name $(date +%H:%M:%S)"
  EXPHIL_GPU_MEMORY_FRACTION=0.7 mix run scripts/dagger_drill.exs \
    --backbone "$bb" \
    --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
    --rollouts "$ROLL" --opening-replays "$OPEN" \
    --snippet-frames "eval_runs/0804_snippets_human_ad2/snippets.frames" \
    --max-epochs 60 --prev-action-dropout 0.6 --transition-weight 2.0 \
    --multi-delay "2,3,4" --pipeline-offset 2 --queue-depth 4 --with-delay-id \
    --scheduled-sampling 0.5 --ss-ramp 10 \
    --out "checkpoints/$name.bin" \
    2>&1 | grep -aE "Snippets:|Converged|diverged|exported|error|\*\*" | tail -4
  [ -f "checkpoints/$name.bin" ] || { echo "=== ARM $bb FAILED" >&2; return 1; }

  echo "--- rung 0"
  EXLA_TARGET=host mix run scripts/probe_opponent_dependence.exs \
    --policies "checkpoints/$name.bin" \
    --out "eval_runs/interp/opp_dependence_$name.json" 2>&1 | grep -aE "DEPENDENCE"

  echo "--- FD d3 (1 run)"
  EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
    "checkpoints/$name.bin" "eval_runs/0804_bakeoff_${name}_stand" \
    --runs 1 --dummy stand --runner sync \
    -- --frame-delay 3 --headless --emulation-speed 0 --blocking-input --slippi-port 51442

  echo "--- YS (3 runs)"
  EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
    "checkpoints/$name.bin" "eval_runs/0804_bakeoff_${name}_ys" \
    --runs 3 --dummy stand --runner sync \
    -- --frame-delay 3 --stage yoshis_story --headless --emulation-speed 0 \
       --blocking-input --slippi-port 51442

  echo "--- chains"
  for phase in stand ys; do
    echo "  $phase:"
    EXLA_TARGET=host mix run scripts/analyze_shine_source.exs \
      "eval_runs/0804_bakeoff_${name}_$phase"/r*.slp 2>&1 | grep -aE "r[0-9] "
  done
}

# GRU control arm: retrained fresh (NOT reusing g10b) so both arms share
# seed-era, data order, and code state — the comparison is architecture only.
run_arm gru bakeoff_gru
run_arm mamba_2 bakeoff_mamba2

echo "=== BAKEOFF dashboard columns"
EXLA_TARGET=host mix run scripts/relapse_dashboard.exs \
  --policies "checkpoints/bakeoff_gru.bin,checkpoints/bakeoff_mamba2.bin" \
  --out eval_runs/interp/relapse_dashboard_bakeoff.json 2>&1 | grep -aE "policy|bakeoff"
echo "=== BAKEOFF done $(date +%H:%M:%S) — read against the prereg header."
