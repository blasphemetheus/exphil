#!/usr/bin/env bash
# Margin-weight lever (task #20, 2026-08-05): buy aerial-shine margin at
# training time — the probe-as-regularizer idea aimed at the delay-break
# mechanism, implemented as critical-frame BCE upweighting (x3 on the one
# B-press-edge frame per cycle, 9.9% of fixture frames; rides the
# validated sampling_weights machinery, composed by elementwise max).
#
# Mechanism being targeted (delay-break study): netplay chains die by a
# per-link Bernoulli at the aerial-shine decision; declared delay-id sets
# its margin (p10: id2 -0.52 / id3 +0.09 / id4 +0.25 on g10b). If
# training weight can widen those margins, chains-vs-human should rise
# without needing ever-higher delay rungs.
#
# Prereg (PRIMARY readout = the margin sweep, not chains):
#   P1 id3/id4 p10 margins widen materially (id3 >= +0.2, id4 >= +0.4)
#      AND stand d3 >= 300 => lever validated; next = human session A/B
#      (g12 vs g10b, same session, chains).
#   P2 margins ~unchanged, stand holds => BCE saturation — upweighting
#      held frames buys little; next lever = explicit hinge margin loss
#      (needs the F3-style batch-field plumbing).
#   P3 stand < 300 => the skew distorts the cycle; halve the weight.
set -uo pipefail
cd "$(dirname "$0")/.."
export ISO="${ISO:-$HOME/isos/melee.iso}"
export DOLPHIN_DIR="$HOME/.config/Slippi Launcher/netplay-beta-nixos"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_SKIP_NIF_COMPILE=1

ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp,eval_runs/0802_d2pool/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

echo "=== G12MARGIN TRAIN $(date +%H:%M:%S)"
EXPHIL_GPU_MEMORY_FRACTION=0.75 mix run scripts/dagger_drill.exs \
  --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
  --rollouts "$ROLL" --opening-replays "$OPEN" \
  --snippet-frames "eval_runs/0804_snippets_human_ad2/snippets.frames" \
  --margin-weight 3.0 \
  --max-epochs 60 --prev-action-dropout 0.6 --transition-weight 2.0 \
  --multi-delay "2,3,4" --pipeline-offset 2 --queue-depth 4 --with-delay-id \
  --scheduled-sampling 0.5 --ss-ramp 10 \
  --out checkpoints/ms_g12_margin.bin \
  2>&1 | grep -aE "Snippets:|Margin weighting:|Converged|diverged|exported|error|\*\*" | tail -6
[ -f checkpoints/ms_g12_margin.bin ] || { echo "=== G12MARGIN FAILED" >&2; exit 1; }

echo "=== G12MARGIN GATE 1: delay-id margin sweep (PRIMARY; unfiltered)"
for id in 2 3 4; do
  echo "--- delay_id $id (g10b reference: id2 -0.523 / id3 +0.091 / id4 +0.248)"
  EXLA_TARGET=host mix run scripts/probe_cycle_margins.exs \
    --policies "checkpoints/ms_g12_margin.bin" \
    --replay "eval_runs/0804_cycle3b_stand/r1.slp" \
    --delay-id "$id" --out "eval_runs/interp/margins_g12_id$id.json" || true
done

echo "=== G12MARGIN GATE 2: stand d3 (1 run)"
EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
  checkpoints/ms_g12_margin.bin eval_runs/0805_g12_stand \
  --runs 1 --dummy stand --runner sync \
  -- --frame-delay 3 --headless --emulation-speed 0 --blocking-input --slippi-port 51442

echo "=== G12MARGIN GATE 3: YS (3 runs, bucket only — n>=8 for claims)"
EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
  checkpoints/ms_g12_margin.bin eval_runs/0805_g12_ys \
  --runs 3 --dummy stand --runner sync \
  -- --frame-delay 3 --stage yoshis_story --headless --emulation-speed 0 \
     --blocking-input --slippi-port 51442

echo "=== G12MARGIN chains"
for phase in stand ys; do
  echo "--- $phase"
  EXLA_TARGET=host mix run scripts/analyze_shine_source.exs \
    eval_runs/0805_g12_$phase/r*.slp 2>&1 | grep -aE "replay |r[0-9] "
done
echo "=== G12MARGIN done $(date +%H:%M:%S)"
