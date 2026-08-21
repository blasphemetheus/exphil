#!/usr/bin/env bash
# Collapse-forensics run (2026-08-20, Bradley's direction: isolate the
# nx bug's mechanism). Warm-start from g15r2's epoch-50 pre-collapse
# state (fox 362.5 c353) on the WILD nx stack (f843aa1a) with
# --collapse-forensics armed: the first batch whose loss < 1e-5 dumps
# pre/post trainer + the batch itself to checkpoints/collapse_scene/
# and halts for offline replay.
#
# PRE-REGISTERED (2026-08-20, before the run):
#   Recipe = g15r2's exactly (champion, NO awbc), --init-from the ep50
#   save, --ss-ramp 1 (ep50 ran at full scheduled-sampling P=0.5; the
#   ramp restarting would change the regime), 40-epoch cap.
#   Reads:
#     F1 scene captured (loss<1e-5 batch) -> offline decomposition:
#        which loss term/head is ~0, are AWBC/mask/weights degenerate,
#        does the same batch+params reproduce ~0 on the CALM stack
#        (a7497612)? A stack-dependent replay difference = the bug,
#        mechanically isolated.
#     F2 no scene in 40 epochs -> the ep50 state does not deterministically
#        re-enter the collapse; rerun once (nondeterministic training);
#        two clean runs -> regime is rarer than hoped, fall back to the
#        commit bisect (eval_runs/0820_nx_bisect/).
#   Per-batch loop is slower than the fast path (device sync per batch)
#   — expected ~2-3x epoch time; irrelevant for a mechanism hunt.
#   NO gates, NO crown implications.
set -euo pipefail
cd "$(dirname "$0")/../.."

# Guard: this run must be on the WILD stack.
NX_HEAD=$(git -C "$HOME/git/nx" rev-parse --short=8 HEAD)
[ "$NX_HEAD" = "f843aa1a" ] || { echo "nx is at $NX_HEAD, expected f843aa1a (wild stack)" >&2; exit 1; }
[ -f checkpoints/ms_g15r2_latest.bin ] || { echo "missing ep50 checkpoint" >&2; exit 1; }

export DOLPHIN_DIR="$HOME/.local/share/slippi/exi-ai/dolphin-emu-headless"
export ISO="$HOME/isos/melee.iso"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_SKIP_NIF_COMPILE=1

OUT=eval_runs/0820_collapse_forensics
ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp,eval_runs/0802_d2pool/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'

echo "=== FORENSICS START $(date +%H:%M:%S) nx=[$(git -C $HOME/git/nx log --oneline -1)]"
EXPHIL_GPU_MEMORY_FRACTION=0.75 mix run scripts/dagger_drill.exs \
  --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
  --rollouts "$ROLL" --opening-replays "$OPEN" \
  --snippet-frames "eval_runs/0804_snippets_human_ad2/snippets.frames" \
  --opp-scramble-frames 12000 \
  --max-epochs 40 --prev-action-dropout 0.6 --transition-weight 2.0 \
  --multi-delay "2,3,4" --pipeline-offset 2 --queue-depth 4 --with-delay-id \
  --scheduled-sampling 0.5 --ss-ramp 1 \
  --lr 2.0e-4 \
  --init-from checkpoints/ms_g15r2_latest.bin \
  --collapse-forensics \
  --out checkpoints/ms_forensics_scratch.bin \
  2>&1 | tee "$OUT/train.log" \
  | grep -aE "COLLAPSE|Warm-started|Scene|Converged|diverged|error|halt" | tail -8 || true

RC=$?
if [ -d checkpoints/collapse_scene ] && [ -n "$(ls checkpoints/collapse_scene 2>/dev/null)" ]; then
  echo "=== FORENSICS: SCENE CAPTURED (read F1) — $(ls checkpoints/collapse_scene)"
else
  echo "=== FORENSICS: no scene in this run (read F2; rc=$RC)"
fi
echo "=== FORENSICS DONE $(date +%H:%M:%S)"