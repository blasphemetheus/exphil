#!/usr/bin/env bash
# g22 (2026-09-10 13:10) = g20b recipe (4 rungs, full pool) trained to 60 EPOCHS.
# g21a (single rung, quarter pool) converged WORSE (loss 0.02-0.03) and chained 1
# at its own rung; g19_ep58 (loss 0.010) chains 12 under sampling -> convergence
# on the FULL pool is the lever to test first. Arms: a = g20b + 60 ep;
# b = a + --action-frame-buckets 24. Gate late epochs at d3 (id 3, g19_ep58s
# rung: 127/min c12 is the bar) and at d1 (derived id 0 and id 1).
# Findings that motivate it (eval_runs/0910_g20_causal): the g19 record was an
# argmax artifact (437 c438 argmax vs 45 c3 at T=1.0); g20a/b wall at chain 2-4
# at every rung/temperature/epoch; per-state probe: g20b is decisive (0.91-1.0)
# and a 9-frame cycle at ~0.95/frame chains ~3; g19 LATE epochs (loss 0.010)
# chain 9-12 under sampling (ep58 127/min c12) -> convergence is the lever;
# the loss floor is set by label conflicts (rung mixing on the shifted labels
# of floating rollouts, one-frame fixture/table disagreements).
# Changes vs g20: ONE rung (--multi-delay "0,1,2,3", the local no-delay rung: no
# rung-mixed labels), 60 epochs (to convergence, gate under sampling), arms:
#   a = independent head + SS (g20b recipe)      b = a + --action-frame-buckets 24
# Gate at live d1 (derived id 0), T=1.0, every 4th epoch late.
# ms_g20 = the g19 champion recipe under the CAUSAL LABEL REBASE + the v3
# recipe changes, aimed at LOCAL play at the tightest rung ("no delay").
#
# PRE-REGISTERED 2026-09-10 (Bradley: "new attempt at a multishining bot
# with v3's recipe ... local play only, no delay").
#
# What changed vs eval_runs/0820_g19_gatesweep/run_g19.sh:
#   * Labels are causal BY CONSTRUCTION (Peppi.causal_pairs; INVARIANTS
#     item 1). All delay numbers are REACTION delay: g19's
#     `--multi-delay "2,3,4"` == "1,2,3" today (same physical labels,
#     same pipeline offset 2 on top). "0" is ADDED = the local no-delay
#     rung. Delay-ids are the new numbers {0,1,2,3}; the Agent derives
#     id = live_frame_delay - 1 for causal checkpoints, so live d1 -> id 0,
#     d3 -> id 2 (== g19's id3 physically).
#   * --head autoregressive --clean-loss (the v3 recipe: joint per-frame
#     heads so down+B lands on the SAME frame; v16a-verdicted loss).
#   * Snippets RE-MINED under the causal parser
#     (eval_runs/0910_snippets_human_causal, --action-delay 1 == g19's
#     ad2). The 0804 file is refused by the drill now (landing-convention
#     expert labels bake in a one-frame-early shift).
#   * The multishine expert table is issued-input now (composed drill
#     labels identical to before at the renumbered delays).
#   * --max-epochs 60 (g19's peak was ep3-8; sweeping 3 rungs x 60 is
#     too slow in 9-min foreground chunks).
#   Unchanged: fixture, rollout pool, opening replays, scramble frames,
#   prev-action dropout, transition weight, queue-as-input depth 4 +
#   delay-id, scheduled sampling 0.5 ramp 10, --awbc, lr 2e-4,
#   --target-loss 1e-9 (no convergence exit), --snapshot-all.
#
# Reads (gate-sweep at live d1, d2, d3 with the DERIVED id; stand-fox FD):
#   R1 any snapshot at d1 >= 300/min -> "no delay" local multishine exists
#      under the causal recipe; that snapshot is the candidate.
#   R2 the best rung: if d3 >> d1 the tight rung is harness-limited
#      (LATENCY_ARCHITECTURE), not recipe-limited; report both.
#   R3 argmax across rungs vs g19's 437.4 c438 (stand FD) — parity or
#      better = the rebase costs nothing on the ms line.
#   NO CROWN from stand numbers (g6 rule); Bradley's local look decides.
#
# Run from repo root inside devenv shell. NO-MIX while training.
set -euo pipefail
cd "$(dirname "$0")/../.."
export EXPHIL_SKIP_NIF_COMPILE=1
OUT=eval_runs/0910_g22_converge
# ARM (2026-09-10, after the first launch died: the drill does not support
# --head autoregressive together with --scheduled-sampling):
#   a = AR head, NO scheduled sampling   (faithful to the v3 recipe)
#   b = independent head + SS 0.5/10     (g19 champion + causal + clean loss)
ARM="${ARM:-a}"
case "$ARM" in
  a) HEAD_ARGS="--head independent --clean-loss"; SS_ARGS="--scheduled-sampling 0.5 --ss-ramp 10"; NAME=ms_g22a ;;
  b) HEAD_ARGS="--head independent --clean-loss --action-frame-buckets 24"; SS_ARGS="--scheduled-sampling 0.5 --ss-ramp 10"; NAME=ms_g22b ;;
  *) echo "ARM must be a|b" >&2; exit 2 ;;
esac
ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp,eval_runs/0802_d2pool/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'
SNIP="eval_runs/0910_snippets_human_causal/snippets.frames"
[ -f "$SNIP" ] || { echo "re-mine snippets first: $SNIP missing" >&2; exit 1; }
echo "=== G22 TRAIN $(date +%H:%M:%S) (g19 recipe, causal labels, AR head, clean loss, rungs 0-3, 24ep, snapshot-all)"
EXPHIL_GPU_MEMORY_FRACTION=0.75 mix run scripts/dagger_drill.exs \
  --expert multishine --fixture test/fixtures/replays/fox_multishine_closed_d1.slp \
  --rollouts "$ROLL" --opening-replays "$OPEN" \
  --snippet-frames "$SNIP" \
  --opp-scramble-frames 12000 \
  --max-epochs 60 --prev-action-dropout 0.6 --transition-weight 2.0 \
  --action-delay 1 --multi-delay "0,1,2,3" --pipeline-offset 2 --queue-depth 4 --with-delay-id \
  $SS_ARGS --awbc \
  $HEAD_ARGS \
  --lr 2.0e-4 \
  --target-loss 1.0e-9 \
  --snapshot-all \
  --out "checkpoints/$NAME.bin" \
  2>&1 | tee "$OUT/train_$ARM.log" \
  | grep -aE "AWBC|COLLAPSE|Converged|diverged|exported|error|Error" | tail -8
[ -f "checkpoints/$NAME.bin" ] || { echo "=== G22 TRAIN FAILED" >&2; exit 1; }
echo "=== G22 TRAIN DONE $(date +%H:%M:%S). Gate-sweep next (foreground, chunked):"
echo "  GATE_DELAY=1 bash scripts/gate_sweep.sh checkpoints/$NAME $OUT/sweep_${ARM}_d1"
echo "  GATE_DELAY=2 bash scripts/gate_sweep.sh checkpoints/$NAME $OUT/sweep_${ARM}_d2"
echo "  GATE_DELAY=3 bash scripts/gate_sweep.sh checkpoints/$NAME $OUT/sweep_${ARM}_d3"
