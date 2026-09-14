# g26 pool, equal budget: off-loop HOLD vs DROP labels (2026-09-14)

The comparison RECOVERY_LABEL_CONFIRMATION asked for. Root:
`eval_runs/0914_g26_hold_vs_drop/` (launcher `run.sh`, `progress.log`).

## Design

Two arms, identical to the g26a recipe (`eval_runs/0912_g26_phase/run_g26.sh`:
g24a pool + re-mined snippets, independent head + scheduled sampling, rungs
3/4/5 with physical ids, queue 4, 40 epochs, snapshot-all, current numerical
epoch guard) except ONE flag:

- **hold** (`--off-loop-labels hold`): the g26a rule. Off-loop expert frames
  get the recovery input held at every shift k.
- **drop** (`--off-loop-labels drop`, the 09-13 default): the expert abstains
  off the loop; those frames are omitted at k > 0. NO recorded recoveries
  exist at rungs 3/4/5, so this arm has no delayed recovery supervision at all.

Pool audit at shifts 3/4/5: 0 conflicts under both. Under drop the pool
shrinks to the loop:

| source | frames | kept under drop |
|---|---:|---:|
| fixture | 8,102 | 8,097-8,099 |
| rollouts | 146,660 | 22,543 (15%) |
| snippets | 12,489 | 358 (3%) |
| openers | 93,687 | 15,592 (17%) |

~85% of what the g-line trained on at its rungs was off-loop states carrying
the held projection (measured wrong 18/21 at shift 4).

Final loss: hold 5.4e-3, drop 2.2e-4.

## Readout at reaction 4 (physical id 4)

Stand-fox floor (sync headless FD, T=1.0, one run per snapshot):

| | ep30 | ep40 | refs |
|---|---:|---:|---|
| hold | 230/min c202 | 371/min c367 | g26a 67/min c4 (worst); ep57 296/min |
| drop | **391/min c392** | 376/min c377 | |

CPU gate (async, level-1 CPU, 2 x 90 s, columns: frames, onsets,
self-initiated, hit onsets, shines/min, max chain):

| | run 1 | run 2 | refs |
|---|---|---|---|
| hold ep40 | 44.8/min **c1** | 51.2/min **c1** | g26a ep38 45-50/min c2-8; ep57 56-71/min c3-13 |
| drop ep30 | 74.2/min **c23** | 54.4/min **c15** | |
| drop ep40 | 8.3/min c14 | 44.2/min **c52** | |

Coverage map (teacher-forced, 43 cells, delay-id 4):

| | baseline | mirror | worst opp_action | worst opp_char |
|---|---:|---:|---:|---:|
| hold | 0.999 | 0.874 (min 0.177) | 0.804 (min 0.060) | 0.932 |
| drop | 0.999 | **0.999** | **0.998** | **0.999** |

Drop is flat at 0.998-0.999 on every axis, including the mirrored stage
that was red for ep57, g25a, and g26a.

## Verdict

**Drop wins or ties every readout; the held off-loop labels were the
g-line's defect.** On the CPU gate the chain goes from 1 (hold) to 14-52
(drop) at the same budget, with no recovery labels of any kind at these
rungs. The technique floor is 371-391/min under both arms, well above
g26a's 67, which says g26a's collapse was not the labels alone: that run
also trained under the old low-loss guard (restore/retrain loops, stopped
by hand at epoch 48), and this rerun of its recipe under the numerical
guard does not reproduce the floor loss. The share between guard damage
and labels is not separated here; only the label effect is.

Caveats: single day, 1-2 runs per cell, stand floor is one run per
snapshot, the drop arm's ep40 run 1 (8.3/min, chain 14) is an outlier not
investigated. Descriptive, not a decider. No crown from stand numbers (g6
rule); nothing installed.

## What this means for the scaled bot

1. Train the g-line with `--off-loop-labels drop` (the default). The
   audit's dropped-frame counts are the number recorded recoveries must
   cover at each rung.
2. Add recorded teacher recoveries at rungs 3/4/5 (`prepare_recorded_context
   --delay k`) as the ONLY delayed recovery supervision. The drop arm
   chains 14-52 vs the CPU with none; that is the floor to beat.
3. Rerun the g26a snapshots' health forensics against this pair if the
   guard-vs-labels split matters for the record; it does not gate the
   scaled run.
