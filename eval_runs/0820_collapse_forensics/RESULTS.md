# Collapse forensics — results (2026-08-20 morning)

Direction: isolate the nx bug behind the GOTCHA #99 collapses.
Machinery built: `--init-from` (warm start), `--collapse-forensics`
(scene traps: batch loss <1e-5 halt+dump; >0.3 spike dumps),
`--replay-scene` (offline single-step replay under any nx checkout),
per-batch loss series logging, 2-epoch cross-stack overlay runs.

## What the experiments showed (each one a negative for the nx-bug theory)

1. **Single-step replay (spike1, batch loss 0.319)**: wild stack
   0.327/0.328, calm stack 0.341/0.328 — IDENTICAL within GPU
   nondeterminism; post-params match the recording (1.5e-5). The spike
   batch is genuinely hard; no per-step miscompute.
2. **Per-batch loss distributions (2 warm epochs, ~12k batches/stack)**:
   calm p50/p90/p99/max = .035/.19/.77/2.89; wild = .040/.21/.82/3.19.
   Statistically identical heavy tails — the calm stack ALSO throws
   1.67-loss batches. The "calm vs wild" era signature was built on
   the LAST-BATCH epoch readout (dagger_drill prints the final batch's
   loss, not an average) — a tail-draw lottery, not dynamics.
   Probe p1814's WILD label is hereby DOWNGRADED to unproven.
3. **Cumulative 2-epoch damage**: overlay exports gate 85.9 (calm) and
   88.9 (wild) — equal.
4. **The <1e-5 collapse batches remain unexplained but RARE**: today's
   23k batches floor at ~4e-3 on both stacks — the recorded 2.5e-8 /
   2.0e-6 events sit 5+ orders below any organic batch loss, so they
   are degenerate computations or corrupted state, NOT tail draws. But
   the tally (2 events in ~150 HEAD epochs vs 0 in ~200 calm epochs)
   is Poisson-consistent with equal rates — not attributable to the
   nx bump on current evidence.

## THE ACTUAL DISCOVERY: unstable peaks + a mediocre attractor

`ms_g15r2_latest` (ep50) re-gated TODAY at 362.5 c353 (intact). Two
epochs of continued training took it to ~87 ON BOTH STACKS. Meanwhile:
nxpin (from scratch, 83 epochs, loss 0.0011) gated 87.9; the overlay
runs (loss ~0.019) gate ~87; ep50 (loss 0.0105) gates 362.5; g18a2
(exited ep13, loss 2e-4) gated 419.4.

**Chain skill lives on TRANSIENT PEAKS that continued training
destroys, and final loss is ANTI-correlated with behavior in this
regime.** Where a run STOPS — convergence-exit timing, itself driven
by last-batch lottery — determines the gate number. This explains the
entire "variance dominates" night without any nx bug: every run
passes peaks; the lucky ones (g18a2 ep13, g15r2's ep50 snapshot) got
sampled there.

## Recipe implication (the actionable one)

STOP selecting checkpoints by loss. Select by BEHAVIOR:
- Save a policy snapshot EVERY epoch (3.8MB each; 90 epochs = 342MB).
- After the run, gate-sweep all snapshots (sync headless gate ~30s
  each; ~45 min for 90) and export the argmax-gate epoch.
- The existing best-by-loss export and the convergence exit both
  actively select AGAINST chain skill late in training.

## nx verdict for Bradley

No reproducible stack difference found at step, distribution, or
2-epoch cumulative granularity. #1814 unproven; the "bump implicated"
call from 0819_nxpin_probe is RETRACTED (readout artifact — the run-
level facts it compared were last-batch lotteries). The rare <1e-5
events remain open; a long calm-stack trap run could settle whether
they are HEAD-only, but the peak/attractor discovery says the GPU is
better spent on behavioral-gating recipes. Training can return to
integration HEAD unless/until a <1e-5 event is caught and replayed.
