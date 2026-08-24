# PREREG — champion-recipe replicates (n=3), written 2026-08-24 before launch

The 0819_g15r2 finding #3 debt, now due (Bradley's pick over new RL
arms): the fixed-grad stack showed ENORMOUS run-to-run variance
(no-awbc fox 116.8 vs 362.5, same recipe; one-epoch collapse mode),
so every single-run arm comparison since is a draw from an unknown
distribution. Before ANY new recipe arm (RTG conditioning, AWBC
knobs, generalist mixes), measure the distribution of the champion
recipe itself.

## Protocol

3 fresh runs of the EXACT g19 command (0820_g19_gatesweep/run_g19.sh:
champion recipe + --awbc, lr 2e-4, 60 epochs, snapshot-all, no early
exit, collapse guard active) → gate_sweep argmax + --confirm each.
Outputs ms_g19r1/r2/r3. Nothing varied — the variance IS the
measurement. Training is already known non-deterministic (0819_g15r2
RC), so no seed knob needed.

## Pre-registered reads

- **R1 (spread)**: argmax-gate fox/min across {r1, r2, r3, original
  g19 = 437.4}. If (max-min)/mean > 0.3, "single-run arm reads are
  uninterpretable" is PROMOTED TO LAW: every future recipe arm needs
  n>=2 or a variance-aware bar. If tight (<0.15), the 08-19 variance
  was the no-awbc stack's, and awbc arms may be read singly.
- **R2 (peak structure)**: argmax epoch + peak width per run. Is
  early peaking (the champion's ep4) typical, or a lottery draw? If
  argmax epochs scatter widely, gate-sweep stays MANDATORY for every
  arm (never trust a fixed-epoch export).
- **R3 (transfer)**: stand-mewtwo at each argmax. The surviving AWBC
  claim (109.8 vs ~0 without) predicts >=60/min consistently; 0/min
  runs would overturn the last standing AWBC benefit.

## Rules

- NO CROWN implications (stand numbers never crown — the g6 rule);
  ms_g19_ep4 stays champion regardless of what r1-r3 gate.
- Collapses are DATA (log epoch + guard action), not failures.
- NO-MIX for the whole program (three train runs + sweeps back to
  back); no lib edits either (the drill re-`mix run`s per stage).
