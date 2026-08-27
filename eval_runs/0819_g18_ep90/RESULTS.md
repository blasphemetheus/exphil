# g18 (90-epoch @ 2e-4 + awbc) — results, both attempts

Prereg in `run_g18.sh` header. Attempt 1 launched 22:39, attempt 2
(with the GOTCHA #99 collapse guard) 23:05.

## Card

| | trained | stand-fox d3 x3 | stand-mewtwo d3 |
|---|---|---|---|
| g16 ref (2e-4, 60ep) | conv 0.00127 @58 | 253.6/min c203 | 109.8/min c14 |
| attempt 1 | COLLAPSED ep10 (loss 2.0e-6) | 0.0 (dead export) | 0.0 |
| attempt 2 | exited ep13, loss 2.06e-4 (GENUINE) | **419.4/min c415** | 99.9/min c2 |

Attempt 2's trajectory was violently unstable (ep7 spiked to loss
1.01; order-of-magnitude epoch-to-epoch swings) and then landed on a
real optimum at ep13 — an 11x one-epoch drop, correctly NOT flagged by
the collapse guard, and the gates prove it genuine: **best fixed-stack
fox checkpoint ever, essentially at old-grad champion level (g15:
430.4 c426), reached in 13 epochs.** Checkpoint:
`checkpoints/ms_g18_ep90.bin`.

## Score vs prereg (strict)

- G1 >=304: **PASS, crushed** (419.4).
- G2 >=100 hold: **marginal FAIL by the letter** (99.9; chains 2 vs
  g16's 14 — the generalization readout genuinely regressed).
- The registered read "epochs is the lever" is NOT supported despite
  the G1 pass: the run exited at epoch 13 — the 90-epoch knob was
  never exercised. The win is a DRAW, not a lever effect.

## The real verdict of the night (g17 + g15r2 + g18 together)

Recipe-identical or near-identical fixed-stack runs tonight spanned:
dead, 12.0, 103.9, 116.8*, 151.8, 253.6*, 362.5, 419.4 (* = 0814-16
era). **Run-to-run variance dominates every knob tested (lr, epochs,
awbc)**. Single-run reads on this stack are draws from a wide
distribution — including, retroactively, the g16-era attribution
verdicts. The knob-science phase of the retune program should stop
until either (a) n>=3 replicates per arm are budgeted, or (b) the
variance source is found and fixed.

Open suspect for WHY tonight's dynamics differ from 08-14/16 (swings,
two collapses, fast optima — none seen before): the **nx bump**
(integration a7497612 -> f843aa1a, 10 fuzz-campaign commits; nx lib
diff touches defn/expr donatable semantics, reduce/clip/gather
backend dispatch, from_binary bitstring; exla defn.ex output-donation
fix). exphil/edifice don't use donatable, so no confirmed mechanism —
but it is the ONLY stack delta (exphil lib mtimes all pre-g15r;
edifice unchanged). Decisive experiment (GPU-idle, bisect-bench
pattern of GOTCHA #98): rerun one arm at a7497612 and compare
epoch-loss dynamics.

## Recommendations for Bradley

1. `ms_g18_ep90.bin` (419.4 c415 fox) is the best fixed-stack G1
   checkpoint — candidate for the blind play rung alongside
   ms_g15/g16. NO CROWN from stand numbers (g6 rule); its mewtwo c2
   says weak generalization.
2. Pin nx to a7497612 for all training runs until the a7497612-vs-HEAD
   dynamics probe settles the bump question.
3. Replicates-first: n>=3 per arm or no arm at all.
