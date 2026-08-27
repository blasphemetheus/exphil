# g20 (g19 exact replicate) — results

Run 2026-08-20 23:09 → 08-21 13:25 (hibernation mid-sweep; setsid
pipeline survived; ep4's gate straddled the freeze — sync frame-locked,
value plausible, re-gate if it ever matters). Prereg in `run_g20.sh`.
Note: decider games ran during epochs ~10-28 (GPU contention) — wall
time only; gradient math unaffected.

## Verdict: R-WANDER — with the ceiling STABLE and the shape NOT

| | g19 | g20 |
|---|---|---|
| argmax | ep4: 437.4 c438 | ep13: 435.4 c436 (x3) |
| skill onset | ep3 | ep8-9 |
| primary peak | WIDE early (ep3-8 all >=380) | scattered: ep8-9, 13, 15-16, 18-20, 24-28, 32-33, 42, 51, 54-58 |
| terminal decay | yes (nothing >=180 after ep42) | **NO — ep54/55/58 ~432, ep60 331** |
| epochs >=300 | ~12, concentrated early | ~24, spread over the whole run |

Two-attractor structure in BOTH runs (epochs are either ~90-180 or
~330-437, almost nothing between), but WHERE the trajectory sits is
stochastic per epoch and per run. g19's "peak early then permanent
decay" was one draw; g20 oscillates to the end. **"Continued training
destroys peaks" is DEMOTED to "the trajectory hops between a mediocre
attractor and a ~435 ceiling, unpredictably; g19's terminal decay was
not a law."** What IS reproducible: the ~435 ceiling height (437/435),
and gate-sweep finding it (2/2).

## The generalization-profile shocker (argmax confirm gates)

| gate | g19_ep4 | g20_ep13 |
|---|---|---|
| stand-fox d3 x3 | 437.4 c438 | 435.4 c436 |
| stand-MEWTWO d3 | 93.9 c1 | **424.4 c422 — ALL-TIME MEWTWO RECORD by ~4x** (prior best: g16 109.8 c14) |
| real d4 + id-3 override | 388.4 c389 | **1.0 c1 — DEAD** |

Same fox-d3 ceiling, OPPOSITE generalization profiles: g19_ep4
covers the d4 deploy rung but is opponent-sensitive; g20_ep13 is
opponent-INVARIANT (the mewtwo penalty every prior checkpoint paid is
just gone) but its id-3 mode does not execute at real d4 at all — the
FIRST fixed-stack counterexample to the id-3-universal-executor rule.
Peak checkpoints are not one thing; each peak visit crystallizes a
different trade. Deploy stories: g19_ep4 = d3 local + d4-id3 netplay;
g20_ep13 = d3 local only (but vs any character?— stand-only evidence).

## Program implications

1. Gate-sweep stays mandatory (fixed-epoch export = coin flip), and
   the sweep's argmax should be scored on a MULTI-GATE profile (fox +
   mewtwo + d4-id3), not fox alone — g20's fox-argmax happened to be
   a mewtwo monster, but a fox-only selector cannot see such trades.
   Cost: 2 extra 30s gates on the top-3 epochs only.
2. Knob arms are readable only against peak STATISTICS (ceiling,
   time-above-300, profile mix), n>=2 sweeps per arm.
3. Candidates for the human decider now: ms_g19_ep4 (deploy-band
   coverage) and ms_g20_ep13 (opponent-invariance). Both stand-only.
   NO CROWN without the blind human rung (g6 rule).
