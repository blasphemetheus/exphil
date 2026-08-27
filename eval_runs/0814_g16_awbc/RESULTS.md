# g16 (champion recipe + --awbc) — gate results & verdict vs prereg

Run 2026-08-14. First launch (02:02) aborted at epoch 17/60 — nx 0.13.1
merge made Nx.while backward O(seq_len²) (nx#1785), 27 min/epoch.
Bisected + fixed via `unroll: :static` (GOTCHA #98); relaunched 10:42,
converged loss=0.00127 @ epoch 58, ~82 s/epoch. Checkpoint:
`checkpoints/ms_g16_awbc.bin`.

**CONFOUND (material, discovered mid-arc):** static unroll gives
provably-correct chain-rule gradients; every prior RNN checkpoint (g15,
b1/b2 arms) trained on the pre-#1785 while-grad rule, which upstream
fixed as INCORRECT. g16 is therefore champion+awbc+grad-fix — two
changes, not one. Prereg was written before this was known.

## Gates (strict counts verified via analyze_shine_source: hit-ind 0 everywhere)

| Gate | g15 ref (08-08) | g16 | Prereg bar | Verdict |
|---|---|---|---|---|
| G1 stand-fox d3 x3 | 430.4/min c426 | 253.6/min c203 (x3 identical, deterministic FD) | hold >=390 | **FAIL** (−41%) |
| G2 stand-mewtwo d3 | 80.9/min c2 | **109.8/min c14** | no collapse | **PASS** (+36%, chains 2→14) |
| G3 YS x3 | 100.9/84.9/95.9 c13/8/19 | 93.9/**146.8**/89.9 c2/**66**/c2 | collapse bucket only | no collapse → PASS (n=3; r2 c66 = best YS chain on record for this lineage; r1/r3 c2 breaks are new) |
| G4 rung-0 | 4.28 (scramble caveat) | 4.16 | big rise = flag | PASS (no rise) |

## Verdict

**Strict prereg: g16 does NOT graduate — G1 hard-fails the 390 hold.**
`ms_g15_oppmask_full` remains champion. Registered prediction
("equal-or-better rate, cleaner breaks") did not hold on G1.

But the pattern is NOT a simple regression: the specialist number
dropped while BOTH generalization readouts improved sharply (mewtwo
+36% with 7x chain, YS best-ever c66). Consistent with either (a) the
grad-fix moving the optimum, or (b) AWBC's regional reallocation
(interp: jumpsquat up-weighted, "no shine ahead" starved), or both —
the confound makes attribution impossible from this run.

> **RESOLVED 2026-08-16 — verdict R1, amplified**: g15r ran (fixed
> stack, no awbc): stand-fox 116.8/min c23, mewtwo 27.0/min c2. The
> grad-fix owns the drop; AWBC exonerated and CARRIES the fixed stack
> (+117% fox / +307% mewtwo vs no-awbc). Full card + caveats:
> `eval_runs/0814_g15r_gradfix/RESULTS.md`.

## Recommended next step (one change at a time, restored)

Disambiguation arm: retrain the champion recipe WITHOUT --awbc on the
fixed stack (same nx tip + static unroll) = "g15r". Then:
- g15r ~ g15  → the drop is AWBC's; AWBC's offline story weakens.
- g15r ~ g16  → the drop is the grad-fix; AWBC keeps its arms-validated
  win and g16's generalization gains stand on their own.
Cost: ~80 min wall. Bradley's blind local g15-vs-g16 decider remains
the promote rung either way; the mewtwo/YS gains argue for playing g16
even with G1 failed.
