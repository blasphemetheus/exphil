# g15r (champion recipe, NO awbc, fixed stack) — attribution verdict

Run 2026-08-16 01:06→02:32 (nx `integration` a7497612; runtime-identical
to g16's stack — nx/exla lib byte-equal to 0e487064, only the CUDA
callback refactor differs). Full 60 epochs, NO convergence: final loss
0.0119 (g16 converged 0.00127 @ 58). Checkpoint:
`checkpoints/ms_g15r_gradfix.bin`. Prereg reads in `run_g15r.sh` header.

## Results (strict pipeline scores; deterministic decode, x3 identical)

| Readout | g15 (old grads) | g16 (fix+awbc) | g15r (fix, no awbc) |
|---|---|---|---|
| stand-fox d3 | 430.4/min c426 | 253.6/min c203 | **116.8/min c23** |
| stand-mewtwo | 80.9/min c2 | 109.8/min c14 | **27.0/min c2** |

## Verdict: R1, amplified

**The grad-fix owns the level drop vs g15; AWBC is exonerated AND
carries the fixed-stack runs**: +117% stand-fox (116.8→253.6, chains
23→203) and +307% mewtwo (27.0→109.8) over the no-awbc arm. g16's G1
"fail" was the corrected gradients moving the whole level down, with
AWBC recovering a large fraction.

Caveats (blocking strong quotes of the % numbers):
1. n=1 seed; g15r's non-convergence could be unlucky. A replicate is
   owed before "+117%" hardens. Note the direction though: AWBC arm
   converged in fewer epochs to 10x lower loss — on correct gradients
   AWBC appears to ACCELERATE optimization, not just re-aim it.
2. Structural: the champion recipe's hyperparameters (lr, epochs,
   schedules) were implicitly tuned under the old, mathematically wrong
   while-grads. Fixed-stack absolute levels are not comparable to
   pre-fix history; the recipe wants retuning around correct gradients
   (see HANDOFF 08-16 addendum: retune plan).

No crown implications (g6 rule). Deploy rung = Bradley's blind local
g15-vs-g16; g16 is the best fixed-stack checkpoint by a wide margin on
both readouts.
