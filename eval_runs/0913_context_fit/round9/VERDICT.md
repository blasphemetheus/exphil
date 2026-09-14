# round9 — frozen 9-epoch context fit: gate FAILED 13/18 (no live evaluation)

Budget 9 epochs x 486 batches = 4,374 updates (HANDOFF_2026-09-13c's suggested
budget). Final loss 0.00708, still falling ~25%/epoch; the prior proof (no-dropout
round21, same recipe, 7,785-target pool) was at 0.0163 at epoch 9 and reached 1e-4
only at epoch 18. Under-spent, by the prior proof's own curve.

Frozen early gate (18/18 argmax + min joint p >= 0.95 over the first 18 SUPERVISED
targets): 13/18 pass. Fails: 4_cold (17/18, min 0.319), 4_warm (18/18, min 0.748),
900_cold (18/18, 0.852), 1389_cold (18/18, 0.875), 3152_cold (17/18, 0.458).
All fails but two are correct-argmax-but-underconfident rows. Every warm clip
except recovery 4 passes; four COLD clips regressed vs the round21 baseline
(cold familiar 18/18 >= 0.997) — the warm copies took the budget the cold ones
used to have. Recovery 4 (reflector hold, action 363, X-tap alternation at
af 11/13/16) is the hardest window in both histories, as it was in every prior round.

Next: round21 = same recipe/pool/init, 21 epochs (10,206 updates), declared before
launch. Not a within-run extension: this directory stays as the failed attempt.
