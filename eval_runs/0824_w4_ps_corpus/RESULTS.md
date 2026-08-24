# 0824 W4 rung 2 — PS transformation: STAGE-BLIND, same signature as FoD

Companion to eval_runs/0824_w4_fod_corpus/RESULTS.md (the FoD height
probe). Question: does the champion trunk know the CURRENT Pokemon
Stadium transformation? The transformation moves the whole collision
landscape and none of it is a network input.

**Corpus**: 3 local unfrozen PS games (240s each; the new
`--no-frozen-stadium` flag — exphil's menu path had silently played
FROZEN PS forever, the libmelee MenuHelper default). 41.2k frames;
transform coverage: fire 4,079 / grass 4,127 / rock 1,871 / water
2,298 / normal 29k. Labels: forward-filled stream `stadium_type`
(events announce each layout incl. the type-5 normal revert — the
same semantics the RAM digit carries).

| probe | bal_acc | note |
|---|---|---|
| trunk | 0.202 | below majority 0.333 |
| input (leakage floor) | **0.375** | above majority — positions on transformed terrain leak layout |
| shuffle floor | 0.124 | |

**Verdict: STAGE-BLIND — and the FoD signature replicates exactly:
the instantaneous input carries stage-layout information and the
recurrence DISCARDS it** (trunk below its own input on both stages).
That's now a two-stage pattern, not a one-off: the champion models no
stage-internal state at all.

**Caveats**: n=3 replays; by-replay split leaves grass/rock absent
from eval (per-class :absent handled); the key trunk-vs-input
comparison is same-data and split-independent. Multishine-centric
play as before.

**Consequence**: both stage-internal feature candidates are now
justified AND fully provisioned by the 08-24 address work — FoD
height f32s ×2 and the PS transform class — labels in every replay
(peppi), live values RAM-readable. One embedding-config addition
covers both. Script: scripts/interp_w4_ps_transform.exs.
