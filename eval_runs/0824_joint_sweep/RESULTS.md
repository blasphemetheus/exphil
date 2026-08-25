# Joint-sweep transfer selection + cross-character transfer structure

Near-ceiling epochs (fox >= 434/min) from the three replicates, gated
vs stand mewtwo / yoshi / popo(ICs); champion row added for
comparison. Single 60s gates — treat every number as a noisy draw
(r3_ep4's mewtwo read 255.6 last night, 344.5 today).

| checkpoint | fox | mewtwo | yoshi | popo |
|---|---|---|---|---|
| r1 ep16 | 438.4 | 25.0 (c5) | 78.9 (c1) | 177.8 (c141) |
| r1 ep17 | 438.4 | 23.0 (c1) | **420.4 (c417)** | 77.9 (c2) |
| r1 ep50 | 434.4 | 36.9 (c4) | 350.5 (c319) | 76.9 (c14) |
| r2 ep5 | 437.4 | 73.9 (c1) | 71.9 (c1) | 99.9 (c1) |
| r2 ep22 | 438.4 | 98.9 (c5) | 98.9 (c6) | 97.9 (c12) |
| r2 ep26 | 435.4 | 79.9 (c9) | 46.9 (c2) | 27.0 (c2) |
| **r2 ep39** | 435.4 | **206.7 (c144)** | **426.4 (c419)** | **434.4 (c433)** |
| r2 ep45 (its fox argmax) | 439.4 | 23.0 (c2) | 36.9 (c4) | 12.0 (c2) |
| **r3 ep4** (its fox argmax) | 439.4 | **344.5 (c320)** | **273.6 (c230)** | **362.5 (c348)** |
| r3 ep5 | 434.4 | 90.9 (c1) | 82.9 (c1) | 71.9 (c1) |
| **CHAMPION ms_g19_ep4** | 437.4 | 92.9 (**c1**) | 97.9 (**c1**) | **439.4 (c440)** |

## Answer to "would mewtwo-good also be yoshi/ICs-good?"

**Partially — a broad opponent-invariance axis EXISTS, but so do
per-character specialists.** Broad epochs (r2 ep39, r3 ep4) chain vs
ALL THREE; specialist epochs are common (r1 ep17: yoshi 420 / mewtwo
23; the CHAMPION: popo 440 / mewtwo+yoshi chain-1). Asymmetry worth
keeping: every mewtwo-good epoch here is broad, while yoshi-good and
popo-good epochs are often narrow — mewtwo looks like the HARDEST
opponent-feature test (furthest off the fox manifold), so passing it
predicts the others, not vice versa. **Selection protocol: gate
mewtwo among ceiling epochs** (cheapest sufficient signal), spot-check
a second character.

## Selection verdict

**`checkpoints/ms_g19r3_ep4.bin` is the transfer-selected candidate**:
fox at the ceiling (439.4, tied best) AND broad transfer
(344/274/363 with real chains), vs the champion's popo-only transfer.
Runner-up r2 ep39 (slightly sub-ceiling fox, broadest yoshi/popo).
NO CROWN from stand numbers (g6 rule) — if this matters for deploys,
the path is a blind human decider r3_ep4 vs ms_g19_ep4 (cheap now).

Fox-argmax alone is a BAD selector confirmed: r2's fox argmax (ep45)
is its WORST transfer row; r3's happened to be its best. The joint
protocol (fox ceiling ∩ mewtwo gate) costs ~10 extra gates per run.
