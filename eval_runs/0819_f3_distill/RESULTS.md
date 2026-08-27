# F3 Route A arms (A2/A3) — results

Run 2026-08-21 15:05–20:57. Prereg: run_f3_arms.sh header (gate-sweep
era amendment) + DISTILL_ANCHOR_SPEC v2. Teacher = ms_g19_ep4. Base
refs: ceiling 435-437 (g19/g20), HOLD bar = sweep argmax >= 390.

## Card

| arm | sweep argmax | fox x3 | mewtwo | d4-id3 | epochs >=300 |
|---|---|---|---|---|---|
| A1 = g19/g20 baselines | 437.4 / 435.4 | 437.4 c438 / 435.4 c436 | 93.9 c1 / 424.4 c422 | 388.4 / 1.0 | ~12 / ~24 of 60 |
| A2 anchor, dose 1x | ep13 438.4 c439 | (confirm pending, sweep truncated ep31 — completion owed) | — | — | 10 of 31 gated |
| **A3 anchor, dose 2x** | **ep8 438.4 c439 (x3)** | **438.4 c439** | **434.4 c433** | **376.4 c372** | **46 of 60** |

A2's sweep died at ep31 to a gate_sweep set-e bug (fixed + made
resumable the same evening); its ep13 438.4 already clears HOLD, and
the remaining epochs re-sweep with the same command whenever the GPU
is free.

## Verdict: BOTH ANCHORED ARMS HOLD — Route A GRADUATES (A4 owed)

1. **The anchor is free at the peak** (A2: 438.4 with dose unchanged)
   and **unlocks dose** (A3: 438.4 with the human-snippet share
   DOUBLED). Per prereg: F3 graduates to a standing recipe lever,
   pending the A4 attribution control (doubled dose, NO anchor —
   queued behind the peak-science arms).
2. **`ms_f3_a3_anchor2x_ep8` is the best all-around checkpoint ever
   measured**: record fox + near-equal mewtwo + working deploy rung in
   ONE artifact. The g19-ep4 / g20-ep13 profile trade (deploy-band vs
   opponent-invariance, each missing the other) does not appear —
   whether the anchor dissolves the trade or ep8 drew lucky needs the
   A4 control + a replicate before believing it as a law.
3. **PEAK-SCIENCE RESULT, TEMPERED after A2's full sweep (08-22
   01:5x):** A3 (anchor + 2x dose) dwelled 46/60 = 77%; but A2
   (anchor, 1x dose) dwelled only 22/60 = 37% — inside the unanchored
   baseline band (20-40%). The anchor ALONE does not explain the
   stabilization; it is anchor + doubled dose (or dwell has large
   run-to-run variance, n=1 per arm). A4 (2x dose, NO anchor) is now
   doubly informative: its dwell separates "dose stabilizes" from
   "anchor x dose interaction". Until then, the stay-on-peak claim
   belongs to the A3 CONFIGURATION, not the anchor mechanism.
4. Interpretive caution: the teacher IS g19-ep4, so some convergence
   toward its behavior is by construction. The mewtwo/d4 gains beyond
   the teacher's own profile (teacher mewtwo was 93.9!) are NOT
   explainable as pure teacher-copying — the doubled human data or the
   anchor-stabilized training found them.

## FINAL ATTRIBUTION (A2 completion + A4 control, 08-22 01:26-04:23)

| arm | argmax fox x3 | argmax mewtwo | dwell (>=300) |
|---|---|---|---|
| A2 anchor 1x | ep13 438.4 c439 | **438.4 c439** | 22/60 = 37% |
| A3 anchor 2x | ep8 438.4 c439 | **434.4 c433** | 46/60 = 77% |
| A4 NO-anchor 2x | ep8 438.4 c439 | **51.9 c1** | 36/59 = 61% |
| unanchored baselines | 437.4 / 435.4 | 93.9 / 424.4 | 20-40% |

- **Peak height (~438): universal** — reached by every configuration.
- **Trajectory stability: DOSE-driven** (2x arms 61-77% vs 1x 20-40%;
  anchor adds maybe +16pts at 2x, nothing at 1x). The F3 premise
  ("composition alone can't absorb more data") does NOT reproduce at
  2x on this recipe — the dose was never toxic. Anchor = harmless but
  not necessary for dose tolerance HERE; its dose-protection claim
  is deferred to higher doses / genuinely off-distribution data
  (the fox-bot program's fight-state corpora).
- **Opponent-invariant argmax profile: ANCHOR-associated** — 2/2
  anchored argmaxes are mewtwo-invariant at full rate vs 0/1 for A4
  and 1/2 for the unanchored baselines. NOT teacher-copying (the
  teacher's own mewtwo was 93.9). Suggestive (small n) but it is the
  only lever observed to produce the all-around profile reliably.
- d4-id3 measured only for A3 (376.4 c372); A2/A4 argmax d4 profiles
  unmeasured.

**Program verdict: F3's value proposition shifts from "dose
protection" to "profile shaping + (with dose) stability", pending
replicates. The all-around checkpoints it minted (a2_ep13, a3_ep8)
stand regardless and join the decider pool.**

## Next

- A4 control queued after peak-science P1/P2 (auto-waiter).
- ms_f3_a3_anchor2x_ep8 joins the decider candidate pool (vs
  ms_g19_ep4) for the next human session — NO CROWN from stand
  numbers (g6 rule).
- A2 sweep completion (resumable) when GPU free.
