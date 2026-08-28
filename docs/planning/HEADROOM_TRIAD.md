# The headroom triad — a standing test for "where is the ceiling right now?"

**Status:** designed 2026-08-28, not yet built.
**Purpose:** answer the recurring question *"given the current state,
what is the highest-return next investment?"* with three numbers in the
same unit instead of an argument.

This is deliberately a **standing instrument**, not a one-off
experiment. The answer changes as the project moves, so the triad is
meant to be re-run at every fork where an expensive commitment is on the
table. Its output is a ranking of where the ceiling actually is.

---

## The question, and why it keeps coming back

As of 2026-08-28 the interp program concluded that fox_gen_v1 "contains
the behavior but lacks a selection rule" (INTERP_GEN_V1 G1/G2/G4/G7).
That is well-evidenced and it points at the value model.

But it is an **inference, not a measurement**. A competing story fits
the same observations: v1 is ~3M params on ~90M frames with validation
loss still descending at epoch 10 — textbook underfit. "The model holds
a broad distribution" is what *both* a well-calibrated model and an
undertrained one look like.

The two stories point at completely different work:

| story | the ceiling is | the investment |
|---|---|---|
| **Selection** | the decode/selector | value model, Best-of-N |
| **Capacity** | model size / training | bigger v2, longer training |
| **Data** | corpus coverage | more games, curation, yeti ingest |

Guessing wrong costs a training run at minimum (~11h) and a research
direction at worst. And this same fork will reappear after v2, after the
value model, after every corpus change. Hence a reusable instrument.

## Why the agreement metric we already have cannot answer it

`scripts/eval_policy_on_fixture.exs` reports offline fixture agreement,
and on the 10-epoch sweep it was **0.974, flat across all ten epochs**
(HANDOFF_2026-08-26 §7). It has zero ranking power.

The reason matters for the design: **most frames are trivial.** Melee is
mostly holding a direction, waiting, and being in an animation you
cannot act out of. A metric averaged over all frames is dominated by
frames where every policy agrees and nothing is at stake. It saturates,
and it saturates *before* it reaches the frames anyone cares about.

So every leg of the triad is computed on **decision frames only**.

## Decision frames

Use the existing labeller, `ExPhil.Situations` — the same one G1's
entropy map is built on. It carries a registry of 30+ labels
(`:conversion_open`, `:pummel_throw_decision`, `:tech_chase`,
`:edgeguard`, `:shield_pressure_ours`, `:being_tech_chased`, ...)
encoded as a u64 bitmask per frame (`lib/exphil/situations.ex`).

A frame counts as a decision frame if **both**:

1. it carries at least one non-geometry situation label (the umbrella
   `:neutral` / `:advantage` / `:disadvantage` children are the
   interesting set), **and**
2. the human's action *changes* at or within a few frames of it — the
   frames where the expert actually did something, not the ones where
   they were committed to an animation.

Report the decision-frame count alongside every result. If it is a tiny
fraction of the corpus, say so — the numbers below are conditional on it.

---

## Leg S — selection headroom (nearly free)

**The measurement: pass@k versus pass@1 on decision frames.**

For each decision frame, draw `k` independent samples from the policy at
the deploy decode. Then:

* **pass@1** — probability a single sample matches the human's action.
* **pass@k** — probability *at least one* of k samples matches.

**`pass@k − pass@1` is the selection headroom**: the accuracy a *perfect*
selector could recover from a distribution the model already produces,
without changing a single weight.

The reading:

* **pass@k ≫ pass@1** → the model contains the right action and fails to
  pick it. Selection is the ceiling; a critic has real room. This is the
  quantitative version of the claim the interp program made
  qualitatively.
* **pass@k ≈ pass@1, both low** → the right action is not in the
  distribution at all. No selector can help. The ceiling is capacity or
  data, and Legs C/D decide which.
* **pass@k ≈ pass@1, both high** → the model is already right at the
  decision points and the failure is elsewhere entirely (closed-loop
  drift, harness, delay). Look outside this triad.

Matching needs a tolerance, because exact stick-bucket equality is too
harsh: buttons match exactly on the pressed set; sticks match within a
bucket distance (start at 1 bucket of 17); shoulder within one bucket.
Report the tolerance with the number — it is part of the metric, and
changing it changes the result.

Suggested `k` = 1, 2, 4, 8, 16. The *curve* is more informative than any
single k: a curve that is still climbing at k=16 says the distribution is
broad and a selector has a lot to work with; one that flattens at k=2
says there is essentially one alternative.

**Cost: no training, no Dolphin, no GPU beyond inference. Existing
replays, existing checkpoints.** This is the cheap leg and it is often
decisive on its own.

## Leg C — capacity headroom (costs training)

Train the current recipe at two or three sizes (e.g. ~3M / ~12M / ~40M
params) on a **fixed** data subset for a **fixed** step budget, and
compare **pass@1 on decision frames**.

`pass@1(bigger) − pass@1(current)` is the capacity headroom.

Fixed data and fixed steps is what makes this a capacity measurement
rather than a compute measurement. If pass@1 is still climbing with size,
the model is the ceiling.

## Leg D — data headroom (costs training)

Fix the model size, train on 25% / 50% / 100% of the corpus, compare
**pass@1 on decision frames**. This is a learning curve.

`pass@1(100%) − pass@1(50%)` extrapolated forward is the data headroom.
A curve still rising steeply at 100% says more games (yeti ingest,
D3) or better-targeted games (defensive/disadvantage curation) pay. A
curve that flattened by 50% says more of the same data is worthless and
only *different* data helps.

---

## The economics: run the cheap leg first

Leg S costs hours of offline compute. Legs C and D cost multiple ~11h
training runs each. So the protocol is staged:

1. **Run Leg S.** If selection headroom is large, invest in the selector
   and stop — you have your answer for the price of an afternoon.
2. **Only if Leg S says the behavior is not in the distribution**, pay
   for Legs C and D to find out whether it is the model or the corpus.

This ordering also matches the project's standing bias: the cheapest
instrument that can decide should run first.

## Validate the instrument before trusting it (required)

The triad must reproduce a **fact already known** before any of its
numbers are used to make a decision. The available known fact is strong
and free:

> Ten epochs of val-loss improvement bought **no visible behavior
> change** — ep1 played about as well as ep10 (0.49 vs 0.50 approaches/min,
> 29% vs 35% conversion), and the live sweep could not rank the epochs.

So: **compute pass@1 on decision frames for ep1 … ep10.** It should be
roughly *flat*, matching behavior. If it instead climbs steeply, the
metric is tracking the same thing val-loss and fixture agreement track —
something behavior does not care about — and it must be fixed (tighter
decision-frame definition, different tolerance) before it is used to
rank anything.

The checkpoints for this already exist. This calibration run is
mandatory, not optional; a decision instrument that has never been
checked against a known answer is just another number.

## Honest limits

* **pass@k is an upper bound.** It assumes a perfect selector. A real
  critic recovers some fraction of that gap, never all of it. Treat the
  number as "the most a value model could possibly buy".
* **It is open-loop.** Teacher-forced on human trajectories, so it cannot
  see closed-loop drift — the failure mode that killed the argmax runs.
  The triad ranks *headroom*; the live bracket remains the confirmatory
  rung, and the human look remains the crowning rung (the g6 lesson).
* **Matching a master's exact action is not the only correct play.** Melee
  has many reasonable options per frame, so absolute pass@1 will look
  low and should never be read as a skill score. Only the *differences*
  and the *pass@k − pass@1 gap* are meaningful.
* **The correlation between pass@1 gains and behavior gains is unproven**
  in this project. The calibration above is the first test of it. Until
  it passes, the triad is a hypothesis-ranker, not an oracle.

## Re-run triggers

Run the triad before committing to any of these:

* a new training generation (v2, v3, …) — to choose its recipe;
* the value model / RL escalation — to confirm selection is still the gap;
* a corpus expansion (yeti ingest, low-tier data) — to confirm data is
  the gap before paying for it;
* any time an expensive direction is being argued about and the argument
  is not settling.

## Results ledger

Append one row per run. Never delete a row — the point of a standing
instrument is the trend.

| date | subject | decision frames | pass@1 | pass@16 | **S headroom** | **C headroom** | **D headroom** | verdict |
|---|---|---|---|---|---|---|---|---|
| _(pending)_ | fox_gen_v1 ep10 | | | | | not run | not run | |
