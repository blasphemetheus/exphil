# Multishine diagnosis: the training data never had the behavior

**2026-07-23.** Bradley watched the Fox multishine policy and the scripted
teacher side by side and said: "the teacher doesn't cleanly multishine — it
shines and jumps and shines on the way back down." He was right, and it
overturned a wrong conclusion I had reached from a broken metric. This
documents the real finding so the earlier (wrong) writeup doesn't mislead.

## The wrong conclusion, and why

I built `ExPhil.Eval.ShineChain` and measured:
- fixture "max 73", teacher live "max 370" → concluded the behavior IS in the
  data and the teacher reproduces it, so the policy's failure was a **training
  problem** (exposure bias / prev-action dropout).

**That was a metric artifact.** The first version of `family/1` lumped
GROUNDED reflector (action 360–363) and AERIAL reflector (365–368) together as
"a shine," and treated any jumpsquat as continuing the chain. So the sloppy
shine → **full jump** → air-shine-on-the-way-down → land → shine loop counted
as one long "clean chain." The 73 and 370 were mostly air-shines. Bradley's
eyes counted air time — the thing that actually distinguishes a real
multishine — which the metric could not see.

**Lesson:** when a metric disagrees with direct observation of the same
behavior, doubt the metric first. I stated "the teacher sustains flawlessly"
with confidence; it does not.

## The real numbers (honest metric: grounded vs aerial)

`ShineChain` now splits grounded (360–363) from aerial (365–368) reflectors; a
clean chain is consecutive GROUNDED shines whose only gaps are jumpsquat, and
it BREAKS the moment Fox goes airborne.

**FIXTURE `test/fixtures/replays/fox_multishine_closed.slp`** (the source the
whole multishine drill is built on):
- clean grounded multishine: **max = 1, mean = 1.0, sustained(≥5) = 0**
- **grounded_fraction = 0.246** (275 grounded vs 844 aerial shine frames)
- **132 of 134 chains ended by `air_shine`** — Fox shines once on the ground,
  then jumps and shines in the air, every single time.

The target behavior — a sustained grounded multishine — is **not in the
training data at all.** Max grounded chain length is one.

## The chain of causation (imitation was FAITHFUL)

```
fixture (grounded_fraction 0.25 — a shine-jump-airshine loop, NOT a multishine)
  -> MultishineExpert table (modal inputs keyed on the sloppy fixture)
     -> policy (BC clones the sloppy teacher)
        -> observed: shine, jump, air-shine on the way down; ~1-2 grounded
```

Live demo confirmed policy ≈ teacher: P1 model max grounded = 2, P2 teacher
max grounded = 1, both spraying empty hops. BC did its job — it faithfully
reproduced a flawed teacher. This is **not** a training/exposure-bias problem.
It is a **rung-1 data problem at the root**: the supervision does not contain
the behavior.

## Root of the sloppy fixture

`scripts/record_multishine.exs` has two modes:
- `:multishine` (open loop) — by DESIGN does an aerial shine every 12 frames
  ("aerial shine on the first airborne frame after 3 frames of jumpsquat").
  Not a grounded multishine, on purpose.
- `:closed_loop` — INTENDS to stay grounded (jump-cancel from action_frame 4,
  pure function of state). But `fox_multishine_closed.slp` came out 75%
  aerial, so **the closed-loop jump-cancel timing is wrong** — it is meant to
  cancel the jump and stay grounded, and instead mostly goes airborne. The
  tech skill was never correctly encoded, at the very source.

## The fix (next work, NOT done here)

1. **Produce a genuinely clean grounded-multishine fixture.** Fix the
   `:closed_loop` jump-cancel timing in `record_multishine.exs` so Fox stays
   grounded (the JC window is af 4 while grounded; the current logic misses
   it), verify with `ShineChain.summary` that `grounded_fraction ≈ 1.0` and
   `max_length` is large BEFORE saving it as the fixture. A human re-record is
   the fallback, but scripting is deterministic and doesn't require perfect
   human execution.
2. Rebuild the expert table from the clean fixture (`MultishineExpert.from_frames`).
3. Retrain the multishine drill; gate on `ShineChain` (B1 grounded entry,
   B2 grounded sustain ≥5), NOT on the old event counters.

## Tooling built during this diagnosis

- `lib/exphil/eval/shine_chain.ex` (+ tests) — grounded-vs-aerial chain
  metric; `summary/2` reports `grounded_fraction`, the number that catches
  airborne "shining."
- `scripts/demo_expert.exs` — drive Dolphin live from a scripted expert (the
  teacher), with a ShineChain self-diagnosis. Answers "can the teacher even do
  it?" without a policy in the loop.
- `scripts/demo_vs_teacher.exs` — policy on P1 vs scripted expert on P2, same
  match, ShineChain summary for both. The side-by-side that made the gap
  visible.

## GOALS.md Track B correction

GOALS.md says Track B is an EXECUTION problem ("the policy can't reproduce a
known sequence"). That framing is now wrong: the sequence it's imitating is
itself not the target. Track B's first gate is really **produce a clean
grounded-multishine fixture** (grounded_fraction ≈ 1.0); only after that does
"can the policy reproduce it" become the question.
