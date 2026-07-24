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

## Attempted fix (2026-07-23, later): B-timing sweep — did NOT crack it

Made `record_multishine.exs` `:closed_loop` JC-shine frame sweepable
(`MULTISHINE_JC_FRAME`) + automatable (built-in port-2 `stand` dummy clears
CSS, `MULTISHINE_HEADLESS=1`), and swept the shine-out frame 0→4, measuring
each with `ShineChain`:

| JC frame | grounded_fraction | clean max chain | outcome |
|----------|-------------------|-----------------|---------|
| 0, 1, 2  | **1.0** (0 aerial) | **1** | 31 empty hops — shine, FULL JUMP, land, shine |
| 3 (old)  | 0.365 | 1 | aerial shines |
| 4        | 0.41  | 1 | aerial shines |

Two findings:
1. **JC frame ≤2 eliminates the AERIAL shine** (grounded_fraction → 1.0). So
   the "shines on the way down" is fixable — press the shine earlier.
2. **But it does not CHAIN.** af 0/1/2 give byte-identical results (93 ground
   frames, 31 empty hops, max chain 1): Fox shines once grounded, then does a
   FULL 20-frame jump (empty hop), lands, shines again. The down+B during
   jumpsquat is **not shine-cancelling at all**, regardless of frame.

Jumpsquat here is 3 frames; pressing at af=0 lands (with +1 bridge latency)
inside the cancel window and STILL fails — so it is **not purely latency**.
The frame-perfect shine-out-of-jumpsquat cancel does not reproduce through
this input path by sweeping the B frame. This is the same wall the original
author hit (the `:closed_loop` code carries "take-5/take-6" trace comments).

**Did NOT retrain.** A fixture of single grounded shines would teach
shine→jump→land→shine — no better than today. Retraining is gated on a
fixture that actually chains (`grounded_fraction ≈ 1.0` AND `max_length ≥ 5`).

## Deeper investigation 2026-07-24: latency=0, the jumpsquat cancel is the wall

Investigated the two proposed fixes — bridge latency compensation and the
closed-loop mechanic — empirically.

**Bridge input latency = 0 (measured, `scripts/latency_probe.exs`).** From a
standing Fox, a single-frame down+B produces the grounded reflector on the
very next observed step (5/5 trials, "N+0"). `console.step()` flushes the
controllers before reading, so an input decided from frame N lands on N+1 —
the minimal unavoidable observe-act frame, no extra. **There is no excess
latency to compensate. The "+1 bridge latency" note in the old closed_loop
comments was wrong. Latency compensation is a dead end.**

**The jumpsquat shine-cancel does not register through the pipe input — any
input strategy** (`scripts/jumpsquat_probe.exs`, exhaustive):

| strategy | result |
|----------|--------|
| held B through jumpsquat | full jump, then nothing (held B = no edge) |
| fresh B edge at jsquat af 1 | full jump |
| buffer X+B on the JC frame | full jump → **aerial** shine (input reaches, but after takeoff) |
| open-loop rhythm sweep (period 6/7/8, jump@4, shine2@5/6) | grounded_fraction 0.11–0.29, max grounded chain 1 |
| JC-shine-frame sweep 0..4 | identical: grounded shine → full jump → land → shine |

Across reactive AND open-loop, across every B timing/edge/buffer: Fox always
completes takeoff and the shine comes out in the AIR, never during jumpsquat.
The shine input clearly reaches the game (it shines from standing, and shines
aerially after the jump) — but the shine-cancel of Fox's 3-frame jumpsquat is
never delivered on frame 1. The observe→react loop lands the shine one frame
late even at 0 latency, and open-loop timing doesn't fix it either.

**Conclusion: a clean grounded multishine cannot be produced through the
libmelee pipe input with the strategies available.** This is a hard wall, not
a tuning problem. It is almost certainly why the original fixture is 75%
aerial — the recorder physically cannot keep Fox grounded.

## What the POLICY actually does (2026-07-24, live capture)

Bradley noticed the trained bot appears to grounded-multishine "moreso than
the teacher." Captured the policy's per-frame action + emitted input live
(`scripts/analyze_policy_shine.exs`, same bridge as inference). Its longest
grounded run:

```
gShine af=2..18  gnd=true | B=false  X=TRUE(held)  stickY=0.0
```

The bot enters a grounded reflector and **holds it ~18 frames** with X HELD
(no fresh jump edge → no jump-cancel), then transitions (action 39) and
enters a NEW grounded shine. So the "grounded multishine" is the bot
**re-entering the shine repeatedly on the ground** — shine, hold, release,
shine — NOT the tight jump-cancel loop. `ShineChain`: grounded_fraction 0.16,
max grounded CHAIN = 1. It shines on the ground more than the teacher, but it
does not multishine.

**No actor — scripted probe, scripted teacher, or trained policy — produces a
jump-cancel grounded multishine through this bridge.** Every one either takes
off (probe/teacher) or holds/re-enters the reflector (policy).

## The deeper implication: the bridge is the wall at BOTH ends

The policy runs inference through the SAME MeleePort → libmelee pipe as the
probes. The probes proved the shine-cancel of jumpsquat doesn't register
through that pipe. Therefore a grounded multishine — which REQUIRES that
cancel — cannot be executed through this bridge by ANYTHING, scripted or
learned. This means:

- The fixture can't be GENERATED through the bridge (why it's 75% aerial).
- Even with a perfect human fixture, the bot running INFERENCE through the
  same bridge likely can't EXECUTE the multishine either.

So Track B is not just a data problem — it may be a **bridge-execution**
problem. Fixing it requires understanding why libmelee's pipe input cannot
deliver a jumpsquat-frame shine-cancel (input-poll alignment / a Dolphin or
ExiAI-build quirk), OR moving to an input path that can. That is the real
blocker, above the fixture question. A human-recorded fixture is still needed
for the DATA, but it will only pay off once the execution path works.

## Real options to get a clean fixture (next work)

1. **Human-recorded fixture (recommended — the only reliably-working path).**
   Record one real Fox multishine on normal Dolphin/console with a physical
   controller (NOT through the bridge — the 2026-07-24 investigation proved
   the bridge physically cannot keep Fox grounded), import it, verify with
   `ShineChain.summary` that `grounded_fraction ≈ 1.0` and `max_length` is
   large, then it's the fixture. Everything downstream (rebuild
   `MultishineExpert` table → retrain → gate on ShineChain) is ready.
2. ~~Bridge latency compensation~~ — **RULED OUT.** Measured latency is 0.
3. **Fix pipe input frame-1 delivery** (deep, uncertain) — figure out why the
   ExiAI pipe input can't deliver a jumpsquat-frame-1 shine-cancel (input-poll
   alignment? sub-frame? a Dolphin/libmelee quirk). This is the only way to
   generate a clean fixture through automation rather than a human recording,
   but it is open-ended infrastructure spelunking. `latency_probe.exs` and
   `jumpsquat_probe.exs` are the tools to continue it.

## GOALS.md Track B correction

GOALS.md says Track B is an EXECUTION problem ("the policy can't reproduce a
known sequence"). That framing is now wrong: the sequence it's imitating is
itself not the target. Track B's first gate is really **produce a clean
grounded-multishine fixture** (grounded_fraction ≈ 1.0); only after that does
"can the policy reproduce it" become the question.

## RESOLVED 2026-07-24 (late): the teacher multishines — the bridge was never the wall

The "bridge-execution wall" above is RETRACTED. The scripted teacher now
produces a TAS-quality 9-frame multishine cycle through the same pipe:
**max chain 20 in a 5s clip** (ShineChain v3, breaks only at clip edges),
grounded_fraction 0.467, 0 empty hops. Trace: `365 x3 -> 366 x1 -> 361 x2 ->
24 x3`, airborne 4 frames at y=0.0.

Three stacked errors had produced the "wall":

1. **The probes tested a nonexistent mechanic.** Down-B cannot cancel
   jumpsquat (only up-smash/up-B/grab can). Every "shine during jumpsquat"
   sweep failed against Melee's rules, not the bridge.
2. **The metric punished the technique.** A real multishine is ~half aerial
   (SmashWiki: "shining again the frame they leave the ground"); ShineChain
   v2 broke chains on ANY aerial reflector, so correct-shaped attempts
   scored max 1. v3 chains through air gaps <=8 frames containing an aerial
   shine.
3. **The one-frame timing that actually matters** (measured via
   `MULTISHINE_TRACE=1` per-frame y/vy): on airborne frame 1 the jump's
   +2.1 rise and the shine's vy-zeroing race — shine activating THAT frame
   stalls Fox at y~0 (lands in 4 frames); one frame later he is at y=2.1,
   and the aerial reflector's real physics (weak 0.027/f^2 gravity,
   fastfall DISABLED — measured) turn it into a 22-frame float. A reactive
   press off the first airborne observation is inherently that one frame
   late; pressing B on the observed LAST jumpsquat frame (af 3 of exactly
   3) arrives on airborne frame 1. Also required: hold B through the
   reflector so it survives landing JC-able (releasing mid-air causes an
   18-frame wind-down), release B only during jumpsquat af 1-2 to arm the
   fresh edge.

Teacher fix lives in `record_multishine.exs` `:closed_loop`. Next
(task list): re-record the fixture 20s+, verify with ShineChain v3
(max_length >= 5), rebuild MultishineExpert, retrain, gate the policy on v3.

## 2026-07-24 (overnight): teacher DONE, policy blocked on a state-stream shift

Ran the full backlog (fixture -> table -> retrain -> policy gate). Steps 1-3
passed; **gate 4 (the policy) FAILS, and the cause is not what any of the
earlier hypotheses predicted.**

### What passed

| Step | Result |
|------|--------|
| Fixture (30s, closed_loop) | max chain **186**, 187 shines, 0 empty hops, grounded_fraction 0.352 |
| Expert table | 9 keys, covers the whole cycle |
| Table-driven teacher LIVE (`demo_expert`) | max chain **103**, 0 empty hops ✅ |
| BC training | loss **0.00148 in 12 epochs** |

### The state-purity bug (cost one training run, then fixed)

The first BC run floored at **loss 0.121 after 2000 epochs** instead of
reaching 2e-3. Cause: the recorder's fastfall stick-flick was keyed on
WALL-CLOCK frame parity, and the cycle is 9 frames (odd) — so identical
states got different stick values on alternate cycles. `closed_loop`'s
premise is that every input is a pure function of observable state, and the
flick broke it; BC was fitting a coin flip.

The flick was also vestigial: it existed to fastfall Fox down from y=2.1,
but with the shine landing on airborne frame 1 he never rises. Removed it →
0 ambiguous state keys → **loss 0.00148 in 12 epochs** (167x fewer epochs).
`inspect_multishine_table.exs` now reports state purity so this is caught
before a training run is spent.

### Gate 4: the policy learned it, then can't run it

The trained policy, measured two ways with the SAME weights:

| fed | B rate | X rate | behavior |
|-----|--------|--------|----------|
| PARSED fixture states (offline) | 78.2% | 11.0% | **99.3% agreement** — reproduces the fixture |
| LIVE bridge states | 100% | 0.1% | holds one reflector ~2000 frames, never shines |

So this is **not** a learning failure, not exposure bias, and not the
fixture. It is the parsed-vs-live state shift of GOTCHAS #81 (Peppi and
libmelee disagree on `action_frame`: parsed jumpsquat af 0,1,2 vs live
1,2,3, and it varies per action). The policy learned "release B at jumpsquat
af 1,2" in PARSED coordinates and never sees those coordinates live.

DAgger was tried first and made it worse, which is itself informative:
- Round 1 rollout was truncated (`analyze_policy_shine` never ended the
  game; Slippi only finalizes on game end) → drill silently trained on the
  fixture alone: "0 frames, 0.0% corrected". Fixed — the script now SDs out.
- Round 1 with a valid rollout: 2609 frames, 94.3% corrected, but ~2100 of
  them were the SAME stuck state. It swamped the fixture and the policy
  collapsed to "always tap X" — from "never jump-cancels" to "never shines".
- Added `--rollout-cap-per-state` (opt-in) to `dagger_drill.exs`: caps
  frames per `{action, af, on_ground}`, keeping full state coverage. 2609 →
  514 frames. Policy still failed — because DAgger cannot fix a feature
  mismatch, only a coverage gap.

### Next experiment (do this first, it is cheap and decisive)

Diff the two state streams directly on the SAME game: the recorder's live
observations (`MULTISHINE_TRACE=1`) against the parse of the .slp it
produced, frame by frame, field by field. That yields the exact mapping
(is it a uniform +1 on af? per-action? do other fields shift too?). Only
then choose the fix:

1. **Normalize at the embedding boundary** — one conversion, benefits every
   policy, but touches shared code and needs a regression pass on the
   Mewtwo tracks (coarse behaviors currently tolerate the shift).
2. **Train on live-captured pairs** — have the recorder dump its own
   `(observed_state, decided_input)` stream and train on that, bypassing the
   replay round-trip. Narrow and safe; only helps drills.
3. **Feed the policy parsed-convention features live** — inverse of (1).

Do NOT spend more DAgger rounds on this until the streams are reconciled.
