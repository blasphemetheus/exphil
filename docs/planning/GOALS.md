# ExPhil: Goals

**Last Updated:** 2026-07-24

This document states **what the bot should do**. It is deliberately not an
inventory of what we've built — that lives in the appendix, and only so we
don't rebuild it. Every number below traces to a real artifact.

> **Why this was rewritten.** The previous version (2026-02-28) tracked
> capabilities: 43 architectures, self-play infra, caching. It never stated a
> behavioral target. That let hygiene metrics — report-card 6.8/10, zero SDs,
> zero shield breaks — stand in for progress while the thing we actually
> wanted went unmeasured for months. The original goal was concrete: *spam
> short-hop fair as an approach, and follow up fair→fair to combo when it's
> an option.* That sentence appeared nowhere in the old doc.

---

## What we're building

A Mewtwo bot that goes in with short-hop fair and combos off it, and a Fox bot
that multishines perfectly. Low-tier characters are the long-term interest;
Mewtwo is the current subject because we have the drills and the corpus.

---

## Current state, measured

From the r16 fan-out (6 headless probe games, `logs/overnight_newera8_r16_20260722.log`)
and the coach report at `logs/coach/20260723_024737/report.md`:

| What | Measured | Reading |
|------|----------|---------|
| Armed approaches/min | **0.17** (per-game 0.13/0.25/0.38/0.0/0.25/0.0) | **It does not go in.** |
| Conversions | **6 / 119 approaches = 5%** | **It does not convert.** |
| Passivity windows | **125** across 6 games | Long stretches in range, doing nothing. |
| Dropped punishes | **42** | When it does open, it doesn't follow up. |
| Top-opener share | **91.4%** one category | **It does one thing.** |
| Opener entropy | **0.501 bits** (35 openers) | Almost no variety in neutral. |
| Report-card | 6.8/10 | Mechanically clean — *this is the misleading one.* |

**The honest read:** the bot is mechanically tidy (no SDs, no shield breaks,
sane jump discipline, real DI) and it *looks* like a Melee player in short
clips. It picks up some movement cues, some aerials, some out-of-shield
behavior. But it does not approach, does not convert, and does not chain
fairs. The hygiene gates are green and the behavioral goal is unmet.

### Diagnosis: it learned the sequence, not the decision

`mewtwo_combo` (fair expert + tech-chase expert) was built to teach
approach → fair → knockdown → punish. What came out throws the aerial, but not
*at* anything, and can't chain it. The drill supervised **which buttons**, not
**when to commit**. That single sentence explains gate-10, the 5% conversion
rate, and the 91% one-note opener distribution at once.

This reframes the problem: it is not "one missing metric." The training signal
never contained the decision.

---

## Track A — Mewtwo: the DECISION problem

**Goal:** approach with short-hop fair; chain fair→fair when the option exists.

All three gates, in order. A1 without A2 is whiffing into space; A2 without A3
is a poke, not the goal.

| Gate | Metric | Now | Target | Tooling |
|------|--------|-----|--------|---------|
| **A1 — goes in** | armed approaches/min | 0.17 | **≥ 1.0** | `ReplayStats.approach_stats/2` ✅ exists |
| **A2 — connects** | conversion rate (approach → hit) | 5% | **≥ 25%** | `ReplayStats.conversion_stats/2` ✅ exists |
| **A3 — chains** | mean connected aerials per opening | *unmeasured* | **≥ 2.0** | ⚠️ **needs building** |

**A3 is the literal statement of the goal and nothing currently measures it.**
`FailureScan.dropped_punish` (`lib/exphil/eval/failure_scan.ex`) only detects
the *failure* case (<15% damage gain). An aerial-chain-length metric belongs
next to it.

Supporting signal (not gates): opener entropy / top-opener share via
`ExPhil.Eval.NeutralScan` + the two `StyleCard` gates — useful for catching
"solved A1 by mashing one move."

---

## Track B — Fox: the EXECUTION problem

**Goal:** multishine perfectly, *no matter the circumstance.* It multishines
sometimes today; the target is reliability, not novelty.

**Corrected 2026-07-23 — this was NOT the execution problem it looked like.**
Measuring the fixture with a grounded-vs-aerial metric (`ExPhil.Eval.ShineChain`)
showed the training source `fox_multishine_closed.slp` is **only 24.6% grounded
shines** (275 grounded vs 844 aerial frames); its max clean *grounded*
multishine is **length 1**. The bot faithfully clones a fixture that itself
does the sloppy shine → jump → air-shine loop, not a real multishine. So the
first problem is a **data** problem: the behavior isn't in the supervision.
Full writeup: `docs/planning/MULTISHINE_DIAGNOSIS_2026-07-23.md`.

Gates, in order — B0 is the actual first step:

| Gate | Metric | Now | Target | Tooling |
|------|--------|-----|--------|---------|
| **B0 — fixture is clean** | `ShineChain.grounded_fraction` of the fixture | **0.25** | **≥ 0.9** | `ShineChain.summary` ✅; fix `record_multishine.exs` JC timing |
| **B1 — policy enters** | % of curated start states where a GROUNDED shine loop begins within 60f | — | ≥ 90% | `scripts/scenario_suite.exs` |
| **B2 — policy sustains** | max consecutive GROUNDED shines | — | ≥ 5 | `ShineChain` ✅ |

Root of the sloppy fixture: `record_multishine.exs` `:closed_loop` mode presses
the next shine on jumpsquat's *exit* frame (airborne), so the shine comes out
in the air. Fix the JC-shine frame → verify `grounded_fraction ≈ 1.0` BEFORE
saving → rebuild the `MultishineExpert` table → retrain. Only after B0 does
"can the policy reproduce it" (B1/B2) become the real question. Reference:
`ExPhil.Agents.MultishineExpert`, `scripts/demo_expert.exs` (drive the teacher
live), `scripts/demo_vs_teacher.exs` (policy vs teacher side by side).

---

## Method: the escalation ladder

Attempt in this order. Do not skip rungs.

1. **Fix what the expert teaches** — supervise the *decision* (when to commit),
   not just the button sequence. ⚠️ **UNSTARTED — this is the next work.**
2. **More / better imitation data** — opener weighting, curated go-in corpora,
   style conditioning, a probe-reg that actually applies.
3. **Invest in RL properly** — only after 1 and 2 are genuinely exhausted.

### We are here

- **Rung 2 has been attempted twice and failed to move A1:** r15 (raised
  conversions, gate-10 flat) and r16 (added human BC, gate-10 flat at 0.17).
- **r17a is in flight** (launched 2026-07-23 17:46) and is *also rung-2 work* —
  opener weighting + curated BC + style conditioning + a fixed probe-reg.
  Scored against `R17_ACCEPTANCE_2026-07-23.md`.
- **Rung 1 was skipped.** Whatever r17a scores, fixing what the expert teaches
  is the next work. If r17a misses, that's rung 2's *third* negative and the
  ladder says rung 1, not RL.

---

## Milestone: Direct exhibition

**Still a goal. Gated on beating someone** — the bot takes **≥ 1 stock off a
real human** in a Slippi Direct match. Not "looks presentable," not a date.

**Standing hard rule (unchanged):** Slippi **Direct only, with consent**. The
bot never queues ranked or unranked matchmaking — the ruleset forbids bots.
Bot account: EXPH#288.

---

## Explicitly NOT goals right now

These are real and will come back; they are *deferred* so they stop competing
for attention. Breadth before the core problem multiplies the bug.

- **Five-character program (#33)** — the RAM ceiling that blocked it is gone
  (streaming shards, `ExPhil.Data.TrainingShards`), but scale doesn't fix a
  training signal that lacks decisions.
- **Game & Watch (#23)** — corpus pulled and characterized (162 replays; mean
  armed/min **2.40**, opener entropy **0.934**, vs the Mewtwo archive's 3.9 /
  1.28). The data says G&W humans initiate *less*, so G&W inherits a **worse**
  version of Track A's problem. Deliberately after Mewtwo.
- **Architecture bake-off / broader backbone screening** — mamba_2 is fine; the
  bottleneck is not the architecture.
- **League / population play** — downstream of a bot that can play.

---

## Next work (derived from the gates above)

1. **Rung 1: teach the decision.** Change what the expert supervises so
   "when to go in" is itself a label. Top priority.
2. **A3 metric:** aerial-chain length per opening, next to
   `FailureScan.dropped_punish`.
3. **B1/B2:** a `multishine` scenario type + shine-chain metric; curate the
   start-state set.
4. Score r17a against `R17_ACCEPTANCE_2026-07-23.md` when it lands; record the
   result in "We are here" above.

---

## Appendix: what already exists

Listed so we don't rebuild it. Not a scorecard.

**Evaluation & diagnosis**
- `ExPhil.Interp.ReplayStats` — approach/conversion/opening stats (A1, A2)
- `ExPhil.Interp.StyleCard` — per-character gates incl. opener diversity
- `ExPhil.Eval.ScenarioScan` / `scripts/scenario_suite.exs` — situational probes
  via input-prefix virtual savestates
- `ExPhil.Eval.NeutralScan` — opener taxonomy + entropy
- `ExPhil.Eval.FailureScan`, `ExPhil.Eval.GapLedger`, `scripts/coach_report.exs`,
  `scripts/auto_bookmarks.exs` — the gap flywheel
- `ExPhil.Eval.Coverage`, `ExPhil.Data.SituationIndex`,
  `scripts/find_situations.exs` — occupancy diffing + situation retrieval

**Training**
- `scripts/dagger_drill.exs` — the drill (conversion + opener weighting,
  style conditioning, probe-reg, optional `--stream-chunk-size`)
- `ExPhil.Data.TrainingShards` — memory-flat streaming (unblocks large corpora)
- `scripts/curate_bc.exs` — corpus curation by initiation richness
- `scripts/human_drill.exs`, `scripts/selfplay_rollouts.exs`,
  `scripts/build_seed_dir.exs` — drill/rollout data generation
- 43 backbones via Edifice; ~2700 tests

**Corrections to the old version of this doc**
- Self-play / RL and "PPO integration" were marked ✅ **Complete**. They are
  **not**. PPO had never executed once; six surface bugs are now fixed and an
  architecture mismatch remains (MLP actor-critic vs temporal Mamba trunk).
  See `docs/planning/PPO_STATUS_2026-07-23.md`. This matters because PPO is
  rung 3 of the ladder above — it is not a quick pivot.
