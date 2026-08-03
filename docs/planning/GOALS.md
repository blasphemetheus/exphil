# ExPhil: Goals

**Last Updated:** 2026-08-03

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

**Corrected twice. Current status 2026-07-24: B0 and the teacher gate are
DONE; this is now an ordinary imitation problem.**

- *2026-07-23:* the fixture was measured (`ExPhil.Eval.ShineChain`) and found
  to be a sloppy shine → full jump → air-shine loop, not a multishine. Real
  finding: the behavior was not in the supervision.
- *2026-07-24:* the follow-up conclusion that the *bridge* could not execute a
  multishine was **WRONG** and is retracted. It rested on probes for a
  technique Melee does not have (down-B cannot cancel jumpsquat — only
  up-smash/up-B/grab can) and on a metric that broke the chain on the aerial
  shine a real multishine requires. Both fixed. The scripted teacher now
  multishines through the normal bridge (max chain 186 in a 30s fixture), and
  the **table-driven** teacher reproduces it live (max chain 103).

Full writeup: `docs/planning/MULTISHINE_DIAGNOSIS_2026-07-23.md` (see the
RESOLVED section at the bottom).

Gates:

| Gate | Metric | Now | Target | Tooling |
|------|--------|-----|--------|---------|
| **B0 — fixture is clean** | `ShineChain.max_length` of the fixture | **186** ✅ | ≥ 50 | `ShineChain.summary_for_replay` |
| **B0b — table teacher drives it** | `ShineChain.max_length`, teacher live | **103** ✅ | ≥ 5 | `scripts/demo_expert.exs`, `scripts/inspect_multishine_table.exs` |
| **B1 — policy enters** | chains started per minute, live | **✅ crushed** | ≥ 1 | `scripts/analyze_policy_shine.exs` |
| **B2 — policy sustains** | `ShineChain.sustained` (chains ≥ 5) | **✅ c434** | > 0 | `ShineChain` |

**Track B RESOLVED (2026-08-03).** Production policy
`checkpoints/ms_g6_sp1.bin` (multi-delay {2,3} + SS-on-queue recipe):
d2 **434.5/min chain-434**, d3 **413.4 c409** (3/3 deterministic),
d4 **332.4 c313** via `--delay-id-override 3`. Verified multishining
over Slippi Direct loopback at sharp 5-frame latency (2026-08-01,
then-champion mdq_ss). The full campaign record — SS-on-queue, jitter
refuted, spacing refuted, ladder stops at d4, CycleSim offline
simulator — lives in `LATENCY_ARCHITECTURE.md`.

**Do not gate on `grounded_fraction`.** A real multishine is roughly *one
third* grounded shine frames (measured steady state: 2 grounded + 4 aerial
frames per 9-frame cycle → 0.33–0.47). The old "≥ 0.9" target was unreachable
by construction and is what made correct attempts score as failures. It
remains useful only as a diagnostic: ~1.0 means lone ground shines that never
cycle, and a low value *with long air gaps* is the sloppy full-jump loop.

Reference: `ExPhil.Agents.MultishineExpert`, `scripts/demo_expert.exs` (drive
the teacher live), `scripts/demo_vs_teacher.exs` (policy vs teacher side by
side), `scripts/inspect_multishine_table.exs` (table coverage without booting
Dolphin).

---

## Method: the escalation ladder

Attempt in this order. Do not skip rungs.

1. **Fix what the expert teaches** — supervise the *decision* (when to commit),
   not just the button sequence. ⚠️ **UNSTARTED — this is the next work.**
2. **More / better imitation data** — opener weighting, curated go-in corpora,
   style conditioning, a probe-reg that actually applies.
3. **Invest in RL properly** — only after 1 and 2 are genuinely exhausted.

### We are here (updated 2026-08-03)

- **Track B (Fox multishine) is resolved** — see the gate table above.
  The July-era "we are here" (r15-r17a rung-2 attempts) is history;
  what actually broke Track B open was the DELAY CAMPAIGN (queue-as-
  input + SS-on-queue + the {2,3} pool) plus the interp instruments
  (early-reject probes, margin export, CycleSim), not more data.
- **Track A (Mewtwo decisions) remains at the rung-1 boundary**: P4's
  verdict (2026-08-02: NO drill policy reads techs — conversion is
  coverage, not reaction; 584 episodes, 4 checkpoints) sharpens rung 1
  into a concrete spec: the expert must supervise reaction-conditioned
  decisions AND the tech-episode frames must be upweighted (the
  P4 offset curve is the acceptance test).
- **P5 curation loop validated both directions** (2026-08-03 cycles:
  whole-rollout mixing destroys core skill; snippet mixing via
  `--snippet-frames` is safe). The fight-state/pressure gap is open
  pending the HUMAN replay corpus (always `--replay-dir`, GOTCHA #84).

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

## Next work (updated 2026-08-03; see newest HANDOFF for the live list)

1. **Rung 1: teach the decision** (Track A) — now with the P4 offset
   curve as the mechanistic acceptance test and tech-episode
   upweighting as the named lever.
2. **Fight-state/pressure gap** — human corpus collection
   (`--replay-dir`!) + richer pressure labels; the P5 loop and
   snippet-miner are ready.
3. **Direct exhibition** — recipe is ready (ms_g6_sp1 @ d3); gated on
   scheduling (ACAB#182 owed a match) + the stock-taking goal below.
4. **Rung-composition theory + train-unroll-=-deploy-step** — the two
   open engineering threads (LATENCY_ARCHITECTURE).
5. **Technique-id conditioning** — "frame-perfect on command": two
   techniques, one policy, one trigger channel; all machinery exists.

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
