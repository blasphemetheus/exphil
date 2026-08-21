# Tech-Drill Curriculum — turning the Melee.Tech catalog into training signal

Written 2026-08-19. Context: libmelee_ex now has **41 live-verified
tech routines** across 6 dolphin suites (`~/git/libmelee_ex`, resume
point `docs/HANDOFF.md`; full catalog + implementation notes in its
`docs/melee-tech.md`). Every champion so far was built on ONE scripted
expert (multishine). This doc plans how the catalog becomes a drill
curriculum, in what order, and how each drill is validated.

## Why this is the highest-leverage use of the catalog

1. **The crown flaw list has direct counters now.** convC2's live flaws
   (0812 decider): combo DI, getup-attack overuse, side-B SD class,
   shield-grab loops, laser camping. The DI/SDI/ASDI routines and the
   defense/tech kits are exactly the scripted experts that can relabel
   those moments (the mechanism that killed the edge-SD loop: specific
   failure→outcome links, NOT blanket oversampling — round-2 lesson).
2. **AWBC is signal-starved.** It works (g16/g15r: +117% fox / +307%
   mewtwo on the fixed stack) but only shines carry signal today (121
   signal-free lists). GameEvents conversions + L-cancel events are new
   outcome channels for the generalist arm (Rewards.Standard).
3. **The Mewtwo kit unblocks the actual long-term goal.** Low-tier
   drills were blocked on having Mewtwo experts at all; teleport
   edge-cancel + the mewtwo suite are the first real curriculum there.

## Phase 0 — the adapter (GPU-free, prerequisite for everything)

`dagger_drill --expert` speaks exphil's scripted-expert interface
(`ExPhil.Agents.*Expert`, multishine + the mewtwo_* experts).
`Melee.Tech` routines are dolphin-verified INPUT ROUTINES on the
libmelee_ex side. Needed: a bridge that wraps a Tech routine as a drill
expert — (a) precondition detector (when does this routine apply),
(b) routine invocation via the bridge, (c) relabel window definition.

> **SETTLED 2026-08-19**: see `TECH_DRILL_PHASE0_ADAPTER.md` —
> fixture-first architecture (routines generate .slp via scenario
> farms; existing table+rules machinery labels), Situations labels own
> triggers (experts keep the `:skip` veto), granularity from a fixed
> 3-class menu (A frame-relabel / B event-weighting / C scenario-gated)
> assigned per drill. Original questions kept below for the record.

Spec questions to settle (planning work, no GPU):
- Which side owns the precondition? (Situations labels already exist —
  47 of them — reuse as triggers rather than re-detecting.)
- Rollout collection: `Match.play(rules: ...)` now supports 1-stock /
  short-timer headless games → cheaper scenario farms.
- Relabel granularity: multishine relabels frames; techs like
  L-cancel are 1-2 frame events inside longer windows — likely reuse
  the transition-weight machinery + AWBC loss-weight channel instead
  of pure relabeling. DECIDE PER DRILL, document per drill.

## Phase 1 — universal movement/execution (Fox lineage, fight-state)

Order chosen by (flaw relevance × signal density × routine maturity):

1. **L-cancel** — densest new outcome signal (every aerial), feeds
   AWBC channel directly (GameEvents L-cancel events already fold into
   stats). Success metric exists without a human.
2. **Live ground tech + tech-in-place/roll options** — counters
   getup-attack overuse (flaw list) + missed techs under pressure
   (fight-state gap). Use the launcher-assisted patterns from
   defense_test.exs for scenario farms.
3. **DI/SDI/ASDI (tier 4)** — counters combo DI (flaw list). Note:
   outcome is measured in opponent-combo context → needs the pressure
   scenario farm (1-stock rules help), not stand dummies.
4. **Waveland/edge-cancelled aerials** (next libmelee round, unbuilt)
   — movement generalization; defer until that round ships.

## Phase 2 — defense/escape (flaw-targeted)

5. **V-cancel** (next libmelee round) — panic-airdodge SD class
   (convC2 g1 flaw): the drill teaches the non-airdodge escape.
6. **Shield options** (shield-grab-loop counter) — needs human or
   scripted PRESSURE opponent; pair with Situations shield labels.
   Unmeasurable vs CPU-1 (round-2 lesson) — gate on scenario suite,
   not stand numbers.

## Phase 3 — Mewtwo kit (low-tier program)

7. **Teleport edge-cancel** — Mewtwo's core movement tech; experts
   exist (mewtwo_* agents + Tech mewtwo suite, TC margins already
   swept). Arm: mc_g2 = mc_g1_mdq_ss recipe + teleport drill mix.
8. **Mewtwo recovery/ledge kit** — pairs with per-stage-ledge (#25,
   implemented default-off) and the v3-edge corpus arm.

## Phase 4 — exotics (research, not curriculum)

SWD, yo-yo glitch, shine grab, instant RAR, Thunder Jacket: keep as
verification targets and future character kits; no drill priority.

## Per-drill validation ladder (unchanged discipline)

Every drill follows the validated pattern, one recipe change at a time:
rollout/scenario farm → mix built with dose control (**absolute shares;
every retrain carries ALL validated mixes**) → offline gates (incl. SD
gate) → async offense rung where applicable → deploy rung (blind,
human) before any crown claim (g6 rule). Prereg the gates BEFORE the
run, in the run script header, every time.

## Dependencies on the retune program

Do NOT start Phase 1 arms until the LR retune lands (HANDOFF 08-19 GPU
queue): drills change the mix, retune changes the optimizer — running
both unresolved reintroduces the g16 attribution problem.
