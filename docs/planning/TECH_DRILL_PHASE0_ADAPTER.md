# Phase-0 adapter spec — Melee.Tech routines as drill experts

Written 2026-08-19 (planning-queue work during the g17 run). This is
the DECISION doc the curriculum's Phase 0 called for
(`TECH_DRILL_CURRICULUM.md`). No GPU, no implementation yet; the
curriculum arms wait on the retune verdict regardless.

## The impedance mismatch (why this needs a spec at all)

Two contracts that look similar and are not:

- **exphil drill expert** (what `dagger_drill --expert` consumes):
  a STATELESS per-frame labeler over recorded replay frames.
  `from_frames(fixture_frames, player_port:)` builds a modal table
  keyed `{action, action_frame, on_ground}`; `label(expert, player,
  prev, opp) -> {:ok, controller} | :skip` must answer "what would the
  expert have pressed on THIS frame" from player state alone, plus
  hand-written recovery rules for off-table states. Labels follow the
  replay convention (inputs land on the frame they land) so they
  compose with `Data.shift_actions/2`.
- **Melee.Tech routine** (libmelee_ex): a STATEFUL frame-stepped
  machine. `new(routine, char, opts)` then `step(tech, player_state)
  -> {status, tech, commands}` (pure core). It emits the input for the
  CURRENT frame given its own internal phase — it assumes it has been
  driving since the routine started.

Relabeling a policy rollout frame requires answering the counterfactual
"what would the routine press here" WITHOUT having driven the preceding
frames — the routine's internal phase must come from somewhere.

## Decision 1 — adapter architecture: FIXTURE-FIRST, not step-wrapping

**Chosen: Route F (fixture-first).** The Tech routine's job in the
pipeline is to GENERATE fixtures, not to label frames directly:

1. **Fixture generation (libmelee_ex side).** A scenario farm plays the
   routine live via `Match.play(rules: ...)` (1-stock / short-timer
   headless now supported) and records .slp. The routine is already a
   live input machine — this uses it exactly as built, zero adaptation.
   Variance knobs (position, percent, opts sweeps like `press_frame:`)
   live here, in the farm script.
2. **Expert construction (exphil side).** The recorded .slp feeds the
   EXISTING `from_frames` table machinery (the multishine pattern:
   modal controller per `{action, action_frame, on_ground}`). Recovery
   rules remain hand-written per drill — that is per-drill work and the
   main quality lever, same as multishine.
3. **Relabeling** then runs entirely on the exphil side with the
   existing table+rules contract. No cross-repo call in the training
   path; libmelee_ex stays a data producer.

**Rejected: Route S (step-rehydration)** — wrap `Tech.step`'s pure core
in `label/4` by inferring the routine's phase from player state each
frame. Rejected because (a) phase inference is per-routine bespoke work
with silent-wrongness failure modes (an off-by-one phase emits a
frame-perfect WRONG input — worse than no label); (b) the table already
IS the phase inference for any routine whose state is recoverable from
`{action, af, grounded}` — which is exactly the dense-loop routines
worth frame-relabeling; (c) routines where that key is NOT sufficient
(DI depends on launch trajectory, V-cancel on incoming-hit timing) are
precisely the ones Decision 3 routes away from frame relabeling anyway.

Consequence: **the per-drill deliverable is (fixture farm script +
recovery rules + relabel-window definition)**, not a generic adapter
module. There is no universal `TechExpert` — a thin
`ExPhil.Agents.TableExpert` generalization of MultishineExpert
(parameterized fixture path + per-drill rules module) is the only new
shared code Phase 0 needs.

## Decision 2 — precondition ownership: Situations labels trigger, experts veto

**Chosen: exphil `ExPhil.Situations` (47 labels) owns the trigger.**
The relabel window opens where a Situations label (or GameEvents event)
says the drill applies — e.g. `:missed_tech`-adjacent labels for the
tech drill, aerial-landing windows for L-cancel, combo/hitstun labels
for DI. Rationale: the labels already exist, are validated (1.5M-event
knowledge model), and are the SAME vocabulary the flaw list and
snippet miners speak — one trigger language across mining, drilling,
and scoring (the round-2 lesson: specific failure->outcome links).

The expert keeps a per-frame VETO via the existing `:skip` return
(dead/respawn/off-table states) — triggers are necessary, not
sufficient. libmelee_ex-side preconditions are used only inside the
fixture farm (where the routine needs a live setup, e.g. launcher-
assisted tech scenarios from defense_test.exs patterns).

## Decision 3 — relabel granularity: three classes, decided by signal shape

Per-drill decision, but from a fixed menu (document the class in each
drill's run-script header):

- **Class A, frame relabel (the multishine pattern):** dense cyclic
  routines whose full state fits the table key — shffl, waveshine,
  wavedash, teleport edge-cancel, ledgedash. Full DAgger relabel over
  the window; fixture farm supplies the table.
- **Class B, event weighting (no relabel):** 1-2 frame events inside
  otherwise-fine windows — L-cancel, tech-in-place/roll, V-cancel.
  Relabeling whole windows would overwrite acceptable behavior to fix
  one frame. Instead reuse the transition-weight machinery + the AWBC
  loss-weights channel: GameEvents (L-cancel success/fail, conversions)
  mark the event frames; weights amplify them within otherwise-normal
  imitation data. This is also the road into the generalist-AWBC arm (GPU
  queue #5): the same events are its new signal channels.
- **Class C, scenario-gated (no offline label at all):** context the
  key can't carry — DI/SDI vs real combos, shield options vs pressure.
  No table is trustworthy here; these drill via scenario farms
  (1-stock rules) + scenario_suite gates, mixed as snippets from
  SUCCESSFUL scenario runs (scenario_dagger_mine pattern). Unmeasurable
  vs CPU-1 stands (round-2 lesson) — never gate these on stand numbers.

Phase-1 assignments: L-cancel = B; ground tech + getup options = B for
the press, C for the option-selection; DI/SDI/ASDI = C; waveland/edge-
cancel aerials = A (when that libmelee round ships).

## Decision 4 — label conventions the adapter must preserve (hard contracts)

- Replay convention: fixture .slp recorded in dolphin already satisfies
  "inputs land on the frame they land" — no translation layer needed
  (another point for Route F; Route S would have needed one).
- `action_delay` contract: fixtures/snippets minted by farms MUST stamp
  the MixFrames envelope's `action_delay` (GOTCHA #86 is FATAL now).
  Farm scripts take `--action-delay` and pass it through.
- Commands->ControllerState: only needed if Route S is ever revisited;
  Route F never converts (dolphin recorded the actual pad).
- Dose discipline: every drill mix enters via absolute shares, carries
  ALL validated mixes, prereg in the run-script header (unchanged).

## What Phase 0 ships (implementation checklist, post-retune)

1. `ExPhil.Agents.TableExpert` — MultishineExpert generalized over
   (fixture path, key fn, rules module). Small; MultishineExpert
   becomes its first instance or stays as-is (do not churn the
   champion path — new experts use TableExpert, multishine untouched).
2. libmelee_ex farm script template: routine + rules + variance sweep
   -> .slp dir + MixFrames envelope with action_delay stamped.
3. GameEvents->loss-weights channel adapter for Class B (feeds both
   the L-cancel drill and the generalist-AWBC arm — write once).
4. First end-to-end proof: L-cancel (Class B, densest signal, success
   metric needs no human) — but ONLY after the retune verdict.

Open (deferred, not blocking): whether Class-B event weights share
AWBC's beta normalization or get their own scale (decide empirically
in the L-cancel prereg); Mewtwo fixture farms needing CSS handling for
non-Fox characters in headless Match.play (verify when Phase 3 nears).
