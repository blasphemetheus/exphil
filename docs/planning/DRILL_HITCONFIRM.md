# Hit-confirm drill design (2026-09-01 night, post-F4)

## Why this drill, and why now

F4 (`eval_runs/0901_combo_depth/RESULTS.md`) settled Bradley's "how do we
show it the deeper layers": the model's action-match on expert mid-combo
states is FLAT across combo depth (pass@16 30.3 → 27.4 from hit 1 to hit
4+) while its LIVE punishes cap at 1–2 hits. The deep layer is learned;
the model never ARRIVES at it. That is a state-visitation problem, and
state-visitation is what the drill system exists for (07-08:
"anything the user can record becomes a drill"; it found two
embedding-level bugs BC diagnostics never caught).

Live evidence the seed exists: Bradley saw up-throw → up-air tonight
(first observed punish structure, v1.3 live look).

## The core idea

Put the policy IN a hit-confirm state over and over — the state where
hit 1 just connected and the expert would continue — and let it close
the loop from there. Score continuation, not the opener. Then feed the
drill's own replays back through advantage-weighted retraining, so
successful continuations (which are now generated ON the bot's reachable
manifold, not the expert's) get upweighted. This marries both F4 levers:
drills fix arrival, AWBC on drill output fixes the class deficit.

## Delivery: two routes

### Route B — scripted state construction (PRIMARY: all plumbing exists)

The port-2 dummy supports `external` mode (Elixir drives its controller
via `send_controller` with `:port` — the self-play plumbing). We do not
restore expert states; we RECONSTRUCT their class:

1. Reset to a mined spec {opener, percent_band, position_band, facing}.
2. SETUP PHASE (scripted, both ports): drive the dummy to the position;
   accumulate its percent to the band with scripted throwaway hits
   (pummel/jab — percent is the only slow state variable; position and
   facing are fast).
3. OPENER PHASE (scripted expert on the BOT'S port): execute the opener
   (up-throw / dair / grab) — frame-precise, from the 07-08 expert-table
   machinery.
4. HANDOFF: the moment the opener CONNECTS (opponent enters hitstun),
   control of the bot port switches from scripted expert to the POLICY.
   This instant is the drill's t=0 — the exact state class F4 probed.
5. CONTINUATION WINDOW (240 f, the edge_scorecard horizon): policy plays;
   dummy plays a mix of DI regimes (see knobs).
6. Score, reset, repeat. Bank every replay.

Fidelity limits accepted up front: dummy DI is scripted, not human;
states are class-matched, not frame-identical to expert moments. Both
are fine for the visitation problem — we need the NEIGHBORHOOD, not the
exact state.

### Route A — improoover savestates (FIDELITY UPGRADE, plumbing open)

`.slp moment → bootable .gci` gives frame-exact expert mid-combo states
(including states 2–3 hits IN, which Route B reconstructs poorly).
Blocked on: bot agent (MeleePort/libmelee) inside a .gci-restored
training-mode state (COACH_ROADMAP #6 plumbing question). Do not gate
the drill on this; adopt it when it lands, reusing the same scoring.

## Mining the drill specs (offline, tonight-able)

`conversion_snippet_mine.exs` / the Situations `:conversion_open` +
hitstun-edge machinery (reused by F4) already find expert conversions.
Mining pass produces a DRILL TABLE:

- Group expert conversions by (opener action, victim percent decile,
  stage zone). Rank by (frequency × mean hits-after-opener).
- Keep the top cells; record for each: expert continuation length
  distribution, damage distribution, and the top continuation actions
  (for diagnosis, not for scripting the answer).
- **Drill 1 (pre-registered): up-throw on FD, victim 0–40%, mid-stage.**
  The canonical Fox low-percent conversion (up-throw → up-air chains),
  and the structure Bradley already saw live once.

## Scoring (pre-registered before the first run)

Per episode, from the handoff instant:
- hits landed in hitstun-linked chain (the F4 depth counter, reused)
- damage dealt within 240 f
- outcome class: extended (≥2 more hits) / one-and-done / dropped-
  actionable (opponent actionable inside our range with us whiffing) /
  reset-safe
Reference distribution: the SAME statistics computed over the mined
expert cell. Report bot-vs-expert side by side (standing rule: expert is
the denominator).

**Gate for "the drill works" (before any retraining claim): after N
drill episodes, hitstun-linked chain length distribution shifts toward
the expert's on FRESH drill episodes** — i.e., measure the retrained
policy in the drill itself, then confirm transfer in a normal live look
(does the up-throw → up-air become up-throw → up-air → up-air?).

## The training half (closing the loop)

Drill replays → standard banking → retrain options, in order of
preference:
1. **AWBC on drill replays** (`--awbc --awbc-reward standard`, now
   port-correct): damage return-to-go inside the continuation window
   upweights successful continuations automatically. No hand labels.
2. Mix drill replays into the main corpus at a weighted rate (the
   validated-mix rule: every retrain carries all validated mixes).
3. DAgger relabel only if 1–2 underperform: relabeling combo
   continuations needs a reactive expert (the corpus's matched-state
   continuations, not a fixed table) — costlier, design later.

## Knobs / variants (later, not first run)

- Dummy DI regimes: none / away / survival / random mix (the drill's
  difficulty ladder; expert data faced human DI).
- Opener set expansion: dair → shine, grab at higher percents, aerial
  hit-confirms.
- Percent curriculum: drill the same opener across deciles (continuation
  changes character with percent — that's the depth the corpus teaches).

## Open questions (answer during build, not before)

1. Handoff detection: opener-connect = victim hitstun rising edge on the
   dummy port — the F4 edge detector, live. Verify latency ≤ 1 f.
2. Percent accumulation speed: scripted pummels are slow — measure
   setup-phase wall time; if > ~10 s/episode, use throwaway projectiles
   (lasers) or accept coarser percent bands.
3. Does `dagger_drill.exs`'s fixture format fit a two-phase (setup →
   handoff) episode, or does the drill runner need a new episode driver?
   (Read DRILL_INFRASTRUCTURE.md protocol lessons first — 07-08 rule.)

## Build order

1. Mining pass → drill table + Drill 1 cell stats (offline; can run now).
2. Episode driver: setup/opener/handoff phases on the existing
   dummy-external plumbing; 20-episode smoke with score printout.
3. 500-episode bank + scorecard vs expert cell.
4. AWBC retrain arm + drill re-measure + live-look transfer check.
