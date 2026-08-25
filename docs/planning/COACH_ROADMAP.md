# The Coach — roadmap for a Melee teaching agent

Status: DESIGN (2026-08-10, from Bradley + brother-in-law's idea).
New long-term goal registered in GOALS.md. Nothing here is scheduled
yet; the point of this doc is (a) name the goal, (b) identify which
infrastructure we should build NOW because it serves both the current
bot program and this, (c) be honest about what's genuinely new.

## The idea

Boot a game vs a program that KNOWS what's good in Melee — per
character, matchup, percent, position, situation — and instead of
trying to win, it TEACHES: it steers the game into a chosen situation,
gives you a concrete goal ("you're on ledge at 80 vs Fox; get down
without getting hit — try ledgedash this rep"), plays the opposition
role for that situation, detects what you did, and gives feedback.
A curriculum engine decides what you drill next. Target user: a new
player getting good, not a lab.

## Is "what's good" already inside slippi-ai-style models?

Partially — three different capabilities hide in there, and none is
directly usable as a coach without extraction work:

1. **Policies (ours, slippi-ai's) encode what strong players DO** —
   conditional behavior p(action | state). That is "what's good" in the
   descriptive sense, at controller level, with no names and no why.
   Usable TODAY as: "in this situation, a strong Fox's action
   distribution is X" — but X is stick coordinates, not "shield-drop
   bair".
2. **Value/outcome models encode how good a state IS** — win-prob /
   reward-to-go. We have the F5 offline-RL direction decided but no
   trained value net yet. A calibrated value model turns "make a
   situation" into "make a situation the learner handles at 30% success"
   and turns feedback into numbers ("that option cost you ~8% expected
   stock").
3. **Corpus statistics encode both, cheaply and legibly** — condition
   on a situation label, aggregate what high-rated players chose and
   what happened next. No new NN. This is the v0 knowledge model and it
   is gated ONLY on the situation-labeling program (SITUATION_LABELS.md)
   plus an option-naming layer.

The honest gap: matchup coverage. High-tier matchups have corpus
density; low-tier matchups (the project's whole motivation) are sparse.
The knowledge model inherits the same data wall as the bot.

## The load-bearing insight for the lab TODAY

Every component the coach needs is a component the current program
already wants for its own reasons:

| Coach component | Current-program justification |
|---|---|
| Situation labeler (shared rules, live + batch) | curation/oversampling, DAgger routing, per-situation eval — the labeling program Bradley already prioritized |
| Option vocabulary (controller streams -> named techniques) | legible evals ("dropped punish", "missed lcancel"), drill scoring; partial versions exist (ShineChain, coverage_ledger, coach report) |
| Moment-inspection API (state -> everything inference exposes) | interp program's daily need, currently scattered across 20 probe scripts |
| Value model | F5 offline RL (already the decided direction) |
| Scenario director (steer game into a labeled situation) | drill seeding — build_seed_dir/scenario seeds already do a static version |
| Per-situation corpus statistics | fight-state gap analysis, curriculum weighting |

So "build the coach" mostly means "finish the labeling program and give
it three consumers we already wanted."

## Infrastructure to build now (ordered)

### 1. `ExPhil.Situations` — the labeler (shared-rules module)
The winnowed SITUATION_LABELS.md set as ONE module with two frontends:
- batch: parsed replay -> per-frame u64 bitmask (stored beside
  MmapCorpus labels; mmap-friendly, filterable)
- live: fold over `Melee.GameEvents`-style diffs -> segment start/end
  events
Same rule code for both so live and post-hoc labels cannot drift.
This is priority 1 with or without the coach.

### 2. Moment-inspection API (the "rewind" backend)
`ExPhil.Inspect.moment(replay_or_ring, frame, policy, opts)` -> one map:
- situation bitmask + active labels (from #1)
- embedding vector + which features are OOD vs corpus stats
- policy forward: per-head distributions (buttons, sticks), sampled vs
  argmax action, entropy per head
- trunk state + probe outputs (early-reject probes, AbsorberEntry,
  cycle margins — the existing instrument zoo, unified)
- value estimate + per-option Q deltas (once #5 exists)
- counterfactuals on demand (the probe_edge_attribution pattern: patch
  x/percent/opponent -> delta in heads) — expensive, opt-in
Almost all of this exists as scattered probe scripts; the work is ONE
callable API + JSON serialization. Serves interp daily work
immediately; serves any viewer forever.

### 3. The rewind viewer (v0 = Livebook)
A scrubber over a replay: timeline with situation-label tracks (like a
DAW), click a frame -> render Inspect.moment. Livebook first (zero new
stack, Kino has sliders/plots); a proper web UI only if it earns it.
Also the natural artifact for "why did the bot do that" questions we
answer ad hoc today.

### 4. Option vocabulary + per-situation statistics
- `ExPhil.Options`: recognize named options from frames (wavedash,
  dashdance, shield-drop, ledgedash, lcancel'd aerial X, getup variants
  ...). Rules-based; the drills/coverage_ledger/coach-report code
  already names many — consolidate.
- Stats job: over the HF corpus, for each (situation label, matchup,
  percent bucket): distribution of options chosen by winners vs losers,
  outcome deltas. Output = queryable table ("v0 matchup knowledge
  model"). This is the first thing that can answer "what's good here"
  in words.

### 5. Value model (F5, already planned)
Offline RL / outcome regression on the corpus. Coach uses it for
difficulty calibration and feedback scoring; lab uses it for RL.

### 6. Scenario director — built AROUND Improoover
**Improoover (ppfiction)** already solves the hard half: it exports a
moment from a .slp into a .gci savestate you can boot, replay, and
TAKE CONTROL of — built for "redo where you messed up in tournament."
So the director's core loop is: pick a moment (from the learner's own
replays via situation labels + knowledge-model deviation, or from the
corpus as a canonical example) -> Improoover-style export -> boot the
.gci -> learner plays the rep, coach bot plays the opposition ->
detect/score the response (labels + option vocabulary) -> repeat or
advance. Integration work: drive the export programmatically from our
labeler's output (a mined SD/failure window IS a coachable moment —
tonight's edge miner literally produces the input list), and get the
bot agent playing the opposition port inside a .gci-restored state
(MeleePort/libmelee vs the training-mode loader — the one open
plumbing question). Live steering (driving the game into a situation
without a savestate) becomes the fallback, not the foundation.

### 7. Curriculum + feedback loop (the actual coach product)
Pick next drill from (learner's per-situation deviation from the
knowledge model) x (spaced repetition); deliver goals/feedback (text
overlay? between-game summary?); measure improvement per situation
over sessions. This is the genuinely NEW part with no current-program
twin — build LAST, on top of 1-6.

## Under the hood reuse

Rep delivery = **Improoover exports** (.slp moment -> bootable .gci
savestate, see #6). Mining which moments to export = the existing
labeler/miner stack (tonight's edge miner is the template: situation
label -> window -> coachable moment). Scoring = the same
labels + option vocabulary that grade the bot. The learner-facing loop
generates exactly the artifacts the bot-training loop consumes: a
human drilling ledge escapes against the coach produces
perfectly-labeled situation snippets. THE COACH AND THE BOT FEED EACH
OTHER: his reps are our corpus; our knowledge model is his teacher.

## What this implies that's genuinely new

- A human-facing product surface (UI, sessions, progress tracking) —
  everything to date is lab-internal.
- Difficulty control as a first-class capability (the bot currently
  optimizes winning the situation, not losing it instructively at a
  target rate). Temperature/handicap curves per scenario.
- Explanation generation: mapping knowledge-model deltas to words a
  new player understands. Corpus stats give the numbers; the phrasing
  layer is new.
- Latency/ergonomics constraints of a consumer session (netplay-safe
  delays are already solved for the bot; menus/UX are not).

## The analysis product surface (added 2026-08-25, Bradley + brother-in-law round 2)

Three product concepts, in increasing ambition. They form a ladder:
each is the substrate of the next, and almost all of the hard parts
are components the lab program already shipped or already decided to
build. The chess analogy runs deep: Stockfish = policy + value +
legal-move enumeration + a UI that overlays them on the board. We have
the policy, we decided to build the value net (F5/D2), the "board
geometry" is done (StageCollision + FrameData + viewer), and the
legal-move enumerator is the one genuinely missing piece.

### C1. The options overlay ("chess-move highlights for Melee")

In the replay viewer, at a decision point, show the *available* named
options the way a chess UI highlights a queen's squares: Fox can JC
upsmash / full hop / side-B / walk forward / shield-drop (with timing
windows) — rendered as affordances at the character's position. Plus
usage analytics over the replay/session: option distribution per
situation, so "you side-B'd 78% of your ledge approaches" is visible
as a bar, and low option-entropy (predictability) jumps out.

What exists: `ExPhil.Options` recognizes options *chosen* (detection);
`ExPhil.Situations` provides the decision-point segmentation (47
labels); `situation_stats` gives per-situation option distributions
over 1.5M corpus events (the "book"); the HTML rewind viewer already
draws precise stage geometry and hitbox-gated rings; opener-entropy
machinery exists in NeutralScan/StyleCard.

The missing piece — the **option ENUMERATOR**: given (action state,
frame within it, position, character frame data, stage geometry),
enumerate what is *legal now* with timing windows. Rules-based, no ML:
frame data gives cancel windows and jumpsquat rules, StageCollision
gives platform/ledge affordances (shield-drop spots, ledgedash
availability). It's the inverse of the detector we already wrote, over
tables we already ingested. This is the C1 build item.

### C2. The engine lens ("what does the bot think here")

The rewind viewer's existing policy-distribution graph, upgraded to
speak the SAME vocabulary as C1: project the policy's controller-space
distribution into *option space*, so the overlay can color each
available option by (a) what the model would do and (b) what corpus
winners did in this situation. Two implementation routes: cheap —
sample N actions from the policy heads and classify the short
continuations with the existing Options detector; richer — short
CycleSim-style rollouts per candidate option. Multi-model comparison
(champion vs generalist vs corpus stats) falls out for free and is
itself interesting lab telemetry (where do our models disagree with
masters?).

Ranking is what makes this Stockfish-like, and ranking needs the
VALUE MODEL (item 5 / direction D2). Until it lands, the corpus-stats
winner distribution is the honest v0 eval bar; after it lands, each
option gets an expected-value delta ("this choice cost ~8% expected
stock" — the number this doc already promised in #5).

### C3. Natural-language replay feedback ("put your replay in, get coached")

Drop a .slp in; an intelligence layer tells you what went wrong in a
neutral exchange and what would have won it, where you dropped a combo
and the follow-up you had, how you could have DI'd out. The key
design read: **the LLM is the THIN layer; the grounded analysis is the
hard part — and the grounded analysis is exactly C1 + C2 + the value
model + the failure scanners we already run** (dropped_punish, SD
classifier, situation exchanges). Pipeline: replay → moment
segmentation (Situations) → per-moment structured facts (options
available [C1], option chosen [Options], what masters chose
[situation_stats], value deltas [F5], failure tags [FailureScan]) →
LLM narrates, prioritizes, and phrases for a human. Never let the LLM
watch raw frames; it only ever sees the structured facts — that keeps
it honest and makes its claims checkable.

One standalone gem inside C3: the **DI report card**. Melee knockback
is deterministic given (move, percent, position, DI input), so optimal
survival DI per hit is *analytically computable* — no ML, no value
net. Compare actual DI stick angles to optimal per death/combo-hit and
report the survivable deaths. Self-contained, high perceived value,
buildable early, and exercises the FrameData/physics tables end to end.

### Sequencing (the ladder)

- **v0 (buildable now):** option-entropy + usage report per situation
  (shipped parts only), rendered in the existing viewer. Eval bar =
  corpus-stats proxy.
- **v0.5:** the option enumerator → C1 overlay proper.
- **v1:** value model lands (D2/F5, the lab wants it anyway) → true
  eval bar + per-option deltas → C2 complete.
- **v1.5:** DI report card (independent track, any time).
- **v2:** C3 narration over the structured-facts pipeline.

Answer to "what direction do we go to make C3 eventually possible":
**the direction we're already going.** The generalist line gives the
policy prior; offline RL (D2) gives the value model; the labeling
program gives the segmentation and vocabulary. The only net-new
investments C3 adds are the option enumerator (C1) and the phrasing
layer — both small next to what's banked.

## Status (2026-08-11)

1. `ExPhil.Situations` — **SHIPPED** (47 labels winnowed with Bradley;
   fold-based batch+live frontends; found GOTCHA #96 external/internal
   stage-id collision in testing).
2. `ExPhil.Inspect` — **SHIPPED** (session + moment/2 +
   counterfactual/4; stub-injectable for tests; smoked on edgeB).
3. Rewind scrubber — **SHIPPED**, then superseded by the HTML rewind
   viewer (`priv/viewer/rewind_viewer.html` + `export_rewind.exs`,
   hitbox-gated rings, bookmarks).
4. Option vocabulary + per-situation corpus stats — **SHIPPED 08-10/11**
   (`ExPhil.Options`, situation_stats v2, 1.5M events).
5. Value model (F5) — unscheduled.
6. Scenario director / Improoover plumbing spike — unscheduled (do the
   spike in a live session).
7. Curriculum/feedback — last.

## Non-goals for now

No scheduling, no UI beyond Livebook, no product polish.
