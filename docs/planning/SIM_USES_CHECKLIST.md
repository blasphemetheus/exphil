# Sim uses — the checklist to run through

**Written 2026-09-21.** Every use of melee-sim-light we have named, as an
empty checklist to complete over time. Bradley's reactions on 09-21 are
recorded next to each; priority marks come from those. Infrastructure
prerequisites live in `SIM_INTEGRATION.md` (steps 1–3 done: mapper,
worker, row fidelity). Tick a box only with a dated line of evidence.

Legend: ★ = Bradley wants it · ◇ = neutral · ? = needs explaining first

## Priority (★)

- [ ] ★ **Search-as-teacher** — from any state, roll N candidate input
  sequences for K frames (save/restore), keep the scorer-accepted one,
  distill by BC/DAgger. "Very cool." → `SIM_INTEGRATION.md` step 7.
  Evidence line: oracle conversion rate vs the prior's on 1,000 starts.
- [ ] ★ **Self-play PPO on the imitation prior** — KL-to-prior, opponent
  pool, fingerprint bound. "Sounds like it'll be good." → `RL_ON_PRIOR.md`
  R2/R3. Evidence: win rate vs frozen prior over ≥ 200 games + fingerprint
  inside the human range.
- [ ] ★ **The coach** — per-situation "what the best players do here",
  "what the engine thinks", counterfactual branches from any frame.
  "Worth doing." → `COACH_STYLE_PRODUCTS.md` C1/C2. Evidence: C1 overlay
  on the rewind viewer; C2 after the critic exists.
- [ ] ★ **Full league across all 26 characters** — population play with a
  matchup table, 16 cores × batch 256. "Worth doing." Needs priors per
  character (only Fox now) or scripted/search agents for the rest; the
  first version can be search-agents-only (no training) to produce a
  matchup table of the *sim's* balance. Evidence: a 26×26 table with CIs.
- [ ] ★ **10,000 Foxes in one situation** — Monte Carlo of one state
  (sampled inputs or the prior at T=1) → heat map of where the bot dies /
  lands hits / gets hit. "Cool." Cheapest ★: needs only the loop + a plot.
  Evidence: one heat map PNG from a real replay moment.
- [~] ★ **Genetic algorithm for the longest combo from 0 %** — v0 BUILT 2026-09-22 (`ExPhil.Sim.GA`, `scripts/ga_combo.exs`, `priv/viewer/ga/`); first run `eval_runs/0922_ga/fd_idle_p512_g200` (see HANDOFF_2026-09-22 15:00) — — evolve input
  sequences against a target; fitness = scorer output. "Fucking dope."
  See the GA section below. Evidence: best-of-generation curve + the
  combo replayed in the viewer.
- [ ] ★ **Reward-hacking zoo** — run RL with deliberately bad rewards,
  catalogue exploits (ledge stall, camping, taunt loops), tune the
  fingerprint bound before the real run. "Very cool, we should do that."
  Evidence: a zoo table (reward → exploit → tell that catches it).
- [ ] ★ **Retry a real moment 100 times** — load a replay frame, let a
  human replay that moment against the bot from that state. "A really
  good goal to work towards." Needs the human-input path into the sim
  (adapter → sim controller row) and the viewer as the display. Evidence:
  Bradley retries one edgeguard from his own replay.

## Neutral (◇)

- [ ] ◇ **Regression-grade evaluation in CI** — fixed start distributions,
  confidence intervals, no Dolphin. The "most professional" one; it is
  also what makes every other box measurable. Evidence: a CI job.
- [ ] ◇ **Curriculum drills with combo scorers** — the drill = (start
  distribution, scorer, horizon). Prerequisite of search-as-teacher.
- [ ] ◇ **Deterministic frame-data queries** — "does this fair hit from
  here at this percent?" by stepping 30 frames. Free once the loop exists;
  useful inside the coach.
- [ ] ◇ **The Viking one** — longboat league (4 slots, teams, last raft
  standing) or the berserker reward (damage only, no survival term).
- [ ] ◇ **Practice partner with named profiles** — CPU that DIs and
  tech-chases like SKWA/C2; the registry profiles as training dummies.
- [ ] ◇ **Cross-character transfer via sim-generated, oracle-labeled
  games** — longest shot; after Mewtwo admission is trusted.

## Needs explaining (?)

- [ ] ? **"Is identity a prior or a costume?"** — the scientific one,
  restated: run the same RL drill for several registry profiles and
  measure which habits survive RL pressure. If the profile's habits (jump
  button, c-stick use) persist while performance improves, identity is a
  *prior* the policy builds on; if RL erases them, identity was a
  *costume* painted over one underlying policy. Either answer is a
  publishable claim about conditioning in imitation-then-RL pipelines,
  with a clean control (anonymous slot 0). Bradley: "would be cool if we
  could say something scientific" — this is the candidate.

## Genetic algorithms — what they could do here, and how to watch them

A GA needs a genome, a fitness function, and a deterministic evaluator.
The sim is the evaluator; save/restore makes every evaluation start from
the same state; fitness comes from the scorers we already have.

| Genome | Fitness | What it finds |
| --- | --- | --- |
| Input sequence, K frames (13 floats/frame, or a discretized 6-head token/frame) | combo length (AerialChain) / damage before the opponent escapes | the longest true combo from a start state — the headline |
| Same, against a *defending* opponent policy (prior at T=1) | damage dealt − damage taken over K | robust openers, not just optimal-vs-idle |
| Start-state parameters (positions, percents, action frames) | how badly the prior does from there | the prior's worst situations — a curriculum generator |
| Reward weights for RL | fingerprint distance + win rate | rewards that don't hack — the zoo's counterpart |
| DI / tech-option sequence for the *victim* | survival frames under a fixed attacker | the best escape from a known combo (coach content) |
| Controller noise/timing offsets | execution under human-like jitter | how much timing slop a combo tolerates |

Visualization: the sim is deterministic and cheap, so every candidate can
be rendered. (a) **Heat maps**: fitness over 2-D genome slices (start x ×
percent), or death/hit density over stage coordinates — the 10,000-Foxes
picture. (b) **Generation strips**: the top genome per generation replayed
side-by-side in the sim viewer (it draws hitboxes/shields) — the
"evolution montage". (c) **Family trees**: which mutations survived, as a
lineage graph with the combo-length curve. (d) Live: the viewer already
reads `*.msltrace.json`; a GA run writes one trace per elite. The YouTube
genre Bradley is thinking of is the "AI learns to X" evolution channel
style (population of agents shown at once, generation counter, best-so-far
highlighted) — reproducible here as a grid of sim viewers.

- [ ] GA v0: fixed start (Fox vs Fox at 0 % on FD, opponent idle), genome =
  90-frame discretized input sequence, fitness = AerialChain length +
  damage, population 512, tournament selection, mutation = per-frame
  token flip; 200 generations; output = curve + top-1 trace. Evidence: the
  trace in the viewer and the curve.

## The video (Bradley 2026-09-21)

- [ ] ★★ **Record our own "AI learns Melee" evolution video** at the end of
  the GA + RL work: population grid in the sim viewer, generation counter,
  best-so-far highlighted, the combo-length / win-rate curve alongside,
  narrated. Every ★ item above produces a segment (10,000 Foxes heat map,
  GA longest combo, reward-hacking zoo bloopers, self-play climbing the
  league). Evidence: a rendered cut. Prerequisite: the viewer-trace export
  from sim runs (`*.msltrace.json`) wired to a batch renderer.
