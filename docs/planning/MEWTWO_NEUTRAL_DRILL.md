# Mewtwo neutral before combo routes

## Latest handoff and sequencing update — 2026-09-15

See [MEWTWO_NEUTRAL_TO_COMBO_HANDOFF.md](MEWTWO_NEUTRAL_TO_COMBO_HANDOFF.md)
for the resumable plan, current failed student results, and the user's next goal:
fair into a percent/DI-dependent follow-up. A controlled conversion drill can now
be developed alongside neutral corrections; this supersedes the older ordering
below that deferred all combo work until neutral passed. Integrated promotion
still requires both neutral and conversion to be validated.

## Requested behavior

Build a learned Mewtwo that uses down tilt, wavedashes, short hops, aerials,
and run-up grabs to create openings in neutral. Other options are acceptable
when they help create an opening. Combo-route training follows this stage.
Do not treat performing the move list as evidence of winning neutral.

Use sampled action selection at temperature 1.0 for the primary play and
qualification mode. Deterministic selection is a separately reported diagnostic.
Keep ordinary local play timing; choose the training reaction delay only after
checking the selected model and runner can sustain the corresponding live timing.

User-confirmed first matchup: Mewtwo versus Fox on Final Destination.
Use this matchup for teacher validation and the first learned neutral drill.
Expand opponents and stages after the first drill works.
The five-character/six-stage Fox multishine coverage is not automatically the
required first training scope for this different task.

## Outcome definition

An exchange starts after a clean neutral interval with both players alive and
out of hitstun, grabs, knockdown, and respawn. Shield is a valid neutral option:
it must not exclude an opponent from the evaluation, especially for run-up grabs.

Each exchange has exactly one outcome: Mewtwo opens, opponent opens, trade,
timeout, or censored by a gap/end/stock transition. Openings require a real
defender hit or successful capture, not attack startup or shield contact.
Report shield contact separately. After an opening, end the neutral trial;
do not credit subsequent combo hits as new neutral wins. Require a new neutral
interval before starting another exchange.

Report:

- Opening wins/losses/trades, with counts and rates for each opponent behavior.
- Time to opening; timed-out and censored trials remain in the denominator report.
- Successful openings by down tilt, aerial, grab, and other options; keep uncertain
  attribution explicit rather than crediting whichever attack was last observed.
- Unpunished versus punished whiffs, unwanted shield holds, and stock losses.
- Movement execution: actual short hops and wavedashes, facing and spacing changes.
  Airdodge frames alone do not establish a wavedash.

Movement is not rewarded just for its frequency. Do not impose an equal move
distribution across situations: the opponent's position and behavior should
change the choice. Verify that the policy can use each requested option in
appropriate held-out situations rather than collapsing to one repeated attack.

## Existing components and audit findings

- `MewtwoFairExpert`: fixture-backed aerial/approach behavior.
- `MewtwoPunishExpert`: rules for down-tilt and dash-grab whiff punishes.
- `MewtwoComboExpert`: includes tech chasing; do not use its whole cascade as
  the objective for this neutral-only stage.
- Existing Mewtwo fixtures cover approach, facing reversal, ground neutral,
  down/up tilt, aerial control, and out-of-shield behavior. Audit ports and
  coverage with the corrected replay parser before assigning training splits.
- `scripts/neutral_exchange.exs` is not suitable unchanged: its forward scans
  may overlap, successful grabs are not an explicit opening event, and shields
  are excluded from the neutral lead-in.
- `NeutralScan` measures attempted opener categories, not successful openings.
  Keep it as a diversity diagnostic, not the primary success metric.

## Build and validation order

**User requirement: validate the teacher on every requested behavior before
training from it.** Existing fixtures and green unit tests are not approval.

Teacher approval has two independent parts:

| Behavior | Execution evidence | Decision evidence |
|---|---|---|
| Down tilt | Actual down-tilt state and contact, not down smash | Both facings, opponent behind/in front, range and height, safe/unsafe punish windows |
| Wavedash | Jump → airdodge → special landing with signed travel in both directions | Useful spacing, no automatic approach into an attack or off an edge |
| Short hop | Measured apex below full-hop control, repeatable takeoff/landing | Appropriate approach/retreat and response when interrupted |
| Aerials | Intended aerial, drift, landing and L-cancel where applicable | Spacing/facing-sensitive choice and hit/shield/whiff outcomes |
| Run-up grab | Approach, grab attempt, opponent capture | Shielding opponent included; out-of-range, airborne, behind, and unsafe approaches tested |
| Combined neutral | Coherent transitions among approved primitives | Non-overlapping opening outcomes against shield, movement, and attacks |

Keep negative cases: the teacher must abstain or reposition when a move is
inappropriate. Approval is per behavior and tested condition; do not approve
the whole teacher because its short-hop routine passes. Preserve failed cases
and fix their cause before generating training labels. No training is authorized
by passing the old implementation-only unit tests.

1. Implement and test the exclusive exchange scorer, including shield/grab,
   whiffs, trades, gaps, stock changes, and end-of-recording cases. Audit examples
   manually against their replay states before using the scores for training.
2. Inventory the existing demonstrations. Preserve whole-recording train/held-out
   splits and source hashes. Check labels for facing, distance, and opponent state;
   the previous direction-blind teacher is a specific regression to avoid.
3. Validate the requested techniques against a passive opponent, then collect
   neutral trials against separate shielding, moving, and attacking opponents.
   Passive-dummy success is an execution check only. Audit demonstrations and
   scripted teachers before treating either as correct labels.
4. Train a small sampled baseline on causal inputs with the corrected parser.
   Use coherent demonstrations; do not randomly switch scripted move labels on
   individual frames. Compare against the existing fair/punish baseline on the
   same held-out neutral scenarios. Declare trial counts and promotion criteria
   before training, after measuring whether the scorer supplies enough events.
5. Collect the learned policy's mistakes, label targeted corrections where a
   defensible teacher exists, and preserve untouched evaluation scenarios.
   Retrain only in a separately named round with an explicit data split.
6. Validate on the actual graphical human-play runner, including stick/trigger
   delivery, input latency, real-time throughput, and rematches. Then expand
   matchup/stage coverage and record a human demonstration.
7. Start combo-route training only once the neutral policy can create openings.
   Reuse opening states as combo-start scenarios, retaining their opponent
   position, percent, defensive input, and checkpoint/source provenance.

## Carry-over lessons

The multishine project's parser, controller, and timing fixes are prerequisites;
its deterministic decoding and specialist-only training mix are not defaults
for this neutral policy. A smaller training loss or a longer attack sequence
does not substitute for successful openings against an active opponent.

## New human demonstrations (2026-09-15)

Preserved five new Mewtwo-versus-Fox games in
`eval_runs/0915_mewtwo_demonstrations/replays/`, with original paths, hashes,
ports, stages, and action-onset counts in `inventory.json`. Mewtwo is port 2
in every game. Stages: FD, Yoshi's Story, FoD, Dream Land, Battlefield.
Total: approximately 16m20s, 93 down-tilt entries, 38 grab attempts, and 305
aerial entries. These counts establish move coverage, not successful openings.
Airdodge and special-landing entries are retained as raw diagnostics; they are
not labeled as wavedashes without sequence/trajectory verification.

There is only one new FD game, so never split its frames across train and
evaluation. Rounds 1 and 2 excluded all human recordings. Round 3 assigns the
whole FD game to training, admitting nineteen audited successful-opening clips
and excluding combo continuations. The other four whole games remain held out;
the frozen assignment is `eval_runs/0915_mewtwo_neutral/v3/human_split.json`.
The user confirmed Fox was
CPU-controlled at level 6 and mostly level 9; per-game levels were not identified.
These are human Mewtwo demonstrations
against a CPU, not evidence of neutral success against a human opponent.

## Implementation and current experiments

The exclusive exchange scorer, stateful neutral teacher, live teacher battery,
physical-input exporters and sampled qualification runner are implemented in
Elixir. See `eval_runs/0915_mewtwo_teacher_audit/README.md` for approval evidence
and retained failures. Training protocols and round-specific changes are in
`MEWTWO_NEUTRAL_BASELINE_V1.md`; each checkpoint and its exact arguments live
under `eval_runs/0915_mewtwo_neutral/vN/`.
