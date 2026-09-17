# Mewtwo neutral → fair conversion: plan and resumable handoff

Updated: 2026-09-15. Intended for Claude or another coding agent resuming this
workspace without the conversation. This document records decisions, evidence,
implementation steps and acceptance criteria. **It does not claim the proposed
combo routes work.** No fair-conversion teacher or learned conversion drill has
been validated yet.

## 1. Goal and user decisions

Build a learned Mewtwo that creates an opening in neutral, converts suitable
openings into a combo, and returns to neutral when continuation is unavailable.
First matchup: **Mewtwo versus Fox on Final Destination**. Broader characters,
stages and long combo routes come later.

The user's current preferred starting point is **fair into a context-dependent
second hit**, rather than one fixed combo at every percent:

- A low/descending fair may require landing, L-cancelling and jumping again.
- A rising fair may permit an airborne continuation using a timed double jump,
  followed by fair or up air.
- Fox's percent and DI affect the available route. Fair being the most consistent
  opener is the user's hypothesis to investigate, not an established result.
- Down tilt → fair remains a later candidate with its own percent/DI window.
- The first conversion milestone is **two connected hits**, not a long combo.

Important constraints:

- Validate each teacher behavior before using it for training. Existing expert
  names, successful move animations and green unit tests are not certification.
- Primary student evaluation and play use **sampling at temperature 1.0**.
  Deterministic tests can diagnose failures but cannot replace that gate.
- Preserve ordinary local timing: reaction delay 0, measured input latency 1.
  Do not change emulator timing to make a route connect.
- Prefer Elixir over Python whenever possible. Current harnesses use Elixir.
- The learned player should actually use learned weights. Scripted teachers are
  data collection tools, not an undisclosed runtime override.

**Sequencing update:** a controlled fair-conversion drill may be developed now,
alongside unresolved neutral work. This supersedes the older plan's instruction
to defer all combo work until neutral passes. The integrated bot still needs
both capabilities to pass independently and together.

## 2. Current state: read this before starting another fit

### What works

The stateful `MewtwoNeutralTeacher` has approved controlled Fox/FD cases for down
tilt, approach wavedash, short-hop aerials, run-up grab, defensive repositioning,
turnaround and resuming after interruption. Approval is limited to the tested
conditions; failed cases are retained and excluded explicitly.

In nine ordinary CPU-6 teacher games, it created openings in every game:
26 Mewtwo openings, 20 Fox openings and one trade. It lost a stock in every game.
Three ordinary standing-Fox teacher games each produced three openings with no
Mewtwo stock loss. Passive repeatability is not evidence of human-match strength.

### What does not work reliably

**No learned Mewtwo checkpoint is promoted.** Five rounds have been trained:

| Round | First openings vs standing Fox, 3 games | First openings vs CPU 6, 3 games | Standing stock losses |
| --- | --- | --- | --- |
| 1 | 0 / 0 / 0 | 1 / 1 / 2 | 0 / 0 / 0 |
| 2 | 12 / 9 / 5 | 0 / 3 / 2 | 1 / 0 / 0 |
| 3 | 4 / 2 / 1 | 0 / 0 / 0 | 0 / 0 / 0 |
| 4 | 0 / 1 / 1 | 0 / 2 / 2 | 0 / 0 / 0 |
| 5 | 5 / 2 / 0 | 0 / 1 / 1 | 1 / 0 / 1 |

Each game is 30 scored seconds. These are development tests, not confidence
estimates. Round 1's standing runs repeated because of a subsequently fixed
sampling seed bug. Scorer corrections are reflected in the table.

Round 4 sometimes repeatedly down-tilted from about 27 units away and missed.
Round 5 used more complete teacher gameplay and a wider model, but still failed
opening consistency and standing survival. Its best weighted training loss was
0.0072696: fitting the demonstrations closely did not solve live play.

All completed rounds delivered inputs at latency 1. Round 5 demonstrated all
requested move categories and avoided prolonged shield holds. **No Mewtwo
graphical qualification has been performed.**

Latest experimental weights:

`eval_runs/0915_mewtwo_neutral/v5/candidate.bin`

SHA-256: `baa6396bf935205e5a17824a1d8bb084dda3b4a582b2eca3da24ff2e16ecb303`.
GRU, hidden 128, two layers, window 16, zero recurrent initialization, f32,
autoregressive controller head, previous action, queue depth 1, reaction delay 0.
The filename is not an alias for an approved model. Earlier weights remain in
their own `v1`–`v4` directories.

## 3. Completion milestones

| Milestone | Deliverable | Evidence required |
| --- | --- | --- |
| A: measurable conversion | Scorer and repeatable first-fair harness | Distinguish first hit, second hit, escape opportunity, interruption and invalid setup |
| B: approved teacher | At least one nontrivial fair-continuation region | Both facings, measured percent/DI coverage, negative cases, actual second-hit contact |
| C: learned conversion drill | Sampled model converts eligible first fairs | Frozen unseen trial configurations; report successes and every failure |
| D: reliable neutral | Corrected sampled neutral model | Existing six-game neutral gate, without weakening it |
| E: integrated bot | Creates fair openings and converts eligible ones | Separate opening and conversion rates, safe exit back to neutral, graphical local play |

A–C can precede D. E requires C and D. Do not advertise an isolated conversion
drill as a neutral bot, or two hits on a passive dummy as a guaranteed combo.

## 4. First work packet: define and measure a fair conversion

### 4.1 Define the event, not just the animation sequence

Implement a separate proposed `ExPhil.Eval.FairConversion` scorer. Do not change
`NeutralExchange` to count combo hits as fresh openings.

For each trial record:

- Actual first-fair contact: Mewtwo's active fair, Fox's damage/hit response and
  attacker/defender hitlag. Ambiguous attribution is a reported invalid case.
- Contact frame and positions, Fox percent **before and after** first contact,
  facing, both velocities, Mewtwo height, grounded state and jumps remaining.
- Actual L-cancel result for a landing branch, jump timing, aerial identity,
  second contact frame and elapsed simulation frames.
- Fox's first opportunity to act between contacts. Account for hitlag, hitstun,
  landing/tech transitions and the relevant action's interruptibility; do not
  equate every non-hitstun state with freedom to act. Verify boundary cases in
  live traces/source before relying on this classification.
- Exit condition: true two-hit conversion, gap followed by another hit, escaped,
  whiffed follow-up, Mewtwo interrupted, no continuation attempted, stock lost,
  timeout or invalid setup.

A second hit before Fox can act is the initial true-combo criterion. A hit after
Fox could act is a separate string/read outcome. A failed escape input is not
proof that no escape was possible. Trades and shield contacts stay separate.
Keep exact frame evidence; until actionability is verified, label that result
uncertain rather than certifying a true combo.

Test the scorer on synthetic boundary cases and inspect real replay traces:
single hit counted twice during hitlag, two distinct hits, same-frame trade,
actionable gap, landing/tech escape, wrong aerial, missing frame and stock reset.

### 4.2 Build a small, reproducible first-fair harness

Extend the existing live Probe-based harness or add a dedicated Elixir script.
Use Dolphin as the initial gameplay reference. The separately developed Melee
simulator has not been qualified as an oracle for this Mewtwo drill.

Start near FD center, with grounded Fox, repeatable spacing and zero movement.
Separate two setup families:

1. Low/descending fair → land/L-cancel → jump and pursue.
2. Rising fair → stay airborne → appropriately timed double jump and pursue.

First prove that the intended initial fair actually connects in each setup.
A missed first hit is a setup/entry failure, not a failed combo continuation;
retain it in the end-to-end denominator. Conditional conversion rates may use
only valid first hits, but must report that denominator explicitly.

Reset and log stock counts, percent, stale-move state, position, velocities and
jump availability. Prefer fresh matches or a verified full reset. If setting
percent directly, verify the resulting game state and intended damage semantics;
do not assume editing a HUD value establishes a valid controlled setup.

Save raw observations, physical controller inputs, complete replay, setup values,
code/checkpoint hashes, emulator build/options and the measured input delay.

### 4.3 Sweep a modest matrix before inventing percent rules

Proposed discovery grid, adjustable **before collection** with a recorded reason:

- Fox starting percent: 0, 20, 40, 60, 80, 100, 120.
- First-fair family: low/landing and rising.
- Facing: left and right.
- Initial defense: neutral stick, then fixed DI in, out, up and down.
- Candidate continuation: fair and up air, with bounded searches over legal
  jump/attack delays and pursuit drift. Test the landing and airborne branches
  separately rather than treating them as interchangeable.

First screen neutral DI to find candidate windows. Then expand promising windows
to other DI and finer percent steps. Record searched timing ranges and rejected
candidates. Exhausting a bounded search without a hit means **unresolved within
that search**, not proof that no combo exists.

Keep pre-contact defense fixed during this first sweep. Apply DI at the correct
hitlag/launch timing and record it. Verify whether the chosen stick schedule also
causes SDI or ASDI; do not silently call the combined effect pure DI. Add crouch,
SDI, tech choices and actionable escape attempts as separately named challenges.

DI in/out must be defined relative to the launch/attacker and mirrored correctly.
The defender schedule is privileged evaluation information: the eventual policy
must react to its available observations, not read future DI or future trajectories.

## 5. Teacher design and approval

Build a **stateful fair-conversion teacher**, separately from the neutral teacher.
Proposed phases: confirm first hit → choose a currently supported branch → pursue
and execute → confirm second hit or abort → return control to neutral.

Use percent, observed relative position/velocity, height, current action, landing
status and remaining jumps. Re-evaluate when new launch motion becomes observable.
Do not commit to the same second hit for every fair.

Mechanical pitfalls to test explicitly:

- The current `Melee.Tech :shffl` routine commits through fast fall and landing.
  It is not already a rising-fair → double-jump follow-up controller. Reusing it
  unchanged can miss the airborne branch's decision window.
- Existing `:djc_aerial` support does not certify the required pursuit height or
  jump timing. Validate Mewtwo's actual trajectory instead of treating all double
  jump aerial inputs as equivalent.
- Release and re-press aerial/jump inputs correctly; a held input is not a fresh
  edge. Preserve facing-sensitive fair inputs and verify actual aerial states.
- Confirm L-cancel success from the game, not merely from an L button press.
- On interruption, discard stale commitments. On a whiff/shield/no confirmed
  first hit, do not blindly execute a combo branch.
- Include no-jump, wrong-facing, high/low launch, out-of-range and near-edge
  negatives. Safe abandonment is a behavior to validate and train.

Suggested initial certification gate, to freeze before student fitting:

- Each approved configuration repeats correctly three times, with no ambiguous
  attribution. Deterministic repetitions verify mechanics, not a statistical
  reliability claim.
- An approved branch covers a declared percent interval and more than a single
  favorable DI example, in both facings. Publish the exact discrete tested cells;
  do not imply every intermediate value was tested.
- Every admitted true-conversion label has a verified no-actionable-gap result.
- Required negative cases abort/reposition without a blind follow-up or an
  unforced SD. Unsupported configurations remain explicitly unsupported.

If neither candidate branch passes, inspect spacing, launch path, landing and
jump timing before training. Request a targeted human demonstration only when
it would resolve a specific missing technique. Do not ask for another large
undirected recording session by default.

## 6. Train an isolated conversion specialist

Freeze a split by complete trial configuration/source trajectory before fitting.
Keep related timing variants, mirrored/augmented copies and repeated versions of
one source in the same split. Hold out intermediate percents, new spacing and
separate defender schedules for generalization tests. Report truly unsupported
DI separately from failures inside the declared supported domain.

Export **actual causal physical inputs**, with enough pre-contact history to
represent approach, first-fair phase and committed actions. History-only prefixes
receive no target loss. Train pursuit, successful conversion and validated
abandonment; omit neither failures from reports nor needed recovery/exit behavior
from the data. Do not relabel incoherent single frames into a supposed combo.

Start with the existing small recurrent/AR infrastructure and a separately named
experiment, e.g. `eval_runs/0915_mewtwo_fair_conversion/v1/`. Exact architecture,
source mix, training budget and test counts must be written before fitting.
Do not automatically adopt v5 as the best initialization: it is a failed neutral
candidate, not a certified conversion model.

Proposed student gate to freeze after discovering the teacher's valid domain:

- At least 20 sampled trials per declared supported branch, balanced over both
  facings and held-out percent/DI/spacing configurations.
- At least 90% verified two-hit conversions in each supported branch, with raw
  counts and per-configuration failures. This is an engineering promotion target,
  not a confidence claim from a large population study.
- Report missed first hits separately and report the unconditional two-hit rate.
  Resetting to contact may evaluate continuation only; it cannot qualify entry.
- No unforced SDs in the controlled central-stage trials; successful, tested exit
  behavior for explicitly unsupported states; latency 1 and no inference errors.

If it fails, collect validated corrections at the student's failure states.
Do not silently narrow the held-out set, change sampling or add epochs indefinitely.
A deliberately narrower domain requires a new named protocol and a candid claim.

## 7. Repair neutral with targeted corrections

The next neutral packet is **not another untargeted fit**. Use preserved v4/v5
replays to locate too-far down tilts and movement leading to edge SDs.

Implement a teacher-takeover collection path: allow the sampled student to reach
a problematic state, then let the stateful teacher take over. Preserve the real
student action history as input-only context. Label the subsequent actions the
teacher actually executes. Do not synthesize teacher actions over an incompatible
student history or restore state fields to conceal rollout drift.

Validate each corrective rollout: appropriate approach/retreat, actual opening
where reachable, and safe interruption/edge handling. A failed teacher takeover
is evidence of a teacher gap, not an automatic training label. If using replay
prefix restoration, require exact prefix state/input auditing first; see the
float-input and signed-zero compatibility notes below.

Retain the existing neutral gate: three standing-Fox and three CPU-6 ordinary
30-second games; an opening in each; all requested move categories represented;
no shield-action streak over 120 frames; no standing-opponent SD; valid latency 1.
Then test the actual graphical runner. These tests remain small development gates.

## 8. Integrate without hiding the handoff

Preferred first path: train one controller with coherent neutral → first fair →
validated conversion → neutral sequences. The teacher may compose state machines
for collection; that does not make the final learned controller scripted.

If separate learned specialists are used instead, explicitly document the runtime
selector and test it. It must use available observations, confirm a real opening,
preserve committed-action history and apply a declared recurrent-state policy.
No future replay labels or assumed DI may drive the switch. Neither architecture
is implemented or chosen as an irreversible requirement here.

Avoid training a conflict: the current neutral teacher often waits during an
opponent's hitstun. For an approved fair conversion, the conversion teacher must
own that interval. Audit overlapping labels and route-specific ownership rather
than blindly concatenating opposing demonstrations. Keep valid waits elsewhere.

Integration acceptance:

- Keep neutral opening counts separate from combo conversion counts.
- Record the number of eligible fair openings, successful second hits, failures
  and unsupported openings; also report two-hit sequences per game. Never improve
  the conversion percentage by hiding entry failures or undefined eligibility.
- Repeat sampled controlled tests and ordinary CPU-6 games with frozen counts.
  If too few eligible fair openings occur, report insufficient evidence and run
  the predeclared larger block rather than claiming a high rate from one hit.
- Preserve the neutral gate and check that adding follow-ups does not create
  new shield holds, edge SDs or a collapse to one neutral attack.
- Verify ordinary-speed graphical play, input delivery and rematches before
  recommending a human recording. Publish actual checkpoint path/hash and a
  direct `mix run scripts/play_dolphin.exs` command, as the user prefers.

## 9. Workspace map and execution instructions

Working repository: `/home/blewf/git/exphil`.
Shared mechanics: `/home/blewf/git/libmelee_ex`.
Read applicable `AGENTS.md` before editing. There is substantial existing dirty
work across this workspace; inspect it and do not reset or overwrite it. The user
also has another agent working on `/home/blewf/git/melee-sim-light`.

### Existing sources and evidence

| Path, relative to ExPhil | Use |
| --- | --- |
| `docs/planning/MEWTWO_NEUTRAL_DRILL.md` | Original neutral requirements and teacher audit rationale |
| `docs/planning/MEWTWO_NEUTRAL_BASELINE_V1.md` | Pre-fit declarations and results for rounds 1–5 |
| `eval_runs/0915_mewtwo_neutral/README.md` | Current model status and round comparison |
| `eval_runs/0915_mewtwo_teacher_audit/README.md` | Approved controlled trials, failures and live teacher evidence |
| `lib/exphil/agents/mewtwo_neutral_teacher.ex` | Current stateful neutral teacher |
| `lib/exphil/agents/mewtwo_fair_expert.ex` | Older fixture-based fair expert; inspect, do not approve wholesale |
| `test/fixtures/replays/mewtwo_fair_chains.slp` | Existing recorded fair sequence; requires contact/DI/continuation audit |
| `../libmelee_ex/lib/melee/tech.ex` | SHFFL, aerial direction, double-jump and other mechanical routines |
| `lib/exphil/eval/neutral_exchange.ex` | Exclusive first-opening scorer; not a combo scorer |
| `lib/exphil/eval/mewtwo_neutral_benchmark.ex` | Existing neutral qualification metrics |
| `scripts/validate_mewtwo_contact.exs` | Controlled live teacher scenarios |
| `scripts/validate_mewtwo_cpu_teacher.exs` | Complete CPU or standing-Fox teacher games |
| `scripts/build_mewtwo_neutral_recordings.exs` | Approved controlled-case physical-input exporter |
| `scripts/build_mewtwo_human_openings.exs` | First-opening clips; intentionally stops before contact, so insufficient for combos |
| `scripts/build_mewtwo_teacher_games.exs` | Complete scored teacher-game exporter, excluding finalization |
| `lib/exphil/training/recorded_frames.ex` | Causal recorded-data and history-prefix contract |
| `scripts/dagger_drill.exs` | `--expert recorded_mewtwo` training; forbids accidental teacher relabeling |
| `scripts/eval_mewtwo_neutral.exs` | Sequential six-game sampled qualification |

New scorer, conversion teacher, conversion harness and takeover collector names
above are **proposed work**, not existing commands to run.

### Human data split

Five preserved games live in `eval_runs/0915_mewtwo_demonstrations/replays/`.
Mewtwo is P2, CPU Fox P1. CPU level was 6 and mostly 9, not known per recording.
The whole FD game (`Game_20260915T165636.slp`) is training-assigned; nineteen
selected opening clips entered rounds 3–5. The other four whole games remain
held out. The frozen manifest is `eval_runs/0915_mewtwo_neutral/v3/human_split.json`.

Audit additional follow-up portions of the training-assigned FD game if useful.
Do not silently move held-out games into training or call reused source frames
unseen combo evaluation. Existing old fixtures also need explicit split ownership.

### Commands and environment

Use `devenv shell -- ...` for Mix, EXLA and Dolphin-backed jobs. Run one
GPU/training/live job at a time; do not compete with the user's local game or
the other agent's GPU work. Check active processes before starting a long run.
Do not launch a second Mix job that blocks on the first one's build lock.

Targeted existing tests:

```bash
cd /home/blewf/git/exphil
devenv shell -- mix test \
  test/exphil/agents/mewtwo_neutral_teacher_test.exs \
  test/exphil/eval/neutral_exchange_test.exs \
  test/exphil/eval/mewtwo_neutral_benchmark_test.exs
```

The last direct ExUnit run of these tests passed 18 tests. Add meaningful tests
for the new scorer and teacher; a passing old suite is not combo certification.

Training launcher, after creating and reviewing a **new** round's arguments:

```bash
devenv shell -- elixir scripts/local_zero_train.exs PATH_TO_NEW_TRAIN_ARGS.json
```

Existing neutral qualification, using a fresh output directory:

```bash
devenv shell -- elixir -pa '_build/dev/lib/*/ebin' \
  scripts/eval_mewtwo_neutral.exs \
  eval_runs/0915_mewtwo_neutral/v5/candidate.bin \
  eval_runs/0915_mewtwo_neutral/v5/qualification_NEW
```

Compile changed scorer modules before using a plain-Elixir runner, so it does
not load stale dev beams. The existing harness refuses to overwrite its output
directory. Qualification exits nonzero when the model fails; preserve the report.
Do not rerun the unchanged v5 just to look for a lucky pass.

Headless executable: `~/.local/share/slippi/exi-ai/dolphin-emu-headless`.
ISO: `~/isos/melee.iso`. Graphical Mainline path is the single directory argument
`$HOME/.config/Slippi Launcher/netplay-beta-nixos`; do not paste a wrapped AppImage
filename. CUDA jobs use `EXLA_TARGET=cuda` and
`EXPHIL_GPU_MEMORY_FRACTION=0.15`. Follow the existing play script's memory-card
and trigger-encoding setup; earlier black-screen and held-shield bugs were fixed.

### Data and diagnostic traps already found

- Slippi causal labels pair state[t] with physical input[t+1]. Do not shift twice.
- Live processed trigger pressure is not a physical analog-trigger target:
  digital L and Z can appear as 1.0 and 0.35 in processed readback.
- Preserve real history, including negative frames where needed for an initial
  action. Confirm how windowing uses that context rather than guessing.
- Isolated contact fixtures used one stock; ordinary games use four. Stock is
  an input. Existing scripted stock augmentation changes complete histories only
  and rejects stock transitions. Never automatically augment human decisions.
- Unkeyed sampling now uses OS randomness. Previously VM-local counters produced
  identical fresh-process trajectories. Keep explicit seeds where reproducibility
  is needed; do not confuse a fixed rollout with multiple independent trials.
- A long throw used to be counted as a fresh neutral opening. Use current scorer
  code and `summary_rescored.json` for rounds 1–2, ordinary `summary.json` later.
- `.slp` files can be gitignored. Use Elixir `Path.wildcard` or an ignored-file-aware
  search before concluding recordings are missing.
- Recorded replays can finalize asynchronously. Poll for a valid completed parse
  rather than treating the first short read as a corrupt recording.
- Replay arithmetic profiles matter for exact restoration. See
  `docs/planning/HUMAN_REPLAY_FAILURES.md`; do not normalize signed zeros or infer
  the profile from human versus CPU. A handoff also exists in the simulator repo:
  `agent_docs/HANDOFF_SIGNED_ZERO_EXPHIL_2026-09-15.md`.

## 10. Resume checklist and durable progress log

Current handoff state: **planning complete; fair-conversion implementation not
started; neutral v5 failed; no model promoted**. No conversion percentage windows,
DI coverage, combo teacher approval or graphical Mewtwo success should be inferred.

Recommended next packet, small enough to finish before another usage limit:

1. Read this file and the evidence map; inspect current git/process state.
2. Audit the existing fair-chain fixture and the training FD game's actual fair
   contacts. Record candidate examples and uncertainty, without fitting a model.
3. Implement the conversion event scorer and its boundary tests.
4. Produce one reproducible first-fair contact on FD and retain its full trace.
5. Test one landing continuation and one rising continuation at a fixed percent
   and neutral DI. Report what actually happened; do not claim a percent window.
6. Update this section with exact output paths, commands, failures and next action.
7. Only then expand the matrix, certify labels and begin a named training round.

At every stopping point append a dated entry containing:

```text
Completed:
Changed files:
Commands/tests and outcomes:
Artifacts and hashes:
Supported configurations / retained failures:
Active processes (PID or session, command, log, whether safe to stop):
Next exact action:
Unresolved question or blocker:
```

Keep experiment protocols beside their artifacts; preserve original failures.
Do not mark the overall task complete because a checkpoint exported. Completion
means the declared teacher, student, integration and graphical gates passed,
with a runnable model and a truthful statement of its supported domain.
