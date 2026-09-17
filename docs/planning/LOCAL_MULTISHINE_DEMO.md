# Local, zero-added-delay Fox multishine demonstration

Goal requested 2026-09-15: a learned Fox that consistently multishines across
characters and stages, playable locally by Bradley and suitable for recording.

## Working scope

User-confirmed scope: stationary multishining, resuming after interruptions;
Fox, Falco, Marth, Peach, and Samus; Final Destination, Battlefield, Dream Land,
Yoshi's Story, Fountain of Dreams, and frozen Pokemon Stadium. Chasing and
shield-pressure navigation would require additional behavior supervision.

User clarified that resuming after landing or respawning is sufficient.
Returning from offstage is outside this demo's scope. Report stock losses
and whether Fox resumes after respawning; do not count an offstage death
as an unexplained grounded-resumption failure.

Zero added delay means causal state[t] -> input[t+1], reaction delay 0,
local online-delay setting 0. Use the synchronous runner and measure the
actual delivery latency. The existing async demo adds a decision frame and
the previous candidate was trained at reaction delay 4; neither can simply
be relabeled as zero delay.

## Ordered work

1. Train a separate reaction-0 baseline with the established small recurrent
   model: hidden 64, window 16, f32, zero recurrent initialization, AR head,
   previous action, queue depth 1, explicit delay ID 0, clean loss, 21 epochs.
   Use the canonical fixture plus expert-relabeled r1 CPU sources. At delay
   zero the expert can label recovery decisions directly, without guessing
   a delayed future. Keep r2 games and the human recording out of this fit.
2. Check actual next-frame delivery and cold game-start behavior on FD.
   Record every failure. Fix the first failed prerequisite before expanding.
3. Collect live states across stages/opponents. Hold out complete games;
   add targeted correction data for observed failures and retest the same
   checkpoint across the complete matrix. Keep platform/edge failures visible.
4. Validate the local GUI/human-controller path at zero added delay. Deliver
   a launch command, checkpoint, replay recording, and video capture guidance.

Do not claim general consistency from one uninterrupted chain. Report
per-stage/opponent starts, sustained chains, interruptions/reentry, deaths,
and measured delivery. Freeze a candidate only after a declared matrix passes.
Do not train from held-out outcomes without declaring a new split/round.

Experiments live under `eval_runs/0915_local_zero/`. Existing checkpoints,
default launchers, and the delay-4 coverage experiment remain reproducible.

## Initial matrix protocol (declared before running)

Use the frozen bootstrap checkpoint, deterministic actions, 30 scored seconds
after frame zero, and one cold start for each of the 30 stage/opponent cells.
First test standing opponents, then level-1 CPUs. Require replay metadata to
match the requested matchup, all 1,800 scored frames, no runner errors, and
measured one-frame input delivery. A standing cell passes the initial gate
with a chain of at least 30 and no deaths. Report CPU interruptions, completed
and censored resumptions separately; a CPU that never reaches Fox supplies no
interruption evidence. These are development checks, not independent statistical
replicates. Repeat qualification after any correction fit, with fresh games.

The two preliminary FD games counted countdown frames toward their nominal
30 seconds (1,679 scored frames each). Both reached chain 187. The matrix
runner now counts 1,800 post-countdown frames and excludes replay-finalization
inputs from scoring.

Also test the six existing held-out human replay handoffs on FD, including
three hit states. Reconstruct both ports with audited float inputs and the
source's accurate-nmsub profile; then give the reaction-0 candidate a 300-frame
response window with a neutral opponent and committed prefix history. This
isolates resumption after human interruptions without training on those games.
Require exact prefixes and valid input timing before interpreting chain scores.
This FD check complements the live CPU matrix; it cannot establish recovery
on other stages by itself.

## Replay-parser correction discovered during the first matrix

The original bootstrap completed all 30 standing matchups with measured
one-frame delivery, 1,800 scored frames per game, and no deaths. Its archived
`result.json` metrics used the old parser and must not be used to count human
or CPU hit recoveries. The parser incorrectly mapped hitlag into the hitstun
channel and treated reflector flags as invulnerability.

The corrected parser reads the same misc-AS counter as the live bridge,
exposes hitlag separately, uses the actual hitstun flag for recovery scoring,
and reads hurtbox collision state for invulnerability. The relevant definitions
are in the [Slippi post-frame specification](https://github.com/project-slippi/slippi-wiki/blob/master/SPEC.md#post-frame-update).
It also exposes the recorded frozen-Stadium setting so stage qualification
can verify that setting from the replay.

Refit the same recipe and whole-game split under `corrected_bootstrap/`.
Keep the original bootstrap and its matrix as historical evidence. Recompute
old replay metrics into a separate audit rather than overwriting old reports;
repeat live qualification with the corrected checkpoint. No held-out human
games or new matrix outcomes are added to this correction fit.

The old human manifest's `hit` labels were also based on the wrong counter.
Several handoffs precede the defender's hitstun, so neutralizing the opponent
at those handoffs does not guarantee an interruption. Preserve the six-case
test as a legacy handoff check. Add `human_verified_hits_manifest.json` with
seven handoffs whose preceding source state has the actual hitstun flag set:
305, 747, 752, 775, 2963, 3081, and 4448. Require audited exact prefixes and
valid timing, and report the entire 300-frame response, including misses.

## Corrected bootstrap evidence

The corrected 21-epoch fit used the same 39,074 frames and exported with loss
0.013497. Its first fresh FD game reached chain 200 over 1,800 scored frames,
with measured latency 1 and no runner errors. Unthrottled headless throughput
was about 99 game frames/second. Graphical real-time performance was subsequently
verified for FD/Fox; see the graphical startup evidence below.

All six legacy human handoffs passed with exact prefixes and valid timing.
All seven verified post-hit handoffs resumed with no deaths and chains of
18–32. The original scenario gate passed 4/7: handoffs 775, 2963, and 3081 took
87, 132, and 153 response frames to reenter, exceeding its 60-frame limit.
The separate replay audit measured 8–13 frames from grounded readiness to a
completed cycle for all seven; preserve both results, without relabeling the
original failures as passes. One episode was interrupted again before recovery.
These are situations from one held-out human game, not independent matches.

Artifacts: `corrected_human/report.json`,
`corrected_verified_hits/report.json`, and
`corrected_verified_hits/recovery_audit.json` under the experiment directory.
The corrected live CPU matrix is the next coverage check, followed by the
standing matrix and graphical/human-controller validation.

### Graphical startup fixed (2026-09-15)

The established async recipe successfully ran the old reaction-delay-2 model;
user confirmed mostly multishines with occasional unwanted laser/up-B actions.
The sync runner omitted its `memory_card: true` configuration. Preserving memory
card settings for graphical runs fixes Mainline's pre-video hang; headless runs
retain their prior setting. Earlier nonblocking-input and OpenGL experiments did
not fix startup.

Corrected zero-delay checkpoint graphical standing checks on Final Destination
against Fox:

- `eval_runs/local_zero_demo_20260915T170005`: 600 scored frames, 59.928 fps,
  max chain 67, 66 completed cycles, zero deaths/errors, measured latency 1.
- `eval_runs/local_zero_demo_20260915T170040`: 1,800 scored frames, 59.942 fps,
  max chain 200, 199 completed cycles, zero deaths/errors, measured latency 1.
  Render visually inspected; `render.png`, `run.log`, `audit.json`, session,
  launch metadata, and finalized replay preserved in that directory.

These establish graphical real-time standing behavior for one cell. They do not
complete corrected-model stage/character coverage or live human interruption
qualification. CPU matrix parent PID 3199366 remains paused for local play.

### Paused for user recording

User requested the direct `mix run scripts/play_dolphin.exs` command and wants
to record the current checkpoint. Keep GPU evaluation paused during this work.
The guide now leads with the direct command; the convenience wrapper remains
optional. Ordinary local play timing is the intended behavior, not removal of
Melee's inherent controller/display latency.

`corrected_cpu_matrix/partial_audit_before_recording.json` preserves the 16
completed result files: all timing-valid, 34 hit episodes, 27 hit reentries,
and four stock losses. Censored episodes remain in each report. Yoshi/Falco's
session and replay finished while parent PID 3199366 was paused; its result
still awaits consumption by that parent. Do not launch a replacement matrix
or resume the parent while the user is recording. Remaining coverage is still
pending; this partial report is not full qualification.

### First current-model live human replay

The direct human launcher is running successfully. Observed process arguments
and checkpoint SHA are saved in
`eval_runs/local_recording_20260915_120657/observed_launch.json`.
The first game finalized and the same process started a second game, confirming
that this launch reaches gameplay and supports a rematch with a human adapter.

`2026-09-Mainline/Game_20260915T120712.slp.audit.json` beside that session's
first replay scores Battlefield Fox versus Fox: 8,267 nonnegative frames,
373 completed cycles, max chain 42, and four stock losses. Of 34 disruption
episodes, 14 completed reentry and 20 were censored by another interruption
or death. Eleven of the completed reentries followed hitstun; three followed
other loop breaks. Grounded-readiness-to-cycle times range from 8 to 245
frames. The readiness proxy is not an actionable-frame oracle; for example,
the long episode includes additional non-hitstun actions and must be inspected
before interpreting it as idle model delay. No training changes are made from
this replay. The full audit preserves every episode.

This direct session did not request a session report, so its replay audit
cannot independently assert measured live input latency. The earlier graphical
standing runs provide separate one-frame timing evidence for this checkpoint
and runner. Preserve that distinction.

### Human shielding pathology: trigger compatibility fix pending live verification

User reports occasional sustained shield and shield breaks; airborne repeated
shines are acceptable. The first human replay and preceding graphical standing
replay both report constant left analog trigger 0.9142857, including every
scored frame. Human replay contains 719 shield-family frames. This is stronger
evidence of controller delivery trouble than a learned shielding preference.

The controller currently applies the bipolar Axis + inverse offset to every
pipe trigger. Added an explicit unipolar encoding option in libmelee_ex and
forwarded it through the bridge to both pipe controllers. The synchronous
local Mainline recipe selects unipolar encoding; the tested headless ExiAI
configuration retains bipolar encoding. No policy weights or airborne behavior
are changed. All controller checks pass (5 doctests, 2 properties, 15 tests).
Live verification is pending the user's current recording session ending;
do not claim the shielding pathology fixed until a fresh Mainline replay
confirms released triggers and resumption behavior.

### Trigger fix verified in Mainline

After the user closed the recording session, two fresh graphical runs used
unipolar trigger encoding with unchanged corrected-bootstrap weights:

- `eval_runs/local_zero_demo_20260915T171722`: FD/Fox standing, 600 scored
  frames, trigger exactly 0 throughout, zero shield frames, chain 67, 59.94 fps,
  measured latency 1, no runner errors.
- `eval_runs/local_zero_demo_20260915T171823`: FD/Fox level-1 CPU, 1,800 scored
  frames, trigger exactly 0 throughout, zero shield frames, chain 46, no deaths,
  59.938 fps, measured latency 1. Seven hit episodes: six completed resumptions
  in 11–13 frames after grounded readiness, one interrupted again before
  completion. Each directory includes `trigger_audit.json`, session metadata,
  launch identity, finalized replay, and run log.

This verifies the unwanted analog-pressure fix and a short interruption check.
The full matrix and a fresh human recording after this fix remain outstanding.
The same direct play command picks up the fix on next launch. No checkpoint
retraining or airborne-behavior changes were necessary. Keep the CPU matrix
paused while the user retries/records.

### Corrected CPU matrix completed

`corrected_cpu_matrix/summary.json` and `completed_audit.json` cover all 30
stage/opponent cells: 30/30 valid, each 1,800 scored frames with measured
latency 1 and no runner errors; minimum max-chain 31. The five Stadium replays
all confirm frozen Stadium. There are 45 hit episodes and 38 completed hit
reentries. Four stocks were lost; three observed respawns resumed in 11–12
frames from grounded readiness, and the fourth had no grounded readiness
before the recording cutoff. Overall disruption censor reasons (including
non-hit loop breaks) are four deaths, two repeat interruptions, two cutoffs.

The longest readiness-to-cycle interval is 212 frames on FoD/Samus, starting
at 865 and ending 1077. Inspection shows repeated ground/aerial shine cycles
as the platform descends from y≈14 to floor level; the fast chain resumes on
the floor. This is a visible moving-platform limitation, not shielding: the
headless trigger remains zero. A separate hit episode takes 119 frames after
the readiness proxy. Preserve these intervals in consistency claims.

The corrected standing matrix is now running under `corrected_stand_matrix/`,
with the same checkpoint and 30-cell/30-second protocol. The CPU parent has
exited; it no longer needs resuming. Prior pause instructions above are history.

### Final automated demo qualification

The corrected standing matrix completed: 30/30 standing gates passed, all
1,800 scored frames, no runner errors or deaths, measured latency 1, chain
range 96–200. All five Stadium headers confirm frozen Stadium. Replay hashes
were rechecked by the report script. CPU range is 31–198 with the outcomes
reported above. Both protocols match checkpoint SHA-256
`b224b9a6953095b107a78396e0642787022212f72f1865aa12ff5b4833835a2f`.

`eval_runs/0915_local_zero/demo_qualification.json` links both full matrices,
post-fix graphical trigger audits, pre-fix live human replays, and the guide.
The experiment processes have finished; the GPU is free. No further automated
matrix is pending. Initial demo qualification is complete, with the stated
moving-platform and offstage limits. A user video after the trigger fix remains
to be recorded; human play/rematches and graphical corrected trigger delivery
have been verified separately. No new human outcomes were used for training.
