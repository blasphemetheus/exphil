# Mewtwo teacher validation

The new stateful `MewtwoNeutralTeacher` is approved for the individually passed
Fox/FD situations below. This does not approve the old FairExpert or ComboExpert
wholesale, or establish performance against a human opponent.

## Evidence

- `teacher_v2/approval.json`: 16/16 controlled first-opening trials, both facings,
  versus standing, shielding, walking and repeated-jab Fox. Includes down tilt,
  run-up grab, nair, fair, approach wavedash and defensive repositioning.
- `shift20_v1/approval.json`, `shift40_v1/approval.json`: 13/14 approved in each
  shifted-position battery. Each excluded right-facing requested-nair case produced
  fair instead. Exports explicitly exclude both failed cases.
- `interruption_v1/summary.json`: 2/2 forced-hit trials recovered and landed an
  attack after damage. Fox gets the first hit in these trials by design.
- `live_fox_execution.log`: actual Fox/FD short/full-hop apices 12.654/33.408;
  8/8 alternating wavedashes, approximately +/-38.805 units after settling.
- Approved aerial sequences show short-hop height and successful L-cancel flags
  at landing. Wavedashes require an airborne diagonal dodge input, special landing,
  and signed travel. Immediate ground collision can skip observable action 236;
  the delivered input is preserved.
- Exclusive exchange scoring agrees with each approved trial's first hit/capture.
  Shield contacts do not count as hits; combo hits do not add openings.

## Fixes and retained failures

- Fixed-right C-stick fair produced bair while facing left. Fair/bair now follow
  facing in SHFFL, DJC and float-cancel, with regression coverage.
- Nair's drift previously started only while falling. It now starts after the
  initial neutral A input, retaining nair while permitting rising drift.
- Fair missed from about 18 units and connected from about 9. Both results remain.
- The first close-jab defense landed in another jab. The teacher now shields the
  immediate threat, creates space and approaches through a short-hop aerial.
- A harness rename temporarily disabled the teacher in edge-defense cases
  (`teacher_v1`). `teacher_v2` reran the complete corrected battery.
- Live snapshots expose processed trigger pressure: digital L appears as 1.0 and
  Z as 0.35. `teacher_v2/input_parity.json` records the difference from physical
  replay input. Round 2 exports use Peppi's causal physical inputs and replay
  hashes. Round 1's dataset remains preserved.
- VM-local seed counters repeated sampled trajectories across fresh processes.
  Unkeyed sampling now uses OS randomness; explicitly keyed analysis remains
  reproducible. Deterministic decoding is not enabled.

Original logs, failed traces, full live `.states` snapshots, and finalized Slippi
replays remain under their named run directories. Only selected approved cases
enter the new training exports.

## Training scope

Round 1: 1,942 frames, 16 sequences. Failed ordinary-start qualification: no
standing-Fox openings and no wavedashes; CPU games yielded 4 Mewtwo versus 22 Fox
openings. All six sessions had measured latency 1. Standing runs repeated the
same sampled trajectory and are not independent reliability estimates.

Round 2: 4,899 frames, 42 approved sequences, broader positions and physical
replay inputs. See `docs/planning/MEWTWO_NEUTRAL_BASELINE_V1.md` for the declared
qualification criteria. Teacher approval does not qualify the student.

Round 3: 8,778 frames, including nineteen audited successful opening clips from
the human FD game. The other four whole human games remain held out. All requested
move categories appeared and standing games had no SDs, but all three CPU games
had zero first neutral openings. Counter-hits are reported separately.

Round 4 adds 26 successful opening clips from nine ordinary four-stock CPU-6
teacher games (`cpu_teacher_v2/summary.json`). The teacher opened Fox in all nine:
26 Mewtwo wins, 20 Fox wins and one trade. It lost a stock in every game; offstage
recovery is not implemented. This approves an initial opening source, not human
neutral strength. The initial `cpu_teacher_v1` harness read the asynchronous
replay writer too early; v2 retries until the finalized replay parses.

The isolated teacher fixtures used one stock, a mismatch with normal starts.
Round 4 varies stock counts across whole scripted histories, preserving inputs
and rejecting stock transitions. The teacher does not read stock count. Human
histories stay unchanged. Real negative countdown frames provide input-only
context for opening-frame targets. The resulting fit contains 38,515 frames in
339 sequences; qualification results live under `0915_mewtwo_neutral/v4`.

Combo routes, broader matchup/stage coverage and human neutral strength remain
outside this initial qualification. No Mewtwo student has been promoted yet.

Round 4 failed opening consistency: standing games yielded 0/1/1 openings and
CPU games 0/2/2. It passed movement coverage, standing survival, shield duration
and input delivery. Round 5 uses complete scored teacher gameplay, including
interrupted approaches, rather than only successful CPU clips. An additional
three ordinary standing-Fox teacher games each produced three openings with no
Mewtwo stock loss (`stand_teacher_v1`). The model width increases from 64 to 128;
this is a combined data/capacity experiment. See the neutral experiment README
for current student qualification status.

Round 5 also failed student qualification: standing openings 5/2/0, CPU openings
0/1/1, and standing stock losses 1/0/1. Teacher validation and low imitation loss
have not yet transferred to a reliable sampled neutral policy. The next proposed
data source is audited teacher takeover from the student's actual failure states.
