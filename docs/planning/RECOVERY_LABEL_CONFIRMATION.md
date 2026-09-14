# Recovery-label closed-loop confirmation — 2026-09-13

## Verdict

**The off-loop HOLD projection is wrong on observed recoveries.** This is
confirmed against the teacher's actual future commands in Dolphin, not another
label table. It does not yet prove how much of G26's live regression it caused;
that requires a controlled corrected-label training comparison.

No expert behavior, production training labels, guard, or checkpoint was changed.

## Experiment

`scripts/scenario_suite.exs --driver teacher --audit-teacher-labels` now records
the issued command and the expert's predictions at shifts 0, 3, 4, 5. After the
run, `ExPhil.Eval.RecoveryLabelAudit` compares each prediction with the command
the teacher actually issued at frame `t + k` while controlling Dolphin. It uses
the same live-to-parsed conversion and previous controller as the teacher.

The teacher runs at the suite's native one-frame application latency, without a
policy delay queue. The audited shifts are **future expert issuance offsets**,
not claims that the teacher itself was evaluated at reaction delay 4. Policy
comparisons use `--reaction-delay 4`, sampled temperature 1.0, and prefix history
warm-up. The opponent replays recorded inputs; it is not reactive.

Digital buttons must match exactly; analog components have tolerance 0.02.
Incomplete horizons and gaps are unscored. Shift 0 checks the instrumentation
wiring, not independent expert correctness. Full per-frame comparisons are in
the scoreboard's `teacher_label_audit`; exclude errored, drifted, or truncated
runs before aggregating. Frame counts are correlated observations, not
independent statistical trials.

## Controls and clean recovery results

The known-good mid-chain teacher control (ep57 replay, frame 900) has zero
handoff drift, chain 14, and **zero mismatches** at shifts 3/4/5 on 117/116/115
comparisons respectively (`control_teacher.json`).

Three grounded off-loop handoffs were selected from the parsed G26 ep33 stand
replay before examining their teacher futures: reflector hold at frame 4,
landing at 75, and landing at 146. They are fixed in
`scenarios/ms_recovery_label_confirmation.json`. All three reproduce without
handoff divergence; the teacher achieves chain 13 in each, with reentry at 9
frames (`teacher_grounded.json`).

| Issuance offset | Off-loop mismatches | On-loop mismatches |
|---|---:|---:|
| 0 | 0 / 21 | 0 / 339 |
| 3 | 16 / 21 | 0 / 330 |
| 4 | **18 / 21** | **0 / 327** |
| 5 | 18 / 21 | 0 / 324 |

At shift 4 the reflector-hold case contributes 4/7 mismatches; each landing
case contributes 7/7. This localizes the defect to the recovery transition,
while preserving an executable happy-path control.

### Do not blindly replace HOLD with phase `k - 1`

The landing handoff at frame 75 remains in action 42 through frame 78, visits
reflector startup 360 at frames 79–81, and enters canonical jumpsquat 24 at
frame 82. At frame 78 the held prediction for offset 4 is B without X; the
actual frame-82 teacher command is B+X. At frame 80 the held prediction is no
buttons, but frame 84 requires B. The off-loop previous-button alternation,
landing lockout, startup, and eventual entry all matter.

The reflector-hold handoff likewise takes seven observed frames to reach a
canonical state. These are counterexamples to assuming every grounded
recovery enters the canonical loop one frame later. They do not establish a
general transition rule for neutral standing, aerial recovery, or hitstun.

## Broader replay pilot: exclusions matter

The first runner used the entire existing break manifest (36 entries, not just
the first 12). Results: 5 clean handoffs, 28 drifted, 3 games ended during the
prefix. Its strict gate stopped before launching policy comparisons. Retain
`teacher_breaks.json` and its logs; do not treat this as a successful 36-case
benchmark or repeat the historical "all 12 pass" claim for this execution.

The first original break case (r1, frame 488) is clean: teacher chain 13,
10/11 off-loop mismatches at offset 4, and 0/105 on-loop mismatches. The short
G26-own-replay cases above provide the cleaner primary confirmation. The
reason the other prefixes drifted has not been established in this work.

## Policy comparison stopped at its control

The ep57 mid-chain policy control at reaction delay 4, sampled temperature 1.0,
reproduced the prefix with zero drift but reached only chain 1 in both runs
(`grounded_control_ep57_k4.json`). Its response contains the familiar aerial
reflector float. The runner's predeclared minimum control chain of 10 therefore
failed and **the G26 policy cases were not launched**. This is not a new G26
performance result, nor proof that the latency mapping is wrong: calibration,
delay conditioning, and policy/harness history semantics still need separating.

The native teacher control and the teacher-future label comparison remain valid;
they do not rely on this failed policy control. Before drawing a matched-rung
policy conclusion, reproduce the historical passing ep57 control at reaction 2,
sweep its trained rungs without overriding delay IDs, and resolve why reaction 4
fails here. Do not evaluate G26 at reaction 2: G26 trained at 3/4/5.

## Artifacts and next step

Artifacts and reproduction scripts: `eval_runs/0913_recovery_labels/`.
Runs use prebuilt dependencies with `mix run --no-compile --no-deps-check`,
load only the new audit module in memory, disable the suite's global orphan
sweep, and launch no training. Output paths are protected against overwrite
by the reproduction scripts; use new names for repeated experiments.

Next: implement and independently validate transition-aware expert futures
for known recovery classes, explicitly abstaining where entry timing is
unknown. Preserve these observed recovery traces as regression evidence,
audit at all training shifts, then compare equal-budget retraining against
the unchanged labels. Do not promote a model from fixture agreement alone.
Restore a passing matched-rung policy control before interpreting that comparison.

Validation: 26 current-runtime audit/scoring tests pass, including five new
audit tests. A broader scan-suite attempt finds an unrelated existing missing
`ScenarioScan.detect(:multishine_reentry, ...)` clause in its fixture integration
test; the scanner is not changed here.
