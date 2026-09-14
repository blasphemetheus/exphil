# Recovery controls: warm history and opponent neutralization

Frozen checkpoint404a1f543889ddd4193380745fbc603c6c94bb4a200a3caa7cbca6ddf823182a.
No optimization or new labels.

1. Known-good six teacher handoffs, two runs each,120f, committed history,
   replayed opponent. Compare with the original12/12 cold-history result.
2. The same three human-hit handoffs (502,1389,3152), two runs each,360f,
   committed history, **neutral opponent inputs after handoff**. Compare with
   the prior0/6 full-cycle return under replayed human inputs.

All runs use Fox, reaction-delay2, temperature1.0, corrected libmelee AF;
prefix replay, drift checks and recorded-input timing gates are unchanged.
Inputs are not a matched random-seed replay, so comparisons are descriptive.

Neutralization cannot cancel active attacks or projectiles. Inspect further
damage before calling a response an isolated-hit case. Preserve censored
episodes and the distinction between a readiness proxy and exact actionability.

Launcher: `bash eval_runs/0913_recovery_controls/run.sh`.
New `--response-opponent neutral` affects response P2 only; default is replay.
No changes to the human replay or prefix inputs. Eleven focused tests pass.

- [x] Collect warm-history controls.
- [x] Collect neutral-opponent responses and check additional damage.
- [x] Record conclusions before proposing training corrections.

## Results

All18 executions have valid input timing, no replay divergence, no errors or
truncation. No training or checkpoint changes. The offline readback audit initially
used ControllerState field names on a raw Peppi.Controller; fixed and rescored
the saved replays without repeating any live runs. Twelve focused tests now pass,
including the raw controller layout regression.

### Warm-history control:6/12, versus12/12 cold

| Known handoff | Warm maximum chains (two runs) | Result |
|---|---|---|
| neutral2228 | 13,13 | both pass |
| neutral2566 | 13,13 | both pass |
| sustain900 | 14,14 | both pass |
| recovery4 | 1,1 | both fail |
| recovery75 | 0,0 | both fail |
| recovery146 | 0,0 | both fail |

This demonstrates sensitivity to actual preceding history at familiar recovery
states, not simply unfamiliar knockback. Every window still starts its internal
GRU state at zero; the difference is **observation/committed-action history**.
The recovery clips were fitted with cold repeated-first-frame context. Do not
"fix" this by resetting history whenever the bot is hit.

### Opponent neutralization:0/6 strict full-cycle returns

Every run records357/357 checked P2 frames as neutral after a three-frame settling
allowance. No further positive P1 damage increments occur in those checked
frames, and no stocks are lost. Active-attack carryover is not the explanation
for these observed failures (damage increments do not detect every possible
zero-damage interaction).

For hit1389 and3152, all four runs eventually reach the existing readiness proxy,
then have220,268,327,291frames of follow-up respectively without a full cycle.
One1389 run starts a second reported-hitstun episode without additional percent;
the scorer retains that censor rather than inferring an extra damaging attack.

The two hit502 runs never satisfy the **current conservative readiness proxy**,
but that does NOT mean continuous hitstun: one sits in action40 (CROUCHING) for
236frames (~3.9s); the other also spends long spans crouching. The proxy excludes
crouching. This is a measurement limitation to address explicitly/versionedly,
not evidence the bot was unable to act. Zero completed cycles is independent of
that readiness-clock limitation.

## Next implementation order

1. Preserve real pre-handoff observation/committed-action context as **input-only
   prefix** on recovery training clips, while keeping every early recovery target.
   Add train/eval window-equality regressions for both cold and warm modes.
2. Validate recorded teacher recovery from the neutral-opponent failure states,
   particularly landing/crouch-to-shine restart. Keep hitstun/lockout separate from
   opportunities to act; do not indiscriminately label every frame as shine.
3. Fit with those matched histories and validated targets in a bounded run,
   keeping previous-action dropout off. Require existing cold12/12, warm controls,
   and isolated-interruption recovery measurements together.

Also refine/version readiness reporting for crouching/other grounded controllable
states; retain raw action traces and censoring. No readiness metric was changed
for the results above. More model capacity or distillation is not the next step.

Artifacts: `eval_runs/0913_recovery_controls/results/warm_controls/eval_candidate.json`,
`results/neutral_opponent/eval_candidate.json`, `results/neutral_opponent/recovery.json`,
raw replays, logs, source hashes, and `tests.log`.
