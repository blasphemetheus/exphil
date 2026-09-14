# Input verification and history diagnosis

## Scope

Offline examination of the saved tiny-overfit recordings and current training /
agent history construction. No training, Dolphin launch, native rebuild, or
score promotion. Original fail-closed evaluation results remain unchanged.

## Input mismatch is not evidence of an extra frame

`scripts/diagnose_scenario_inputs.exs` compares individual controller components
against raw parsed replay pre-frames at send time + 1, and scans digital-button
alignment over offsets 0–8. Reports in `eval_runs/0913_tiny_overfit/`:

| Checkpoint | Sent button patterns matching at +1 | Analog-mismatch frames |
| --- | ---: | ---: |
| Initial | 1,440 / 1,440 | 1,402 |
| Corrected, before boundary fix | 1,440 / 1,440 | 1 |

Each initial-control run uniquely matches all digital patterns at offset +1
among the tested offsets. This is strong evidence of correct send timing, not
a full certification of analog delivery or a replacement behavioral score.
Reports: `components_initial.json`, `components_corrected.json`.

The corrected candidate's sole mismatch is frame 2347 in neutral-start 2228,
run 1: sent `button_r: true`, shoulder 0; recording has right trigger 1 and all
sticks/buttons match. The verifier compares requested shoulder to the **sum**
of recorded L/R triggers, incorrectly treating this digital-click consequence
as a delivery mismatch. The bridge sends analog shoulder to L separately from
digital buttons, so the verifier's single summed channel loses that distinction.

Initial-control examples also show:

- Requested main X 0.375 records as 0.5 (neutral), while main Y 0.3125 matches.
- Requested C-stick (0.0625, 0) records approximately (0.175, 0.125).
- Requested left shoulder 0.25 records 0; requested 1 records approximately
  0.178571 in sampled commands, while digital R produces right trigger 1.

These are observed analog transformations, not proven packet loss. Native
Peppi parsing reads `pre.joystick`, `pre.cstick`, and `pre.triggers_physical`;
libmelee_ex quantizes requested axes/triggers before transmission. Its existing
live analog test already documents deadzone readback differing from requests,
but only tests main-stick horizontal sweeps. It does not establish the complete
two-axis/trigger contract needed here. Do not guess a normalization formula or
increase the tolerance until these failures disappear.

### Required correction

Separate digital timing, stick readback, and left/right trigger checks. Establish
the exact wire-to-recording analog contract with a deterministic sweep covering
deadzone boundaries, diagonals, shoulder buckets, and L/R clicks. Preserve these
observations as regression fixtures; accept only specified transformations,
while retaining rejection of wrong offsets, dropped inputs, and missing frames.
Then re-audit the saved recordings under a versioned verifier. Current scores
are still withheld; no updated pass rate is claimed in this diagnosis.

## History contracts differ at the training handoff

`scripts/audit_clip_history_contract.exs` uses the actual corrected Data sequence
target mapping on all six recorded teacher clips. `history_contract.json` shows
118 targets per clip, 15 padded input windows per clip, and 18 windows containing
embeddings whose previous-action queue is not yet filled to depth 3.

The first training decision sees 16 copies of the handoff embedding with an
empty action queue. A saturated committed-prefix evaluation instead sees the
preceding 15 frames plus the current frame, with a populated decision queue.
Even after input padding ends, those early queue-reset embeddings remain in the
window for three more targets. This is a structural contract comparison, not
a measurement that all actual embedding values or outputs differ.

The committed-history **clock** is consistent: at prefix frame t the suite
observes recorded input t + 1 + delay, and Agent embeds before pushing that
controller. At the next frame this is the previous committed decision, matching
delay-shifted recorded labels once training history is available. There is no
new off-by-one found in that lookup. The problem is different initialization /
available context, not a justified reason to alter the delay ID.

### Required correction

For the narrow memorization proof, add an explicit cold-at-handoff policy-history
mode that resets Agent's recurrent buffer **and** controller queue without
changing the physical replay prefix or pending input delivery. Evaluate that
mode as the matched training contract; keep the warmed-prefix result separate.
For the eventual continuously running bot, train with real input-only prefix
context and masked prefix supervision, or otherwise validate recovery with
populated recurrent history. Cold-start success alone is not continuous recovery.

## Next order

1. Verify and encode the analog readback contract, with component-level errors.
2. Add a matched cold-history proof mode and regression for history/queue reset.
3. Re-audit saved inputs; run a separately recorded bounded overfit proof using
   the boundary-fixed trainer. Do not reuse the old run as evidence for the fix.

## Follow-up: conversion and matched-history fixes validated

The conversion sweep identified and fixed a real libmelee_ex pipe-trigger scaling
bug, separately from expected stick deadzones/clamping. The versioned verifier
now compares converted sticks and independent triggers, with component errors
and a tighter floating-point tolerance. Both 549-command sweeps validate under
their explicit old/new profiles. All 12 initial and all 12 corrected saved runs
revalidate under the historical profile; original scoreboards remain untouched.

`--prefix-history cold` is implemented and live-tested at two handoffs with the
untrained checkpoint: complete responses, zero drift/errors, valid timing.
Regression coverage includes actual Agent buffer/queue reset without changing
physical prefix or pending delivery. No retraining launched.

See [contract, evidence, and usage](../guides/INPUT_READBACK_CONTRACT.md).
Next: a separately recorded bounded run with the boundary-fixed trainer and
matched cold-history evaluation, then a distinct warmed/continuous recovery test.
