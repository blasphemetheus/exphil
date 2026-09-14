# Frozen per-case learning and live failure diagnosis

Follow-up correction: [producer contract and matched handoffs](MATCHED_HANDOFFS_2026-09-13.md).
Do not widen the historical action-only table: use the actual libmelee
character/action indexing rule instead. The earlier table-based proposal below
is superseded by that producer-specific implementation.

## Checklist

- [x] Fix omitted `--snapshot-all`; cover omitted/false/true options.
- [x] One-epoch omitted-flag smoke: final export exists, no epoch snapshot.
- [x] Measure the already-selected candidate and initial control, without gradients.
- [x] Trace all five remaining failures against validated teacher responses.
- [x] Probe the actual Agent embedding seam for action-frame mismatch.
- [ ] Correct and regression-test the action-frame contract, including action 363.
- [ ] Test the saved candidate with teacher-consistent pending handoff inputs.
- [ ] Repeat the fixed 12-response proof; retain original-source recovery separately.
- [ ] If needed, rebalance early recovery targets before another bounded training run.

No new candidate selection, extra fitting epochs, promotion, or Dolphin run.
The one-epoch CLI smoke uses the existing initialization with a fresh optimizer;
it is not the evaluated candidate.

## Snapshot fix

The drill tests `opts[:snapshot_all] == true`, not strict `and` on omitted nil.
A regression evaluates the actual script guard with OptionParser's omitted,
disabled, and enabled representations, including invalid loss.
`eval_runs/0913_epoch_metric_validation/omitted_fixed.log` exits 0 and exports
after 122 batches / mass 7,785. No `_ep1.bin` is created. The unrelated inhibitor
child still prints its shutdown warning. The smoke runner now supports
`omitted` and `disabled` and checks export behavior.

## Frozen learning

Candidate: `eval_runs/0913_tiny_overfit_cold/candidate.bin`, unchanged from the
original experiment. No later snapshots or smoke checkpoint substituted.
Initial control scores zero conditional all-component matches on every case,
with mean joint NLL approximately 22.95–23.04.

Metrics are **unweighted joint action NLL and teacher-forced conditional argmax
matches**. Analog heads receive true earlier within-frame actions. These are
not free-running joint-decoder accuracy or live success probabilities. The
product likelihood is the teacher action's AR joint probability. Inference
disables dropout; parameters are frozen; clips use boundary-safe cold windows,
delay-2 labels, queue depth 3, and the original recorded fixture semantics.

| Case | Targets | Joint NLL | All-component matches | First 18 matches | Off-loop matches |
|---|---:|---:|---:|---:|---:|
| Canonical | 7,077 | 0.000146 | 7,077/7,077 | 18/18 | — |
| Neutral 2228 | 118 | 0.002887 | 118/118 | 18/18 | 4/4 |
| Neutral 2566 | 118 | 0.002888 | 118/118 | 18/18 | 4/4 |
| Sustain 900 | 118 | 0.067044 | 114/118 | 14/18 | — |
| Recovery 4 | 118 | 0.074861 | 114/118 | 14/18 | 3/7 |
| Recovery 75 | 118 | 0.037378 | 116/118 | 16/18 | 5/7 |
| Recovery 146 | 118 | 0.037398 | 116/118 | 16/18 | 5/7 |

All twelve teacher-row errors are button errors within the first six frames;
analog conditional argmax is correct everywhere. Teacher clips score 696/708,
but only 21/29 off-loop targets match. Overall 99.85% hides these sparse critical
errors. Recovery 4 misses X at relative frames 0, 2, 4, and B+X at 5; landing
recoveries miss B at 1 and X at 2. Sustain's first four cold targets also remain
underfit. This is real learning, not complete overfitting.

Even perfectly argmax-matched neutral starts retain uncertainty: the first
target has only about 0.82 joint probability. Temperature-1 sampling can still
choose the wrong opener. Argmax accuracy alone is not a sufficient gate.

## Confirmed live mismatches

### Pending inputs follow a different trajectory

The policy suite keeps two original-source commands in flight at handoff;
the executed teacher took over immediately. Delaying teacher labels does not
make those physical trajectories equal. Resetting model history does not reset
physical pending commands.

| Handoff | Mismatches in first two sends | First player-state divergence, relative frame (run 1 / 2) |
|---|---:|---|
| 2228 | 1 | 1 / 1 |
| 2566 | 2 | 1 / 1 |
| 900 | 0 | 5 / 4 |
| 4 | 2 | 7 / 7 |
| 75 | 0 | 4 / 4 |
| 146 | 0 | 4 / none in response |

Neutral handoffs have the same initial measured player state as their teacher,
but diverge at +1 before new delayed decisions can repair the prefix. At 2228
the teacher enters action 360 while the live run enters 39, then 24; both live
runs reach action 25 at +5 instead of establishing the teacher chain. This is
not evidence that the model failed to memorize its neutral teacher windows.

After state divergence, differences from teacher future commands are reference
trajectory differences, **not automatically wrong actions** for the student's
new state. Pending mismatch is not the only cause: matching-pending cases also
have imperfect behavior.

### Live action frames reach a parsed-space embedder

Saved `agent_runtime.af_convention` is explicitly `parsed`; the launcher omits
`--live-af`. Raw action frames disagree with the same-frame replay on
**1,059/1,440** observations, while action IDs match on all 1,440. Existing
conversion reduces that to **39/1,440**, all action **363** with measured
raw-minus-parsed delta **+1**. The conversion table lacks this action.
Thus simply adding `--live-af` does not completely repair recovery inputs.

An observe-only probe through the actual Agent seam compares the first cold
embedding with training. Normalization reduces max difference from about 1/60
to below 1.2e-7 on five of six handoffs; recovery 4 retains the action-363
mismatch. These are replay-reconstructed observations, not a fresh live run:
P1 raw AF comes from the trace; P2 is reconstructed using the convention inverse.

First-frame button argmax does not change and teacher-action probability changes
by less than 0.0001 within this probe. The mismatch is real but **not proven to
explain the behavioral failures**. The probe uses batch size 1; likelihoods
differ slightly from the batch-64 pool audit, so compare its normalization
variants with each other, not against another evaluation's likelihoods.

## The five failed responses

- **2228 runs 1 and 2, chain 1:** pending-prefix state divergence at +1; action
  25 lasts 11/19 frames, followed by prolonged aerial reflector states. Neither
  establishes a sustained chain in the response.
- **2566 run 2, chain 1:** pending-prefix divergence at +1, action 25 at +6,
  repeated aerial reflector states and another long jump later. Its sibling
  reaches chain 13: sampled behavior is unreliable, not uniformly unlearned.
- **Recovery 4 run 2, chain 6:** misses the initial teacher jump target,
  remains in action 363 through +6, reaches action 25 at +11 for 14 frames,
  and starts regular cycling only after the long aerial detour. Recovery is
  late, rather than absent.
- **Recovery 75 run 1, chain 7:** pending commands match, but early decisions
  diverge; state diverges at +4 and exits the teacher loop at +8. An aerial
  detour and nine frames in action 363 precede regular cycling later in the
  response. This cannot be attributed to pending-prefix mismatch alone.

All twelve original responses retain valid input timing, complete 120-frame
observations, and zero prefix drift/errors. Do not turn behavioral failures
into transport-invalid exclusions.

## Next order

1. Make the suite's live/parsed action-frame contract explicit, retain a legacy
   override, and pin action 363 with trace regression coverage. Verify the
   full response; do not blindly subtract one from every action.
2. Before retraining, rerun the **same saved candidate** using validated teacher
   replays as handoff sources, with the same delay and cold history. Their first
   two recorded pending sends follow the training teacher. Name this a separate
   matched-teacher proof, not a silent replacement of original-source recovery.
   Confirm first-state and pending-command agreement before scoring learning.
3. Compare with the original-source challenge under the corrected convention.
   Preserve temperature 1 and chain-10 gating; any deterministic diagnostic
   must be separately reported.
4. If early targets remain underfit, explicitly increase their sampling mass
   or reduce canonical dominance in the next bounded experiment. Gate frozen
   per-case likelihood and early/off-loop accuracy, then 12/12 live responses.
   More undifferentiated epochs or architecture sweeps are not the immediate
   answer. Warmed and varied-input recovery gates still follow.

## Artifacts and validation

Authoritative outputs in `eval_runs/0913_per_case_fit/`:
- `candidate_v2.json`, `initial_v2.json`: correct pool, exactly 7,785 targets.
- `live_trace_v3.json`: pending inputs, first divergences, full action runs,
  normalization residuals, and loop-exit context for every response.
- `action_frame_probe.json`: actual Agent cold-embedding normalization A/B.
- `tests.log`: 14 passing guard/metric tests, including three new regressions.
- `sources.sha256`: final diagnostic sources and input checksums for reproduction.

Earlier `candidate.json` / `initial.json` are superseded: the first audit
incorrectly expert-tagged the canonical fixture, retaining two extra tail
targets. The drill treats it as recorded. The corrected audit asserts 7,785
targets; teacher-case conclusions are unchanged. `action_frame_probe.log` is
a failed prototype with destructive tensor transfer; the successful probe uses
non-destructive copies and has `action_frame_probe_v2.log`. Failed artifacts
remain visible and were not overwritten.

Reproduce without native rebuild or training:
`bash eval_runs/0913_per_case_fit/run.sh NEW_OUTPUT_DIRECTORY`.
The runner refuses existing output directories and an already-running BEAM.
