# Producer action-frame contract and matched-teacher revalidation

## Protocol

- [x] Inspect the producer and historical action-frame caveats before changing defaults.
- [x] Implement character-aware inversion of the actual libmelee indexing rule.
- [x] Validate all 1,440 saved observations and pin a compact regression fixture.
- [x] Validate scalar/batched embedding parity and retain generic Agent defaults.
- [x] Build six matched handoffs from hash-verified executed teacher recordings.
- [x] Evaluate the same candidate, two runs per handoff; no retraining or reselection.
- [x] Verify pending inputs, initial states, timing, and action-frame conversion.
- [x] Record every result against the predeclared chain-10 gate.

Candidate remains `eval_runs/0913_tiny_overfit_cold/candidate.bin`. Six cases
in order: neutral 2228, neutral 2566, sustain 900, recovery 4, recovery 75,
recovery 146. Two temperature-1 responses each, 120 frames, reaction delay 2,
delay ID 2, cold history. The source replay is now the validated teacher replay
for that case so its two pending sends follow the training teacher trajectory.
This is a separately named matched-teacher proof, not a replacement of the
original-source recovery benchmark. Chain >=10 is required in every response,
in addition to valid timing, complete observations, and zero drift/errors.

## Contract correction

The previous recommendation to extend the historical action-only table was
too broad. That table was explicitly disproven for varied play; it must not
become a global normalization default.

The current producer provides a better contract: `Melee.Events` truncates
the raw Slippi animation age and adds one iff
`Melee.FrameData.zero_indexed?(character, action)`. The new
`ActionFrameConvention.libmelee_to_parsed/3` reverses that exact indexing
adjustment, including Fox action 363, using the producer's own predicate.
It does not modify the old measurement table or claim it became valid.
Fractional precision lost to producer truncation cannot be recovered by this
inversion; it is an indexing contract, not a universal raw-float reconstruction.

The suite explicitly uses `af_convention: :libmelee` by default. `--live-af`
selects the same contract; `--no-live-af` retains the historical unconverted
policy behavior for named comparisons. Teacher decisions use the same
producer inversion. Generic Agent and general CLI defaults remain `:parsed`;
the old `:live` table remains legacy-only. Both scalar and batched player
embeddings support the producer-specific mode.

All 1,440 original candidate response observations match parsed replay action
frames with the producer inversion, including the 39 reflector-hold samples
the old table missed. This validates this workload, not every possible action
in arbitrary play. The regression additionally inverts the producer rule
across 33 characters, 399 actions, and negative/zero/positive ages.

## Artifacts

Directory: `eval_runs/0913_matched_handoffs/`.
- `manifest.json`: six teacher sources, each checked against validation SHA-256.
- `run_eval.sh`: sequential live protocol with no native rebuild; refuses an
  existing result or an already-running BEAM.
- `original_trace_corrected.json`: zero remaining action-frame mismatches on
  the original 12 responses.
- `tests_full.log`: 12 passing tests including five new regressions, scalar/
  batch parity, and historical default guards.
- `test/fixtures/statestream/libmelee_action_frame_pairs.json`: deduplicated
  observed pairs with counts and the source scoreboard checksum.

No inference about the behavioral contribution of AF normalization alone can
be drawn from this joint change to normalization and pending handoff sources.
No model promotion or new training is implied by a positive subset.

## Results

**7/12 pass, not the required 12/12.** All responses are complete, timing-valid,
and free of prefix drift/errors. Runtime records `af_convention: libmelee`,
reaction 2 and delay ID 2. All twelve have zero mismatches in the two pending
sends and matching measured initial player states. All **1,440/1,440** new
live/replay action-frame observations match the producer contract.

| Case | Previous original-source chains | Corrected matched-teacher chains |
|---|---|---|
| Neutral 2228 | 1, 1 | **14, 14** |
| Neutral 2566 | 13, 1 | **14, 14** |
| Sustain 900 | 13, 14 | 8, 6 |
| Recovery 4 | 12, 6 | **13, 13** |
| Recovery 75 | 7, 13 | 1, 7 |
| Recovery 146 | 13, 13 | **13**, 1 |

The overall pass count is unchanged; the case distribution differs. This is
two sampled runs per case, not a statistically powered improvement claim.
All four neutral trajectories now match the teacher's measured player states
throughout the response. This resolves the immediate neutral-start diagnosis
for these trials, not arbitrary starts or general recovery.

The remaining misses start with incorrect/uncertain cold decisions:
- Sustain: both first decisions differ from the B target (X / no buttons),
  followed by state divergence at +3/+4. Both eventually recover but lose
  enough of the response to miss chain 10.
- Landing recovery 75: both diverge in state at +4; chains 1 and 7.
- Landing recovery 146: one succeeds, the other misses the B target at +1,
  diverges at +4, and spends the response in repeated aerial detours.

These match the earlier frozen audit's weakness on the first four sustain
targets and the early B/X landing targets. Once trajectories differ, later
teacher-command differences are not automatically incorrect actions for the
new state. No evidence here requires additional input-delay changes.

The suite completed normally; `run_eval.sh` exits 1 because its stricter
chain-10 behavioral gate fails. The suite's own generic `pass` flag is not
the tiny-overfit gate. No failed response was excluded or rerun for selection.

Additional artifacts: `eval_candidate.json`, `matched_trace.json`, and
`af_audit.json`. The last report has 1,440 observations and zero mismatches.

## Next

Rebalance/oversample the first few cold sustain and landing-recovery targets
in the next bounded proof, rather than simply extending a canonical-dominated
run. Require frozen per-case likelihood and early-target fitting before live
gating. Keep this matched-teacher gate fixed and retain the original-source
pending-queue challenge as a separate recovery evaluation. Do not promote the
candidate or broaden the architecture sweep from 7/12.
