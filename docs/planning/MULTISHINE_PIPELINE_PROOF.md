# Multishine pipeline proof

## Current result: frozen local proof passes (2026-09-13)

[Same-initialization no-dropout fit](NO_DROPOUT_FIT_2026-09-13.md): all six
offline confidence gates pass, followed by12/12 valid matched-teacher local
responses with13–14-shine chains. The next step is longer/held-out local behavior
on the frozen checkpoint, not another training sweep. Canonical cold frame0
remains a known miss; general gameplay/recovery is not established.

## Historical checkpoints and blockers

Newest blocker: [random recurrent initialization and precision mismatch](EARLY_WINDOW_DIAGNOSIS_2026-09-13.md).
The three failed windows have no exact label conflict. Fix batch-dependent
initial state and define precision parity before further training.

Latest: [21-epoch targeted fitting result](EARLY_PREFIX_ROUND21_2026-09-13.md).
705/708 teacher matches; three early X decisions still fail the readiness gate.
No new live proof or promotion. Diagnose those exact windows before more training.

Prepared next: [targeted early-prefix fitting](EARLY_PREFIX_FITTING_2026-09-13.md).
Sampling and GPU preflight pass; no new bounded training run yet. Early teacher
windows now have an opt-in x64 repetition recipe and a frozen confidence gate.

Latest follow-up: [producer AF contract and matched teacher revalidation](MATCHED_HANDOFFS_2026-09-13.md).
Contracts now verify, but the same candidate still passes 7/12. Neutral starts
pass 4/4; early sustain and landing recovery need targeted fitting next.

Latest: [per-case learning and live failure diagnosis](PER_CASE_LEARNING_2026-09-13.md).
Snapshot CLI crash fixed. Before more training, correct the live action-frame
contract (including action 363) and distinguish teacher-consistent pending
handoffs from original-source recovery challenges.

Started 2026-09-13. Goal: one learned bot starts and sustains multishines,
recovers after interruptions, and retains those abilities across varied states.
This is a pipeline correctness demonstration, not a general fighting bot.

Current requested sequence:
- [x] Stress input timing: reproduced failure, patched retries, 12/12 reruns pass.
- [x] Test recovery targets: verified teacher delivery and actual delayed targets.
- [ ] Tiny overfit proof: preflight issues and required data path recorded below.

Preflight corrections completed 09-13: numerical guard, neutral-start coverage,
and recorded-teacher ingestion are now implemented and validated. The earlier
preflight blockers below are historical; see the latest progress entry.

## Ordered gates

Current authorized order: stress input timing, test recovery targets, then
tiny overfit proof. `scripts/stress_input_timing.sh NEW_OUTPUT_DIRECTORY`
runs twelve fresh reaction-2 launches: four each with blocking, 100 ms polling,
and aggressive 1 ms polling. Every run must have verified input timing,
zero prefix divergence/errors, and a passing control. Failures remain in the
summary and stop promotion to training. Queued frames and stream restart
behavior also need deterministic transport tests; this launch matrix alone
does not cover every same-process restart or paused-game condition.

- [ ] 1. Restore the known-good policy control and explain the failed delay-4
  control. Check actual timing, input history, and checkpoint conditioning;
  do not substitute whichever delay scores highest for a latency measurement.
- [ ] 2. Build a small trusted dataset from executed teacher recoveries and
  continuations. Delayed targets must follow the teacher's actual future.
- [ ] 3. Overfit one model at one validated delay; the same checkpoint must
  start, sustain, and recover in closed loop. Preserve an unchanged-label
  comparison and repair the false low-loss guard before training.
- [ ] 4. Expand position, facing, opponent context, and interruptions one
  dimension at a time, retaining the original cases as regression gates.

Do not move past a failed prerequisite. Record failures and excluded runs,
not just successful configurations. No architecture sweep or broad collection
before the small correctness proof.

## Gate 1 protocol

Checkpoint: `checkpoints/ms_g23a_ep57.bin`. First reproduce the historical
passing reaction-2 and reaction-3 controls, alongside the failing reaction-4
control. Fixed own-replay handoff at frame 900, temperature 1.0, two runs per
setting, automatic checkpoint delay-ID resolution, prefix history warm-up.
No action-frame conversion override. Require zero handoff divergence and a
chain of at least 10 in both runs for a passing 120-frame control. This is a
diagnostic gate, not a population performance estimate.

Then explain any cross-rung difference with recorded input timing and training
semantics before changing runtime conventions or starting a new training job.

## Progress

- Prior evidence: recovery HOLD labels disagree with executed teacher futures
  (18/21 off-loop at shift 4 versus 0/327 on-loop). See
  [recovery confirmation](RECOVERY_LABEL_CONFIRMATION.md).
- Gate 1 in progress. No training launched.

### 09-13: reproduced controls and measured delivery

Artifacts: `eval_runs/0913_policy_control/`. All runs are sampled T=1.0; delay
IDs are automatically derived and now recorded in `agent_runtime`.

| Reaction | Derived ID | Original warm-up chains | Committed-history chains |
|---|---:|---|---|
| 2 | 0 | 14, 14 | 14, 14 |
| 3 | 1 | 12, 14 | 14, 12 |
| 4 | 2 | 1, 1 | 1, 1 |

The historical positive control is reproducible; reaction 4 is **not** a
passing positive control for this checkpoint in this harness. Changing history
alone did not fix it. Do not claim the gate is complete from those chains alone.

Raw Slippi input comparisons find a second issue: the send-to-record latency
is sometimes 1 frame and sometimes 2, despite identical requested settings.
For example, committed reaction-3 runs have 117/117 transmitted decisions
matching at latency 4 and 5 respectively; sends match 120/120 at latency 1 and
2 respectively. Reaction-4 runs similarly match decision latencies 5 and 6.
Unsent tail decisions are excluded. These are input measurements, not estimates
from whichever chain looks best.

Disabling transport polling gave two correct-latency reaction-4 runs, both still
chain 1. Combining blocking transport with the legacy action-frame conversion
gave chains 1 and 3, both at the wrong measured latency. Neither is a proven fix.
Same-frame live/recorded action and grounded flags agree on all 120 response
frames in these traced runs, excluding a simple whole-state clock shift there.

Thus delivery variability is real but does not explain every reaction-4 failure.
The model's delay-conditioned behavior / remaining feature mismatch is still
unresolved. Do not retag checkpoints or change the latency table to fit scores.

Added a [default policy timing gate](../guides/SCENARIO_INPUT_TIMING.md) so wrong-
latency runs are withheld rather than scored as learning failures. A first
integration attempt hit a new nil/boolean check error; it was fixed and those
six errored runs (`ep57_verified_k2.json`) are not evidence. The standalone
verifier correctly accepts the recorded latency-4 reaction-3 run and rejects
the latency-5 run. Immediate post-shutdown reads initially found unfinished
recordings (`ep57_verified_fixed_k2.json`); the reader now retries briefly and
retains explicit errors if the recording remains unreadable.

The corrected live smoke (`ep57_verified_retry_k2.json`) successfully reads its
recording, detects a genuine delivery mismatch, withholds the behavioral score,
and exits with status 2. Six fresh gated controls in
`ep57_verified_final_k2.json` all pass: chain 14, zero drift/errors, and
correct measured input timing. This restores a timing-verified positive rung
but does not establish that transport timing is reliable across launches.

Next prerequisite: resolve delivery variability and confirm enough correctly
timed controls before moving to recovery data and training. Gates 2–4 remain
queued, not silently skipped. The intended minimal proof can use a single
validated rung; it does not require salvaging every legacy delay ID.

### 09-13: libmelee_ex queued-frame fix

The user authorized fixing the maintained sibling dependency directly.
`libmelee_ex/lib/melee/console.ex` now drains already completed frames without
flushing controllers. Previously each drain advanced controller input before
returning a stale observation. The regression test fails against the original
code and checks both pipe flushes and direct pad batches; an empty queue still
flushes before waiting for the next frame. Polling retries, game-start neutral
inputs, and rollback handling are unchanged.

The in-memory prototype (`ep57_no_prefetch_k2.json`) passes six runs: all chain
14, zero drift/errors, all recorded inputs at the expected latency. The source
patch is now in the sibling repository, uncommitted. Evaluations load it with
`-r ../libmelee_ex/lib/melee/console.ex`; shared compiled artifacts are untouched.
`run_library_control.sh` checks the source patch at additional delay settings.

The source-patched reaction-3 batch (`ep57_library_queue_k3.json`) also passes
all six runs: chain 14, zero drift/errors, expected latency. Total patched
validation is 12/12 clean controls across reactions 2 and 3 and handoffs
900/1500/2100. No BEAM evaluation remains running. Library suite: 119 doctests,
3 properties, 565 tests, zero failures, 71 live-Dolphin tests excluded. Exphil
timing/history/recovery/scoring/rung tests: 49 passing. Diff and shell syntax
checks pass in both applicable repositories.

The original console also passed a separate six-run batch. These small samples
do not prove that the queued-frame bug explains all intermittent extra-frame
delivery. Keep the timing gate enabled; do not declare the transport universally
calibrated or change training shifts on the strength of chain length.

### 09-13: adversarial timing stress

First matrix: `eval_runs/0913_timing_stress/`. Four blocking and four 100 ms
polling launches pass with chain 14; all four 1 ms polling launches fail the
recorded-input timing gate (raw chain 1, scores withheld). The first two
scoreboards predate console-timeout metadata; their filenames identify their
settings. Do not discard this failed matrix or count it as model regression.

Two more defects were reproduced and patched:

- The timing verifier accepted traces with missing intermediate frames. It
  now requires an ordered contiguous trace; regression also covers long varied
  streams and late segments across delays 0–8.
- `MeleePort` retried nil console results by committing inputs again, despite
  no new observation or decision. New sibling `Console.step/3` supports
  `flush: false`; internal bridge retries now wait without recommitting.
  The bridge regression fails against the old compiled bridge and passes
  against patched source. This does not change independent external
  `poll: true` calls; that runner needs separate qualification.

Added transport tests for 24 timeout/late-frame cycles, 20 game restarts with
neutral-input reset, and 25 wait-only retries without extra pipe or direct
commits. The retry-fixed matrix in
`eval_runs/0913_timing_stress_retry_fix/` passes **12/12**: all chain 14, all
commands at expected timing, zero drift/errors. Source hashes are retained.
The aggressive condition changes from 0/4 to 4/4 passing. Library suite:
119 doctests, 3 properties, 568 tests, zero failures, 71 live-Dolphin tests
excluded. The session test double was updated for the extended console call.

### 09-13: recovery targets checked through the actual training label path

`scripts/check_recovery_targets.exs` independently checks the teacher's issued
commands against its Slippi recording, then runs `Labels.at_delay` on the
recorded teacher segment and checks each target against the teacher's actual
future command. This does not use `label_ahead` to grade `label_ahead`.

All three grounded recovery recordings pass native teacher delivery checks
(360/360 commands). At reaction delays 2/3/4/5, all 354/351/348/345 training
targets match, including 21 off-loop targets per delay. Reports:
`eval_runs/0913_recovery_labels/targets_verified_k*.json`. These are overlapping
correlated windows, not independent episodes. Delays 4/5 here certify label
pairing, not a passing policy control at those delays.

`test/fixtures/recovery_targets.json` preserves the first 14 frames of each
recording with its replay hash, independently recorded controllers, and issued
teacher commands. The regression requires correct recovery targets at delays
2–5 and explicitly detects wrong held-current-command targets.

### Tiny overfit preflight: not launched

The next training experiment is not yet ready to execute safely:

1. Export only verified teacher-controlled windows as **recorded** frame lists,
   preserving their boundaries. Do not include their scripted replay prefixes.
2. Add a recorded-frame ingestion path to the drill or use a trainer that
   already preserves that provenance. The drill's current `--snippet-frames`
   path intentionally retags every list as expert-relabeled; feeding these
   recordings through it would undo the corrected-future semantics.
3. Resolve the still-active low-loss collapse guard in `dagger_drill.exs`
   before an overfit test; it still rejects loss below 1e-5 or a 100x drop.
   Do not silently bypass it or claim low loss alone proves recovery.
4. Include a verified neutral-start example as well as sustain/recovery
   windows, then train one delay-2 model with fixed budget and evaluate the
   same checkpoint on all three behaviors. Keep a pre-training control and
   untouched-label comparison, save snapshots, and retain timing gates.

No new training or dataset promotion was launched in this session. Timing
stress and recovery-target checks are complete for this scoped synchronous
harness; the tiny overfit run remains queued behind these explicit preflights.

### 09-13: all three training preflight corrections complete

- Replaced loss-scale collapse rejection with `EpochHealth`: finite numeric
  loss and finite, nonempty parameter tensors. Healthy losses below 1e-5 and
  large improvements are no longer rolled back. Invalid numerical epochs
  retain the bounded restore/halt behavior and cannot become best candidates.
  This is not a replacement for live behavioral gates.
- Added two standing-start cases at ep38 frames 2228/2566. Both start grounded
  in action 14, reproduce without drift, and teacher-chain 14. Their input
  timing and delayed targets validate independently against recordings. Added
  a curated neutral-start fixture and regression at delays 2–5.
- Added `--recorded-frames` with strict unshifted causal provenance, preserved
  boundaries, no previous-action overrides, content-sensitive resume identity,
  and explicit rejection of unsupported streaming ingestion. No expert retagging.
- Exported six verified windows: 720 frames / 708 delay-2 targets covering
  neutral starts, recovery, and sustain. Final exports are under
  `eval_runs/0913_teacher_ingestion/validated/`. Initial exports exposed unknown
  metadata atoms during safe decoding; final envelopes store validation metadata
  as JSON and passed actual fresh-VM ingestion.
- Full GPU preflight (`preflight_final.log`) passes: 7,079 canonical fixture
  frames + 720 recorded frames, embeddings, an optimizer step, numerical guard,
  policy export, and trainer serialization. Exit 0; 17 seconds after JIT starts,
  peak RSS 2 GB. The inhibitor's shutdown child prints an exit-1 message after
  completion; the preflight process itself exits successfully.

Usage and constraints: [recorded teacher training](../guides/RECORDED_TEACHER_TRAINING.md).
The next step is the bounded tiny-overfit experiment and evaluation of the same
checkpoint on all three behaviors. No long training or new bot promotion occurred.

### 09-13: tiny overfit executed; behavioral proof failed

- Completed the predeclared 40-epoch delay-2 run. Best loss approximately
  0.0001 does not establish behavioral success. Sustain chains 14 twice, but
  only 4/11 timing-valid runs pass chain 10; one other run is timing-invalid.
  Starts and two recovery handoffs fail. No checkpoint promotion or sweep.
- Actual lazy-batch provenance audit finds 90 cross-clip windows. All 29
  off-loop teacher targets identified by that audit have wrong prior-clip
  context. Loading/shifting preserved label boundaries, not recurrent history.
- Deferred the prepared old-label comparator because it shares this defect.
- Next: boundary-safe temporal histories retaining early targets, an emitted-
  window regression, and a separately recorded repeat of the bounded proof.
  Do not merely filter crossing windows and lose start/recovery supervision.

Full protocol, artifacts, and results: [tiny overfit](TINY_OVERFIT_2026-09-13.md).

### 09-13: clip-boundary history correction

The eager drill now retains clip/gap boundaries in lazy temporal windows and
previous-action queues. Repeat-first cold-start padding preserves early targets
without borrowing another clip's history. The real-data audit now has 7,785
sequences, zero cross-clip windows, and all 29 audited off-loop targets retained.
Weight/teacher/probe indexing follows the new supervised-frame mapping; resume
fingerprints version the change. Flat callers/streaming shards are not certified
by this correction. Next: resolve input-verification mismatches, then repeat the
bounded behavioral proof; padding is not reconstructed replay-prefix context.

### 09-13: remaining input/history discrepancies diagnosed

All 2,880 recorded sent button patterns match at +1 frame. Analog comparison,
not demonstrated extra latency, accounts for the flagged runs: the corrected
candidate's one mismatch is a digital R click recording right trigger 1. A full
analog contract still needs verification before changing the gate.
Cold-start training also differs from warmed-prefix evaluation for early
windows/action queues; the delayed committed-history clock is consistent.
See [diagnosis and next order](INPUT_AND_HISTORY_DIAGNOSIS_2026-09-13.md).

### 09-13: input/readback contract and cold-history mode validated

Fixed a real pipe-trigger scaling bug in sibling libmelee_ex. Versioned
verification now handles measured stick readback and independent L/R triggers;
549/549 commands pass each live old/new sweep. Historical initial/corrected
recordings both revalidate 12/12 under the explicit legacy profile.
Matched `--prefix-history cold` passes a two-handoff untrained mechanics smoke.
No retraining yet. Next: a new fixed-budget, boundary-safe, cold-history proof,
followed separately by warmed-history recovery. [Contract](../guides/INPUT_READBACK_CONTRACT.md).

### 09-13: boundary-safe cold-history run executed; not yet a proof

The fixed 40-epoch candidate passes 7/12 handoffs; all 12 pass timing/drift/full-
response checks. Starts and recovery remain unreliable. No promotion or budget
extension. A new metric defect is confirmed: the drill logs/selects/stops on
the final minibatch's loss rather than aggregate epoch loss. Tiny printed loss
does not prove the pool was fit. Next: correct aggregation and measure fixed
full-pool/per-handoff teacher-forced fit before another bounded run.
[Protocol and results](TINY_OVERFIT_COLD_2026-09-13.md).

### 09-13: aggregate training metric corrected

The drill now accumulates denominator-weighted losses over every batch, in both
normal and forensics paths. Checkpoint selection/stopping no longer follow the
last minibatch. Eleven metric/health tests pass; resume semantics are versioned.
This running training objective still does not measure a frozen checkpoint's
per-case fit. Next: fixed-checkpoint teacher-forced diagnostics, then repeat the
scoped live proof and separately qualify continuous/warmed recovery.
[Correction and remaining work](EPOCH_METRIC_CORRECTION_2026-09-13.md).
