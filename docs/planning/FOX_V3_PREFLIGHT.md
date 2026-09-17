# Fox V3 preflight — 2026-09-15

Status: **Expanded preflight in progress; do not launch the long run.** The
original NO-GO findings below are retained as history. User authorized the V3-specific preflight, not the full
approximately 48-hour generalist training run. Mewtwo work is parked; its handoff
remains unchanged. Evidence goes in `eval_runs/0915_fox_v3_preflight/`.

## Scope and frozen initial protocol

Exercise the generalist's actual GRU BPTT training/export/stateful-inference path.
The short-window multishine result does not certify this different execution path.
Preserve the earlier V3 recipe: fresh initialization, causal labels, reaction 0,
GRU hidden 512 / two recurrent layers, autoregressive head, F32, dropout 0.1,
unroll 80, overlap 1, batch 128, streamed replay chunks of 200, stage internals,
player-style conditioning and clean loss weights. No legacy-model warm start,
specialist-data mixing, action-frame buckets or new architecture in this check.
Confirm resolved model dimensions rather than inferring them from CLI names.

1. Run focused label, BPTT carry/reset, export and live-stateful contract tests.
2. Freeze a manageable whole-game subset of the general Fox corpus, preserving
   names, subject-port selection, explicit ditto tie-breaks, stage coverage, source hashes
   and separate validation games. Audit the actual resolved split.
3. Run a small two-epoch trial, initially 256 files including 16 validation files.
   Use seed 905; a second seed 906 uses the same files and split if the first
   passes the mechanical gates. Neither tiny model is expected to be a good bot.
4. Check trained export/reload and full-sequence versus streaming inference,
   recurrent boundary resets, autoregressive conditioning and metadata. Compare
   logits on identical inputs, not sampled actions from independent RNG draws.
5. Run fresh-process held-out evaluation and sampled local gameplay at temperature
   1, reaction delay 0, stateful step enabled. Include both bot ports, more than
   one stage/opponent, and one ordinary-speed graphical run. Record latency,
   errors, actual elapsed time, move/idle/shield/death behavior and replays.
6. Estimate full-run cost from observed batches, parse/embed overhead, validation
   and checkpoint time. Use the historical roughly six-hour V2 epochs only as a
   prior. Freeze source/dependency revisions and a reproducible launch command.

## Decision rules

- Mechanical GO requires correct causal data and subjects; finite optimizer
  updates; consistent recurrent carry, export and sequential head semantics;
  complete held-out coverage; functioning sampled stateful gameplay at measured
  latency 1 and ordinary local speed; explicit checkpoint and source provenance.
- A newly found contract bug is a NO-GO until fixed and the affected check rerun.
  Preserve the failure and explain any resulting protocol changes before fitting.
- Small-model behavior is a diagnostic. Catastrophic collapse requires diagnosis;
  low win rate after two small-data epochs alone is not proof of a broken harness.
- Do not require multishining from a general Fox model or promote it on training
  loss. Do not silently switch to windowed or deterministic inference to pass.
- Completion report separates mechanical readiness, observed behavior and limits.
  The full V3 run is a subsequent action, not automatically launched by this file.

## Resume log

- Initial process/GPU check: no existing training or Dolphin process was active.
- Working tree contains earlier Fox/Mewtwo changes; preserve them. No reset or
  blanket cleanup. Use the current sibling dependencies and record their state.
- Initial contracts: 57 tests, 0 failures, 7 excluded; selected native label/eval
  checks also passed. Seed regression failed before fixing both Axon parameter
  initialization and inter-layer dropout RNG state; now passes.
- Added BPTT-specific export contract requiring stateful F32 GRU inference.
  The affected contract suite passed 37 tests. Legacy unstamped exports remain
  compatible. Newly stamped BPTT exports reject windowed playback.
- Found live cold-start padding repeated the first frame even for BPTT. Removed
  padding for BPTT decision and observation paths, and rejected window resync /
  LEACE fallback for BPTT. Startup/reset regression passed: 8 agent tests.
- Fixed subset: 240 train / 16 validation, all six stages; SHA256 source manifest
  in `corpus.json`. Filename-screened Fox candidates then metadata-verified;
  this is a bounded diagnostic subset, not a representative full-corpus estimate.
- Seed 905 two-epoch CUDA trial launched with the saved exact args and the
  existing Elixir launcher (GPU memory fraction 0.15). No full run launched.

## Result: do not start the long run yet

The multishine specialist exercises a different windowed path. The generalist's
BPTT path still has independent problems. Three localized fixes landed in the
working trees during this preflight:

1. Explicit training seeds now reach Axon parameter initialization and Edifice's
   carried-GRU dropout initial state. Same-seed initialization matches; a
   different seed changes it. This does not claim bitwise reproducible GPU fits.
2. New BPTT exports carry a `bptt_gru_f32_v1` execution contract and require
   stateful inference. Legacy unstamped exports retain compatibility.
3. Live BPTT startup consumes the first frame once from zero carry, including
   after reset. It no longer repeats it to fill a legacy window. BPTT rejects
   window resync and LEACE fallback options that would violate this contract.

### Unresolved training blockers

The executable reproduction is `eval_runs/0915_fox_v3_preflight/loader_audit.exs`;
its results are in `loader_audit.json`. Run it with:

```sh
devenv shell -- env EXPHIL_GPU=0 mix run eval_runs/0915_fox_v3_preflight/loader_audit.exs
```

1. **Repeated frame with advanced carry.** `TrajectoryCursors` advances by
   `unroll - overlap`, but the model returns carry after the entire unroll.
   With the saved 80/1 recipe, frame 79 ends one batch and starts the next with
   `is_resetting = 0`. Training therefore consumes it twice; ordinary live play
   consumes it once. Labels are already aligned by the data pipeline. For this
   implementation, use contiguous, nonoverlapping input batches (overlap 0),
   or deliberately return carry from before the overlap. Do not merely change
   a test to accept the repeated frame. Add an end-to-end fixed-weight sequence
   equivalence check across the actual loader boundaries.
2. **Premature exhaustion drops long game tails.** When any cursor needs a new
   segment and the queue is empty, the whole stream stops. On synthetic games
   of 80 and 8,000 frames with batch size 2, it scores 160 unique frames and
   drops 7,920, for either overlap 0 or 1. The module's documented bound of
   fewer than `batch_size * unroll` lost frames is false. This can bias training
   toward earlier portions of games; the same cursor loader is used for training
   validation. The 98% loss here is a counterexample, **not a measured corpus
   loss rate**. Fix with explicit valid-frame masks / padded inactive rows, or
   another approach that processes all active rows to completion. Normalize
   loss and metrics over valid frames. Never carry state across a real segment
   boundary or score padding as neutral controller targets.
3. **Small final file chunks cannot fill the batch.** `chunk_files/2` makes
   `[200, 40]` for the preflight's 240 training files. The cursor loader rejects
   40 ordinary segments for batch size 128. A full corpus may encounter the
   same problem depending on its remainder and segment counts. Correct the
   generic final-chunk handling (and parse-error cases), rather than selecting
   an artificially convenient dataset size to make this preflight pass.
4. **Configured dropout is inactive during gradient computation.**
   `Imitation.new/1` builds `predict_fn` with `mode: :inference` and supplies it
   to both BPTT loss/gradient and evaluation builders. The seed fix controls
   initialization but does not activate dropout. Wire a separate training-mode
   forward and propagate its updated Axon model state, retaining an inference
   forward for evaluation. Test fresh training masks, reproducible seeded
   initialization, stochastic train behavior and deterministic evaluation.
   Alternatively explicitly choose and document dropout 0 as a methodology
   change; do not claim that the current 0.1 flag supplies regularization.
5. **Interrupt checkpoint recovery failed in the real launch.** Stopping this
   trial with SIGTERM triggered a `System.SignalHandler` match error because
   the trap callback returned nil. The VM then shut down during preparation;
   no interrupt checkpoint was saved. Fix the callback return/lifecycle and
   prove interruption plus resume in a subprocess before trusting a two-day
   training run. The subsequent EXLA cache/process errors in this log followed
   shutdown; they are not evidence of a spontaneous training failure.

### What actually ran

- Final focused contract suite: **65 tests, 0 failures**, including native
  label alignment, BPTT, checkpoint and agent checks (`final_contracts.log`).
  Existing cursor tests encode its old overlap/exhaustion behavior; their
  passing status does not clear the separately reproduced loader failures.
- Replay selection/metadata, style registry and held-out preparation completed.
  The chosen subset has 79 Fox subjects on port 1 and 177 on other ports,
  zero dittos, and 44 training filename tags. Ditto behavior remains a separate
  test case; this subset does not provide a live ditto demonstration.
- The resolved model initialized with approximately **2,901K parameters**.
- The trial started at 19:12:12 local time on September 15 and was deliberately
  stopped at 19:13:58 while preparing the first 200-file chunk, after the audit
  exposed the loader failures. No optimizer batch or epoch completion was
  recorded, and no trained/exported candidate was produced.
- Thus there is **no new training-throughput estimate**, trained export parity,
  complete held-out score, seed-906 comparison or Dolphin gameplay result.
  `parity.exs` is a prepared, unexecuted follow-up script, not passing evidence.
  Neither seed's saved args should be used as a production launch command.
- The earlier six-hours-per-epoch / roughly 48-hour run estimate remains a
  historical prior only. Correct loader coverage may change that cost.

### Resume order

1. Fix and regression-test the five blockers above. Add exact frame-accounting
   checks (including short segments, long/short game mixtures, tails and gaps).
2. Document the corrected recipe, including overlap/dropout decisions, and
   retain the original failed attempt's logs. Do not overwrite its evidence.
3. Restart the bounded fit in a new attempt directory; freeze source revisions
   and dirty diffs, data manifest, exact args and selected checkpoint hashes.
4. Complete trained full-sequence / sequential-head / export parity, fresh-process
   full-coverage validation, second-seed run, and sampled Dolphin checks from
   the initial protocol. Check default anonymous style as well as any selected
   trained style; keep this distinct from teacher-forced validation.
5. Only then issue a new GO/NO-GO and updated full-run duration estimate.

Mewtwo's handoff was not changed. No full training, promotion, commit or push
was performed as part of this preflight.

## Repair pass — September 15, later session

- Loader now requires overlap 0 and retains every frame, including short
  segments, partial tails and file chunks smaller than the batch size. Invalid
  slots repeat a safe input with zero weight; carry resets before another game.
- CLI/default overlap is 0; an explicit nonzero value fails validation early.
- Training and validation aggregate by loss weight, including partial batches;
  entropy regularization also excludes zero-weight padding.
- BPTT gradients use train-mode forward, propagate updated Axon dropout state,
  and keep a separate inference forward for validation and play.
- SIGTERM coordinates with the training owner: the default Erlang shutdown
  handler waits until checkpoint/export cleanup completes. A batch halt now
  stops the outer fit rather than starting the next epoch.
- Focused suite: **70 tests, 0 failures**, `repairs/full_contracts.log`.
  This includes complete frame accounting, weighted padded validation versus
  full replay forwards, dropout state updates, exact next-update resume and
  stopping/cleanup semantics.
- Real SIGTERM probe: `repairs/signal_probe_after.log` and `signal_probe.json`.
  Saved step 1 and reproduced step 2 exactly before coordinated VM shutdown.
  The earlier `signal_probe.log` captures why returning `:ok` alone was
  insufficient: Erlang's default handler still stopped EXLA before resume.
- Corrected two-epoch CUDA run launched in `repaired_seed_905/`, preserving the
  original failed attempt. Same 240/16 replay split, model and optimizer recipe;
  overlap changed to 0 and configured dropout 0.1 is now actually active.
- Full-training GO still requires the real trial and remaining deployment gates.
- Full-size attempt `repaired_seed_905` completed both chunks' preparation
  (1,878,999 and 378,517 frames), then exhausted the launcher's 0.15 GPU-memory
  allocation on its first gradient step. Retrying in `repaired_seed_905_gpu70`
  with allocation 0.70; all model/batch/data/dropout settings are unchanged.
  The dedicated Elixir launcher and failed log are retained under this experiment.

### Completed repair validation

The 0.70 allocation run completed at 19:48:30 local time, exit status 0 from
the launcher, with 1,084 optimizer updates (542 per epoch):

| Epoch | Training loss | Held-out loss | Reported epoch seconds |
| --- | ---: | ---: | ---: |
| 1 | 4.5326 | 4.9834 | 249 |
| 2 | 3.8528 | 4.4590 | 172 |

The successful run took about 7.5 minutes including initialization, validation
and exports. Each epoch loaded 2,257,516 training frames across the same
200/40-file chunks. The second epoch needs more optimizer batches than the old
loader; the old progress estimate undercounts padded batches and is not an ETA.
The final and best policies are diagnostic models, not promoted generalist bots.

Final policy: `repaired_seed_905_gpu70/model_policy.bin`, SHA256
`c93b2fc82c14d0991fd7eed314d7cdbd4096aa07471770a5ea23cf096ce410f8`.
Exported parameter data exactly matches `model.axon`.

An existing `sh failed with exit status 1` message appears after the success
summary, also present in the earlier successful multishine logs. The actual
launcher returned 0. Its emitting helper has not been identified; this message
is not being used to override the observed exit status or completed artifacts.

### Additional finding: GPU arithmetic must be explicit

Float32 tensors alone do not imply full-precision dot products in EXLA.
With default arithmetic, trained sequential/sequence logits differed by up to
0.002848, while carry differed by less than 5e-7. Keeping the same parameters,
inputs and tolerance and setting EXLA `precision: :highest` reduced the largest
logit discrepancy to 2.15e-6, below the unchanged 1e-4 threshold. Chunked/full
sequence logits and carry also passed. See `parity_default.json` and
`parity_highest.json`; the first failure was preserved.

`EXPHIL_EXLA_PRECISION=highest` is now a supported opt-in configuration for
normal Mix training/evaluation/play commands. It applies to all processes in
that VM. Use it consistently in the remaining V3 checks and future launch.
Existing runs without the variable retain their previous arithmetic. This
setting is separate from tensor `--precision f32` and is not inferred from a
checkpoint. The completed two-epoch fit used default arithmetic; **a full-size
highest-precision training/timing check remains necessary before a long run**.
Do not present the diagnostic highest-precision inference check as proof that
the completed fit trained with that setting.

The normal Mix path with `EXPHIL_EXLA_PRECISION=highest` was independently
verified in a fresh process: `parity_configured.log` / `parity.json`, exit 0,
maximum logit difference 2.15e-6. No special diagnostic override was used in
that run. Machine-readable repair summary: `repairs/completion.json`.

Still pending for the original full preflight: that arithmetic training check,
second-seed trial, fresh-process complete held-out scoring, and sampled live
Dolphin checks (including both ports and ordinary graphical speed). No full V3
run has been launched, and the earlier historical 48-hour estimate has not been
replaced with a reliable full-corpus estimate.

## Expanded preflight — September 15, evening

The user will have **Claude Fable run the long training job**. This session is
authorized to finish bounded preflight trials and add regression tests, not to
launch the eight-epoch full-corpus fit. Preserve this document as the handoff.

Additional audit findings and repairs:

- New streaming style registries now reserve ID 0 for anonymous/unknown players.
  Previously `from_tags/1` assigned the first named player ID 0 too, contradicting
  the pipeline's anonymous-style contract. New registries use `first_id: 1` and
  version-2 JSON; old version-1 files retain their original mapping. There are
  111 named slots within the unchanged 112-channel embedding. Unknown/overflow
  tags fall back to anonymous zero; no separate hashed overflow bucket is used.
- BPTT rejects gradient accumulation: the accumulated trainer branch uses the
  windowed gradient interface and cannot carry recurrent state correctly.
- Resumed fits initialize callback step counters from the restored trainer step.
  Resume still starts at a fresh data/epoch boundary; it is not exact mid-replay
  cursor/carry recovery.
- Empty epochs now fail instead of becoming zero-loss successes. The ordinary
  batcher's empty/insufficient-delay input also avoids Elixir's descending
  `0..-1` range.
- Required checkpoint/policy/config write failures propagate instead of being
  swallowed as warnings. A failed temporary checkpoint write preserves the
  previously published checkpoint, covered with an actual filesystem failure.
- Training's completion summary distinguishes batch-boundary interruption from
  completed epochs and does not invent a zero loss for an uncompleted epoch.

Expanded suite: **117 tests, zero failures** in
`repairs/expanded_contracts_fixed.log`. New tests include 81 loader combinations
of unroll/width/seed, valid-frame accounting and reset laws, masked input/target
invariance of the optimizer update, future-input causality, registry persistence
and capacity, accumulation rejection, resumed step numbering, real NaN data,
empty-epoch rejection and failed-checkpoint preservation. The initial run's two
failures are preserved in `expanded_contracts.log`: one found the empty-range
bug; the other was a test helper assuming a two-level parameter map.

`highest_seed_905/` was deliberately interrupted after the registry collision
was found. It completed one highest-precision optimizer update and saved
`model_interrupt.axon`, `model.axon`, and `model_policy.bin` before exit 0.
This is full-size interrupt evidence, **not a completed epoch or seed trial**.
Its old completion log says zero loss; the new summary fixes that misleading
display. It must not be promoted or compared as a completed fit.

Fresh matched attempts: `verified_seed_905/` and `verified_seed_906/`, same frozen
240/16 subset, corrected registry, overlap 0, active dropout 0.1, F32 tensors,
highest EXLA arithmetic, GPU allocation 0.70. Exact args are in each directory.
Run sequentially. `heldout.exs` checks every held-out file's source hash and frame
coverage and reports anonymous and training-registry conditioning separately.
`full_corpus_audit.exs` inventories the eventual full corpus; it does not train.

### Remaining gates for the Fable handoff

| Gate | Required evidence |
| --- | --- |
| Matched seeds | Both corrected two-epoch fits, finite losses, actual elapsed time and artifact hashes |
| Export / carry / AR | `parity.exs` passes at unchanged 1e-4 tolerance under highest arithmetic |
| Held-out evaluation | All 16 source hashes, all parsed causal frames, anonymous and registry scores |
| Live deployment | Temperature 1, stateful, local reaction 0 / measured latency 1; both ports, multiple stages/opponents, ordinary graphical speed |
| Behavior | Score only the intended play interval, excluding forced SD/LRAS replay-finalization tail; record idle/shield/movement/deaths |
| Corpus / provenance | Full-corpus metadata/hash inventory, no train/validation byte-identical overlap; source/dependency heads and dirty changes |
| Long-job handoff | Exact unlaunched command, resource/timing estimate, checkpoint recovery procedure, remaining limits and GO/NO-GO |

Passing these gates demonstrates the exercised paths, not every possible failure
or competent generalist play. Hardware/power failure, distribution shift,
replay near-duplicates, and long-horizon learning quality remain distinct risks.
