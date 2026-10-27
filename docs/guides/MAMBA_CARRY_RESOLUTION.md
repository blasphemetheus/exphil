# Mamba carried-state investigation

**AI-generated — 2026-10-27.** This note accompanies
`scripts/resolve_mamba_carry.exs`, `scripts/bench_mamba_precision.exs`, and
`scripts/lib/mamba_precision_experiment.exs`.

## Finding

### SSM and gating isolation follow-up

**AI-generated.** Promoting the SSM update, its output calculation, gating, or
carried SSM memory to f64 did not clear the numerical gate. Frame-level traces
show that much of the remaining error already exists in each operation's
inputs. Learned projections and gates amplify those small differences. This is
evidence of error propagation, not evidence of a single incorrect SSM equation.
The precise earliest source across the whole replay is not yet isolated.

These runs reuse the saved 512-frame, eight-row staggered captures. The Mewtwo
capture with Fox observations is still a stress test. Its baseline is
`dense_norm` (dense calculations and normalization in f64, rounded back to f32).
Counts below are output values exceeding the unchanged reference bound.

| Extra precision boundary | Batch reference failures | Single reference failures |
| --- | ---: | ---: |
| None: previous `dense_norm` control | 232 | 232 |
| SSM update only | 197 | 196 |
| SSM output sum only | 219 | 219 |
| Entire SSM, carry stored in f32 | 193 | 193 |
| Gate activation and multiplication, then round to f32 | 232 | 232 |
| Keep gate result in f64 through output projection | 233 | 233 |
| Entire SSM, carry stored in f64 | 188 | 188 |

Fewer failures do not imply every error improved. For example, the full-SSM
variant's worst error/bound ratio rises from about 16.7 to 19.2 on this stress
capture. Compiler fusion can also change when precision boundaries change;
these are end-to-end interventions, not a promise that every other f32 rounding
decision is held fixed.

The actual Fox-policy CUDA capture uses the cheaper `chunk8` control. Its
convolution-history reference failures change from **15 batch / 12 single** to
**9 / 6** with the full SSM in f64. Keeping the output path in f64 gives **12 /
12**. Storing SSM carry in f64 gives **9 / 15**. All of these still fail. Output
features and final SSM state pass on this Fox capture. No precision API or
production default was changed, and these runs make no speed claim.

#### Where the error grows

The new trace runner selects the largest error relative to its bound, replays to the state just
before that frame, and exports the layer intermediates. It then recomputes each
operation in f64 using exactly that operation's f32 inputs, promoted without
changing their values. This separates two quantities:

- **Local error:** the operation's result minus that f64 recomputation.
- **Propagated error:** that recomputation minus the full f64 trajectory result.

These signed errors sum to the total error at the same coordinate. Separate
maximum errors can occur at different coordinates and must not be added.

On the Fox capture, row 3, tick 264, layer 2, channel 166, the input projection
that feeds convolution history differs from the reference by about
**−0.00019347**. Its own calculation contributes **+0.000000225**; its inputs
contribute **−0.00019370**. Improving that projection alone cannot remove an
error that it receives from the previous layer.

On the Mewtwo stress capture, row 4, tick 184, the second layer's worst gated
coordinate has **+0.00092526** total error. Local multiplication contributes
about **−0.0000000263**; its inputs contribute **+0.00092529**. The final worst
output channel is 148: **0.20071316** versus reference **0.19871665**. At that
coordinate the residual addition contributes zero local rounding error; the
difference is already in its operands.

The trace runner checks its exported output, SSM state and newest convolution
slot against an ordinary compiled step. It refuses to interpret a trace that
exceeds the original tolerance. Inspect `instrumentation_vs_step` for the exact
differences. The convolution target selects the newest history slot, whose
producer is the current frame; it does not misattribute an older replayed slot
to the current computation.

#### Reproduce stage isolation

Use a fresh VM for each mode. A `+` adds a stage change to an existing dense
experiment. Supported stages are `update`, `readout`, `ssm`, `gate`, and
`output`; one SSM stage and one output stage may be combined. `--carry-f64`
promotes the initial SSM memory only; use it with `+ssm` to compare storage
precision while retaining f32 frames and convolution history. Report labels
retain `f32_*` for compatibility, but `carry_h_type` records the mixed state.

```sh
RECURRENT_CLIENT=host ERL_FLAGS='+S 2:2' devenv shell -- \
  elixir -pa '_build/dev/lib/*/ebin' scripts/run_recurrent_validation.exs \
  scripts/resolve_mamba_carry.exs \
  --capture eval_runs/mamba_accuracy_extended/mewtwo_host/capture.nx \
  --mode dense_norm+ssm --carry-f64 --out /tmp/mamba-ssm-carry

RECURRENT_CLIENT=cuda ERL_FLAGS='+S 2:2' devenv shell -- \
  elixir -pa '_build/dev/lib/*/ebin' scripts/run_recurrent_validation.exs \
  scripts/trace_mamba_precision.exs \
  --capture eval_runs/mamba_accuracy_extended/fullscale_cuda/capture.nx \
  --mode chunk8 --row 3 --target conv --out /tmp/mamba-fox-trace.json
```

For the stress trace use `--mode dense_norm --row 4 --target output` with
`mewtwo_host/capture.nx`. Row numbers and ticks are zero-based. Trace modes are
restricted to unmodified SSM/gate baselines, so the tracing equations remain
comparable to the ordinary step. Captures remain local. Reports and traces are
under `eval_runs/mamba_stage_precision/`; regenerate its summary with
`scripts/summarize_mamba_accuracy.exs`.

Tests cover unchanged f64 equations, f32 interfaces, mixed-precision carry,
readout cancellation, trace equivalence and reset handling at a target frame.
The next useful decision is whether these feature errors change controller
choices on matching replays. The current evidence does not justify a blanket
precision mode or demonstrate a gameplay regression.

**Update after broader validation:** the 160-frame pass below does not
generalize. The 512-frame, staggered, three-checkpoint checks described next
reject promotion of the eight-term method. It remains a diagnostic experiment.

## Extended promotion check

The next pass used eight rows and 512 real frames per row, with staggered
resynchronization every 31 decisions after the initial full history. Each row
uses its own recorded reset mask. The independent reference replays all rows
and selects only those due; the Agent gathers only due rows. Those two batch
shapes may round differently. Their difference is not an exact-reproduction
test for staggered f32 execution. A focused regression checks the mask logic
against explicit single-row replay, including separate and simultaneous resets.

The first three-policy sweep uses the same recorded Fox inputs. For the Mewtwo
policies this is a cross-character stress test, not a deployment-quality check.
A follow-up uses recorded Mewtwo inputs with the full Mewtwo checkpoint and
passes all comparisons. This distinction matters: the large output errors below
are sensitive to the observations, not evidence of equivalent errors in normal
Mewtwo play. The tolerance remains `atol = rtol = 1e-4`.

| Checkpoint / backend | Eight-term batch vs single | Compared with f64 reference |
| --- | --- | --- |
| `coh_fullscale_mamba_v1/model_best_policy.bin`, CUDA | Pass | 15 batch / 12 single convolution values fail |
| `mewtwo_combo_newera_mamba_full_r13_policy.bin`, CPU | 187 output values fail | 405 batch / 401 single output values fail |
| Same full checkpoint, CUDA | 477 output values fail | 632 batch / 714 single output values fail |
| `mewtwo_combo_newera_screen_mamba_r13_policy.bin`, CPU | Pass | 0 batch / 2 single output values fail |
| Full Mewtwo checkpoint, Mewtwo replay, CPU | Pass | Pass; worst error/bound ratio below 0.07 |

The full checkpoint's CPU dense-f64 experiment makes batch and single results
identical, yet both fail 315 output values against the f64 reference. Adding
f64 normalization leaves 232 failures. Therefore identical batching results
alone are not an accuracy guarantee. Dense rounding explains the earlier batch
sensitivity, but other f32 operations or intermediate rounding also matter on
these weights. The SSM update, gating and residual sums are candidates for the
next isolation experiment; this pass does not establish which is responsible.

No production API was added because the agreed numerical gate failed. No
tolerances were widened. Existing behavior and deployment defaults are intact.

### Complete Agent latency

Separate sequential CUDA timing runs were made with no other GPU compute job.
They use the original fullscale checkpoint, eight rows and staggered resets.
The first 160 calls are excluded to remove cold compilation and warm the replay
shapes. The remaining 352 calls include 261 ordinary and 91 resync decisions.

| Method | Ordinary median / p95 | Resync median / p95 |
| --- | ---: | ---: |
| Existing f32 | 1.217 / 1.700 ms | 4.972 / 5.622 ms |
| Eight-term experiment | 1.100 / 1.443 ms | 6.287 / 6.668 ms |

These are full `batch_get_controllers` calls, including embedding and heads,
excluding diagnostic tensor copies, Dolphin and IPC. One run per method is
insufficient to claim an ordinary-frame speedup. The experiment adds roughly
1.3 ms to the median resync call. Numerical CPU captures overlapped other
diagnostics; their incidental timings are retained but are not idle benchmarks.

Raw reports, checkpoint/replay hashes and individual timing samples are under
`eval_runs/mamba_accuracy_extended/`. `summary.json` is generated by
`scripts/summarize_mamba_accuracy.exs`. The original fullscale extended capture
predated the metadata sidecar; its fixture hashes are the same as the dedicated
`timing_baseline` capture. Captures themselves remain local and untracked.

### Reproduce the broader check

```sh
RECURRENT_CLIENT=host ERL_FLAGS='+S 2:2' devenv shell -- \
  elixir -pa '_build/dev/lib/*/ebin' scripts/run_recurrent_validation.exs \
  scripts/resolve_mamba_carry.exs --ticks 512 --rows 8 --stagger \
  --policy checkpoints/mewtwo_combo_newera_mamba_full_r13_policy.bin \
  --mode chunk8 --require-pass --out /tmp/mamba-extended
```

This command deliberately exits **1** when the reference gate fails, after
saving the report. Omit `--require-pass` for exploratory runs. Use
`--capture /tmp/mamba-extended/capture.nx --mode dense_norm` in a new process
to test dense and normalization precision together on identical inputs.
`--replay PATH`, `--character mewtwo` (default `fox`) and `--offset N` select
another replay, player or starting point. The matching-character follow-up uses
`replays/greg/mewtwo/Game_20240222T222930.slp`. Ensure there are at least
`ticks + offset + (rows - 1) * 37` usable frames.

For complete Agent timing, use `--capture-only --mode chunk8
--capture-mode chunk8` (or both modes `baseline`) with a fresh output directory.
The replacement is installed before starting the Agent in that private VM.
Timing captures alone do not perform numerical validation.

```sh
devenv shell -- elixir -pa '_build/dev/lib/*/ebin' \
  scripts/summarize_mamba_accuracy.exs eval_runs/mamba_accuracy_extended \
  eval_runs/mamba_accuracy_extended/summary.json

devenv shell -- elixir -pa '_build/dev/lib/*/ebin' \
  -e 'Application.ensure_all_started(:axon); ExUnit.start(); Code.require_file("test/scripts/mamba_carry_resolution_test.exs")'
```

## Original 160-frame finding

The measured batch-versus-single discrepancy comes from floating-point
accumulation in dense layers. We found no reset or carried-state bookkeeping
error in the tested trajectory. This is a numerical diagnosis, not a claim that
every checkpoint or resync schedule is correct.

Mamba keeps both a convolution history and an SSM hidden state between frames.
Small differences in a dense layer become inputs to later frames. A batch of
eight and eight separate evaluations can use different orders of addition.
With f32 numbers, these orders can give slightly different answers.

We captured every embedded input, reset, convolution buffer and output feature
from the real batch Agent. An independent compiled loop then reproduced the
Agent **bit for bit**, on both CPU and CUDA. Running that same loop in f64 made
batch-versus-single differences fall to about 1e-12 in convolution history and
1e-13 in output features. Raising dense accumulation precision removed the
strict comparison failures; raising normalization precision alone did not.

The earlier four-frame output check missed failures. Checking all 160 frames
found 24 failing output values on CPU. Thus unchanged deterministic controller
choices are useful evidence, but do not prove numerical equivalence.

## Scope and results

The fixture is `checkpoints/coh_fullscale_mamba_v1/model_best_policy.bin`, with
eight streams from `Game_20261022T170737.slp`, spaced 37 replay frames apart.
The Agent uses deterministic output, parsed action frames, zero delay, an
80-frame window, and aligned resync every 30 steps. Each stream runs 160 frames.
All runs use EXLA `precision: :highest` and the installed dependency build.
The reference uses the same already-rounded inputs and parameters promoted to
f64; it isolates execution error, not checkpoint quantization error.

Source revisions: ExPhil `f8ebabcb3270c4ee79c8df6bb6a671aeac783106`,
Edifice `6f369863ad2455c92d88394e9cfacb753b74a70f`. The runner reuses the
installed dependency beams; this is not the separate dependency-candidate build.
Fixture SHA-256 values:

```text
checkpoint 3eeb7556fe9c5d1f3fc7c3c29f993e50c9e257620615afcd702217af1d268da6
replay     98a1d4a3fe94f7f2147c60c4d99c289d1de79368ee9868aa2e9c5cbda54cd28e
```

The elementwise bound stays `1e-4 + 1e-4 * abs(reference)`. Counts below are
individual tensor values, not frames. Hidden state `h` passes throughout.

| Method | CPU convolution failures | CPU output failures | CUDA convolution failures | CUDA output failures |
| --- | ---: | ---: | ---: | ---: |
| Existing f32, batch vs single | 764 | 24 | 32 | 0 |
| Dense products accumulated in f64 | 0 | 0 | 0 | 0 |
| Eight-term f32 products, partial sums combined in f64 | 0 | 0 | 0 | 0 |

Both modified methods also pass comparisons of batch and single executions
against the f64 reference. The unmodified path fails some of those comparisons.
The eight-term method's worst error/bound ratio is below 0.87 across the CPU and
CUDA reference checks. A ratio above one is a failure. This is useful evidence,
but not a guarantee on longer or different trajectories.

Other experiments narrow the cause: f64 normalization, f64 input projection
alone, explicit row-wise dots, and 32-term partial products all leave failures
on CPU. The successful experiment changes all dense calls in the step path;
it does not yet identify a single responsible projection.

## Cost and decision

Idle RTX 5090 timings, microseconds, 40 synchronized warm calls after four
warmups. Compilation is excluded. Replay processes 79 frames from zero state.
These timings exclude Agent embedding, controller sampling, and Dolphin.

| Method | Single step median/p95 | Eight-bot step median/p95 | Eight-bot replay median/p95 |
| --- | ---: | ---: | ---: |
| Existing f32 | 145 / 210 | 178 / 255 | 5354 / 5594 |
| Dense f64 | 178 / 230 | 1314 / 1425 | 77552 / 77903 |
| Eight-term partial products | 149 / 232 | 188 / 246 | 7254 / 7429 |

Full f64 dense accumulation is too costly for the batched replay path. The
eight-term experiment is much cheaper, but still adds about 35% to replay time.
Keep production behavior unchanged for now. The original tolerance remains a
failed numerical gate on the existing path; this investigation explains the
failure rather than waiving it.

Before adopting the cheaper method, test longer trajectories, additional trained
checkpoints, and staggered row resets, then measure complete Agent latency.
If it holds up, implement a narrow Mamba stepping option with explicit accuracy
and speed expectations. The current in-memory replacement is diagnostic only.
No win-rate, netplay, training, dependency-candidate, or Mamba-2 claim follows
from these results.

## Reproduce

Run from the ExPhil root. The bare-VM runner uses existing dependency beams and
does not rebuild shared native libraries. The checkpoint and replay are local
fixtures, not included in this commit. See `scripts/lib/benchmark_sweep.exs`
for their locations. Check that the GPU is idle before measuring timings.

```sh
RECURRENT_CLIENT=cuda ERL_FLAGS='+S 4:4' devenv shell -- \
  elixir -pa '_build/dev/lib/*/ebin' scripts/run_recurrent_validation.exs \
  scripts/resolve_mamba_carry.exs --ticks 160 --rows 8 --out /tmp/carry-baseline

RECURRENT_CLIENT=cuda ERL_FLAGS='+S 4:4' devenv shell -- \
  elixir -pa '_build/dev/lib/*/ebin' scripts/run_recurrent_validation.exs \
  scripts/resolve_mamba_carry.exs --mode chunk8 \
  --capture /tmp/carry-baseline/capture.nx --out /tmp/carry-chunk8

RECURRENT_CLIENT=cuda ERL_FLAGS='+S 4:4' devenv shell -- \
  elixir -pa '_build/dev/lib/*/ebin' scripts/run_recurrent_validation.exs \
  scripts/bench_mamba_precision.exs --mode chunk8 \
  --capture /tmp/carry-baseline/capture.nx --out /tmp/carry-timing.json
```

Use `RECURRENT_CLIENT=host` for CPU. Modes are `baseline`, `dense`, `norm`,
`input`, `row_dot`, `chunk32`, and `chunk8`. `input` selects this fixture's
264-wide input projection and is not a general architecture selector.

Captures contain model parameters and large tensors; keep them local. Reports
are retained under `eval_runs/mamba_carry_resolution_20261027`. Historical
reports predate the explicit mode/verdict metadata; their directory names
identify the mode. `actual_agent_vs_compiled_trajectory` is a reproduction
check only for baseline. For modified modes it compares the original Agent to
the experimental computation, so a difference there is expected. Reference
checks cover every convolution buffer and output feature, plus the final SSM
state; they do not store every intermediate SSM state.

**AI-generated poem**

> A sum remembers how it grew;
> Each frame can carry rounding through.
> We trace the numbers, count the cost,
> And check that no old frame was lost.
