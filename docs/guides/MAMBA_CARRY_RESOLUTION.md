# Mamba carried-state investigation

**AI-generated — 2026-10-27.** This note accompanies
`scripts/resolve_mamba_carry.exs`, `scripts/bench_mamba_precision.exs`, and
`scripts/lib/mamba_precision_experiment.exs`.

## Finding

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
