# Trained Mamba batch validation

**AI-generated implementation notes and report.**

[Measured results](../../eval_runs/mamba_batch_parity_20261026/RESULTS.md)
include the remaining strict carry failures and runtime comparisons.

A batch of bots and separate bots should receive the same input values. Small
floating-point differences inside the compiled model are a separate question.
This investigation checks both boundaries and keeps the existing elementwise
`atol=rtol=1e-4` comparison. It does not relax any regression tolerance.

## Embedding fixes

Two single-input paths used Elixir floating-point division and only then
converted to f32. Training and batch inference first converted to f32 and
multiplied by an f32 reciprocal. The affected features were hitstun frames,
action frames, and player distance. The single-input paths now use the same
primitives as training. Checkpoint shapes and training preprocessing are unchanged.

The new exact-equality tests fail against the old implementations. All 152
focused player/game embedding tests pass after the fixes.

## Diagnostic method

`scripts/diagnose_mamba_batch.exs` loads the trained Fox Mamba policy at
`checkpoints/coh_fullscale_mamba_v1/model_best_policy.bin`. Its metadata specifies
an 80-frame window, frame delay zero, no delay-ID input, and no BPTT. Each of eight
rows receives a different section of a real Fox replay. Each independent Agent
receives exactly the same observations as its corresponding batch row.

The diagnostic records input embeddings, carried state, and deterministic
controller equality for 160 decisions per row. `--stagger` uses the supported
phase API: batch staggering and independent phases `[0,3,7,11,15,19,23,27]`.
It also compares one Mamba step with identical inputs and identical incoming
carry at four sampled frames. The stage probe exposes the existing primitives;
returning intermediate tensors can change compiler fusion, so its results are
reported separately from the ordinary complete step. A double-precision input
projection helps distinguish batch-shape rounding from different equations.

These checks use recorded game observations. A short replay cannot prove every
possible controller choice will match. Near a
sampling or argmax boundary, small numerical differences can still matter.
The diagnostic uses deterministic heads so random sampling does not obscure the
first source of a difference.

```sh
RECURRENT_CLIENT=host ERL_FLAGS='+S 4:4' devenv shell -- \
  elixir -pa '_build/dev/lib/*/ebin' scripts/run_recurrent_validation.exs \
  scripts/diagnose_mamba_batch.exs --stagger --out /tmp/mamba-host.json
# Use RECURRENT_CLIENT=cuda for GPU validation.

devenv shell -- elixir -pa '_build/dev/lib/*/ebin' \
  scripts/summarize_mamba_batch.exs /tmp/mamba-host.json

RELOAD_PLAYER=1 RECURRENT_CLIENT=host ERL_FLAGS='+S 4:4' devenv shell -- \
  elixir -pa '_build/dev/lib/*/ebin' scripts/run_recurrent_validation.exs \
  scripts/test_mamba_embedding.exs
```

## Framework comparison

`scripts/run_framework_validation.exs` merges an isolated dependency build's
BEAM paths over the installed application paths. It invokes Elixir directly and
does not fetch dependencies, rebuild native libraries, or alter the main build.
It replaces the Python launcher for these inference checks. The report records
loaded versions and paths, representative BEAM and EXLA native-library hashes,
and input/source hashes checked at both ends of the run.

```sh
RECURRENT_CLIENT=cuda ERL_FLAGS='+S 4:4' devenv shell -- elixir \
  scripts/run_framework_validation.exs \
  ../exphil-nx-upgrade/scripts/framework_stack/_build/dev/lib \
  scripts/diagnose_mamba_batch.exs --stagger --out /tmp/mamba-candidate.json

devenv shell -- elixir -pa '_build/dev/lib/*/ebin' \
  scripts/compare_mamba_framework.exs /tmp/mamba-baseline.json \
  /tmp/mamba-candidate.json /tmp/mamba-comparison.json
```

The comparator requires matching inputs, sources, backend, and schedule. It
compares all deterministic controllers exactly and sampled output features at
`atol=rtol=1e-4`. This complements the training-update and resume experiments
already recorded on `chore/framework-baseline`; it does not repeat them or
certify fused training kernels, a full application rebuild, or match quality.

## Native Dolphin measurement

`scripts/measure_dolphin_resync.exs` runs two independent Fox agents controlling
both ports through the native Elixir bridge. It uses the checkpoint's trained
zero-delay setting, live action-frame conversion, and a separate temporary
Dolphin home. It explicitly paces requests because the headless executable
advanced faster than its requested emulation speed in the smoke run. Requests
follow an absolute 60 Hz schedule; the actual intervals are recorded. This
avoids adding console-read time to a fresh 16.67 ms sleep on every frame.

```sh
RECURRENT_CLIENT=cuda ERL_FLAGS='+S 4:4' devenv shell -- \
  elixir -pa '_build/dev/lib/*/ebin' scripts/run_recurrent_validation.exs \
  scripts/measure_dolphin_resync.exs --phase 15 --frames 600 \
  --out /tmp/dolphin-spread.json
# Compare --phase 0. Run arms sequentially on an otherwise idle machine.
```

The first 160 frames are excluded from timing summaries, covering compilation,
history fill, and the first complete resync. Pair inference time includes
waiting for both agents. A pair taking over 16.67 ms exceeds the inference
budget. Blocking input and explicit pacing mean these counts are **not** an
independent asynchronous deadline measurement. They also do not measure wins,
long-match quality, eight-emulator contention, or netplay behavior.

**AI-generated poem**

> One small sum can turn a stream;
> We check the numbers, then the scheme.
> Two clocks share a measured night;
> The facts decide which steps are right.
