# Stochastic sampling kernels — 2026-10-26

**AI-generated benchmark report.**

Edifice's `SamplingRandom.uniform/2` statically unrolls the twenty Threefry
rounds, allowing EXLA to fuse small uniform draws. ExPhil uses it throughout
stochastic controller sampling. Distribution, random stream, returned keys,
draw order, temperatures, and action selection remain unchanged. Sampler cache
keys are versioned so older compiled executables cannot mask the upgrade.

## Full-controller GPU results

RTX 5090, Ryzen 9950X3D, EXLA CUDA, highest precision, portable recurrent paths
(`EDIFICE_FUSED_CUSTOM_CALL=0`). Median milliseconds per full Agent controller
call, including embedding, model, fresh RNG keys, heads, and host transfer.
Eight rows means one batch of eight policies, not latency divided by eight.

| Policy | One row: reference → unrolled | Eight rows: reference → unrolled |
|---|---:|---:|
| MinGRU | 3.526 → 1.396 | 3.770 → 1.330 |
| GRU | 3.957 → 1.121 | 3.698 → 1.221 |
| Mamba | 3.999 → 1.064 | 3.860 → 1.119 |
| Mamba-2 | 3.405 → 1.567 | 2.930 → 1.082 |

GPU ordinary-call median speedups span 2.17–3.76x. The larger graph adds setup
cost: recorded CUDA setup increased by approximately 0.35–0.98 seconds. Setup
includes more than compilation and is only one observation per profile. These
are warm inference improvements, not cold-compilation improvements.

CPU ordinary-call medians (milliseconds) are mixed:

| Policy | One row: reference → unrolled | Eight rows: reference → unrolled |
|---|---:|---:|
| MinGRU | 0.573 → 0.517 | 0.650 → 0.757 |
| GRU | 0.513 → 0.467 | 0.843 → 0.778 |
| Mamba | 0.574 → 0.613 | 1.266 → 1.328 |
| Mamba-2 | 0.659 → 0.647 | 1.175 → 1.138 |

CPU changes range from about 10% faster to 16% slower. These sequential paired
runs are not enough to distinguish small effects from workstation variability;
there is no demonstrated general CPU speedup. All optimized CPU ordinary-call
medians remain below 1.33 ms per batch in these fixtures.

See [TABLES.md](TABLES.md) for p95, cold resets, partial/full
resynchronization, and windowed inference. Raw JSON contains all 100 durations
per scenario. CPU performance should be assessed separately; the GPU loop
overhead does not predict equivalent CPU gains.

## Correctness and method

- Three Edifice tests pass: exact f32 output bytes and next-key bytes across
  multiple seeds and odd/even shapes, chained draws, and invalid keys.
- ExPhil targeted sampling suites pass on both host and CUDA: 34 tests,
  0 failures, 1 excluded on each. Independent and autoregressive samplers,
  per-head temperatures, deterministic buttons/hysteresis, mode-of-N,
  event context, release events, and duration commitments are covered.
- Paired profiles run the same sampler with only uniform draws switched back
  to Nx.Random for the reference. Fixed-key priming controller streams match
  exactly for every policy/batch pair. Timed calls use fresh keys; timing
  samples are not asserted to produce identical stochastic trajectories.
- Each backend runs 16 sequential profiles: four checkpoints, one/eight rows,
  reference/candidate. Source, checkpoint, and replay hashes are verified
  before and after the suite. Each CUDA launch checks that the GPU is free.
  This is a shared workstation benchmark, not a controlled load experiment.
- Explicit-key rollback checks remain enabled. Compilation and restore/reset
  calls themselves are excluded from per-controller measurements. A cold-reset
  scenario measures replaying state with already-compiled executables.

The archived Mamba-2 independent-head checkpoint remains a speed fixture with
the previously documented leaked-label caveat. These measurements do not promote
it for play. Recurrent/resync defaults remain unchanged; BPTT GRU retains carry.

## Reproduce

From ExPhil, with no training or other GPU job running:

```sh
RECURRENT_CLIENT=host EDIFICE_FUSED_CUSTOM_CALL=0 devenv shell -- elixir -pa '_build/dev/lib/*/ebin' scripts/run_recurrent_validation.exs scripts/test_sampling_kernels.exs
RECURRENT_CLIENT=cuda EDIFICE_FUSED_CUSTOM_CALL=0 devenv shell -- elixir -pa '_build/dev/lib/*/ebin' scripts/run_recurrent_validation.exs scripts/test_sampling_kernels.exs
devenv shell -- elixir -pa '_build/dev/lib/*/ebin' scripts/sweep_sampling_kernels.exs --client cuda --out /tmp/sampling-pairs
devenv shell -- elixir -pa '_build/dev/lib/*/ebin' scripts/sweep_sampling_kernels.exs --client host --out /tmp/sampling-pairs
devenv shell -- elixir -pa '_build/dev/lib/*/ebin' scripts/summarize_sampling_kernels.exs /tmp/sampling-pairs
```

Manifests list exact local checkpoint/replay paths and commands. Inputs must be
available locally. `SAMPLING_REFERENCE=1` applies only in the validation runner;
production uses the optimized sampler. Use the implementation commits that
accompany this report; raw JSON revisions identify their pre-change parents,
while manifest hashes identify the measured source precisely.
The original orchestration helpers were later replaced with reusable Elixir
scripts. Historical manifests retain their original paths/hashes; the numerical
harness and sampling implementation are unchanged by that tooling migration.

## Next work

Measure sustained actor throughput and p95 latency with multiple bots and
realistic resource contention. Compare CPU actors with GPU batches before
changing defaults. Use that profile to decide whether combining recurrence and
sampling into one compiled call is worthwhile. Activation checkpointing still
depends on the upstream Nx work; it is a training-memory improvement rather
than the next inference-latency task.

**AI-generated poem**

> The same drawn numbers leave the gate;
> A shorter path can cut their wait.
