# Nx upgrade validation

**AI-generated.** Changes and measurements by Astra.

The upstream configuration now selects Nx/EXLA
`ebe2b9fad5e4eba64004342ebf78607e057ffab8` and Complex 1.0.
The November 6 successor requires Fine 0.2, which conflicts with Pythonx's
Fine 0.1 requirement. The selected revision includes the October gradient
fixes, unrolled random generator and block-container support without forcing
that native dependency conflict.

The normal ExPhil devenv sets `EDIFICE_LOCAL_NX=1`. To resolve the pinned
configuration explicitly, use:

```sh
EDIFICE_LOCAL_NX=0 EDIFICE_UPSTREAM_NX=1 EDIFICE_PATH=../edifice-nx-current mix deps.get
```

The regression runner below uses existing ExPhil beams plus a separately built
candidate Edifice/Nx/EXLA. It exercises current test sources without rebuilding
ExPhil's unrelated native dependencies. Its JSON records the loaded module
paths and hashes. This is runtime compatibility validation, not a clean build
of every ExPhil dependency.

```sh
elixir -pa '_build/dev/lib/*/ebin' scripts/check_nx_upgrade.exs \
  --edifice-build ../edifice-nx-current/_build/exphil_runtime --client host --out /tmp/nx-host.json
elixir -pa '_build/dev/lib/*/ebin' scripts/check_nx_upgrade.exs \
  --edifice-build ../edifice-nx-current/_build/exphil_runtime --client cuda --optimizer-only --out /tmp/nx-training.json
```

The new optimizer cases run the autoregressive policy, loss, gradient, clipping
and optimizer update for Mamba-1 and canonical Mamba-2. They require finite loss,
updated weights, and finite parameters/optimizer state.

For complete stochastic controller sampling:

```sh
elixir -pa '_build/dev/lib/*/ebin' scripts/validate_nx_sampling.exs \
  --edifice-build ../edifice-nx-current/_build/exphil_runtime --out /tmp/nx-sampling --rounds 2
```

Run with no other compute job on the GPU. The runner uses existing policy and
replay paths from `scripts/lib/benchmark_sweep.exs`, checks input/source hashes,
records candidate beam hashes, compares exact priming controller streams, and
rejects detected GPU contention. It compares Mamba-1 and the older Mamba-SSD
policy at batch sizes 1 and 8. The latter is not canonical `:mamba2_full`.
Each round uses fresh VMs; the second reverses the implementation order.
Browser GPU use remains a source of timing variation.

Historical labels in the reused policy profiler say `rolled`/`unrolled`.
Here `rolled` means upstream Nx, whose RNG is now also unrolled. The new
manifest calls the implementations `nx` and `helper` to avoid that ambiguity.
The existing Edifice sampling helper remains necessary for consumers on
published Nx 1.0; this change does not switch ExPhil's controller implementation.


**AI-generated poem**

> Keep each seed and measure twice;
> a smaller diff can still suffice.

## Recorded results

The 16-run sampling comparison used `c0d3bd26` before the Fine dependency
conflict was found. Nx's Elixir source is identical to the selected `ebe2b9fa`;
the successor changes native Fine integration. The compatible build was then
checked separately: nine sampling/stateful tests passed on host, and both
complete Mamba optimizer-update cases passed on CUDA (49 unrelated tests were
excluded). A clean dependency resolution succeeded with Complex 1.0/Fine 0.1.6.
The exact original beam hashes are retained in the sampling manifest.

All eight paired controller priming hashes matched. Ordinary-call p50 ranges
across the two runs, in milliseconds:

| Policy | Batch | Current Nx | Edifice helper |
| --- | ---: | ---: | ---: |
| Mamba-1 | 1 | 1.496–1.727 | 1.328–1.854 |
| Mamba-1 | 8 | 1.333–1.750 | 1.153–1.781 |
| Legacy Mamba-SSD | 1 | 2.166–2.412 | 2.595–3.296 |
| Legacy Mamba-SSD | 8 | 1.554–1.707 | 1.433–1.619 |

Each call category has 100 samples. Reset, resync, windowed, p95 and individual
samples are retained in `eval_runs/nx_upgrade_20261106/sampling/`. No competing
GPU compute process was detected; desktop GPU activity remained present. The
variation does not support a universal speedup or a strict latency-equivalence
claim. It does show that the old several-fold isolated RNG advantage no longer
translates into a clear across-the-board advantage over current upstream Nx.

The two optimizer checks use a constant positive learning rate and disable
warmup. The normal warmup schedule starts at zero, so checking for changed
weights on its first step would be an invalid regression test.

## Final runtime and precision checks

Edifice's shell ran OTP 27 and ExPhil's shell ran OTP 28. The final downstream
candidate was rebuilt under ExPhil's own devenv into `_build/exphil_runtime`;
its native library was materialized separately from the OTP-27 build's native
library. Do not assume separate Mix build directories isolate EXLA's native
library: development builds can symlink both to the same dependency cache.
Avoid rebuilding that dependency while either application is running.

Both full optimizer checks passed again with the OTP-28 candidate. The nine
sampling/stateful cases passed on CUDA with `precision: :highest`, matching the
existing complete-controller benchmark. With the default matrix precision, the
canonical Mamba-2 sequence/step comparison differed by about 0.00058 and failed
its existing tolerance. The runner now selects the intended precision explicitly;
no tensor tolerance or model implementation was changed. This does not establish
sequence/step parity at the default lower precision.

The selected candidate is therefore validated for the stated high-precision
configuration. The initial sampling timing table is exploratory; the final
compatible-build check is recorded separately in `compatible_sampling/`.

For a new validation build, use a separate Edifice checkout with its own
`deps/` and `_build/`, and compile it from ExPhil's devenv. Supply that build's
absolute path to `--edifice-build`. The `_build/exphil_runtime` paths above name
the local build used for these results; they are not a required directory name.

The final compatible/OTP-28 controller check completed all eight fresh-VM runs
(four pairs, one round). All paired priming hashes matched. Ordinary-call p50,
in milliseconds, was Nx/helper: Mamba-1 batch 1 **1.395/1.338**, batch 8
**1.139/1.353**; legacy Mamba-SSD batch 1 **2.029/2.155**, batch 8
**1.274/1.364**. This confirms that keeping the existing helper is a maintenance
and published-version compatibility decision, not a claim that it is faster
than current Nx. The raw results include p95, reset/resync/windowed cases,
runtime versions, source/beam hashes and the native-library hash.
