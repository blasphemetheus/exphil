# Reusable inference benchmarks

**AI-generated tooling and documentation.** The tools listed here were written
by the coding assistant. This attribution does not claim human review or approval.

All runtime measurements, sweep orchestration, integrity checks, and report
generation use Elixir. Run from ExPhil's root inside its own devenv environment.
The compiled dependencies must already be available. The bare-VM runner avoids
rebuilding native libraries used by other processes.

| Tool | Purpose |
|---|---|
| `profile_actor_throughput.exs` | Continuous public Agent calls: single, eight concurrent actors, or eight-row batch; saturated or 60 Hz; idle or controlled GPU load |
| `sweep_actor_throughput.exs` | Run four checkpoint fixtures on host/CUDA; hash inputs, wait for idle GPU, detect and archive interference |
| `summarize_actor_throughput.exs` | Recompute metrics from compressed raw samples and produce tables; `--allow-partial` explicitly supports unfinished sweeps |
| `sweep_sampling_kernels.exs` | Compare reference and unrolled sampling with exact seeded-controller parity; `--dry-run` prints the planned commands |
| `summarize_sampling_kernels.exs` | Validate paired sampling reports and render their comparison table |
| `lib/benchmark_sweep.exs` | Shared streaming hashes, JSON files, GPU process checks, subprocess execution, and fixture paths |

```sh
devenv shell -- elixir -pa '_build/dev/lib/*/ebin' scripts/sweep_actor_throughput.exs --out /tmp/actors --seconds 3
devenv shell -- elixir -pa '_build/dev/lib/*/ebin' scripts/summarize_actor_throughput.exs /tmp/actors
devenv shell -- elixir -pa '_build/dev/lib/*/ebin' scripts/sweep_sampling_kernels.exs --client cuda --out /tmp/sampling
devenv shell -- elixir -pa '_build/dev/lib/*/ebin' scripts/sweep_sampling_kernels.exs --client host --out /tmp/sampling
devenv shell -- elixir -pa '_build/dev/lib/*/ebin' scripts/summarize_sampling_kernels.exs /tmp/sampling
```

Use a new output directory after changing measured sources or inputs. Resume
an unchanged actor sweep with the same command. A run that detects another GPU
process is archived under `contended/` and excluded from completed runs. The
actor runner waits up to ten minutes for a sixty-second quiet interval before
each new profile. It does not stop another process.

The recorded actor sweep at `eval_runs/actor_throughput_20261026` has all eight
validated CPU/CUDA profiles (96 phases). Earlier contaminated attempts are
excluded. All four eight-row CUDA batches met every deadline in this short
screen. See its `RESULTS.md` and `TABLES.md` for the measurements and limits.

The controlled matrix workload shares a process/client with the actors. These
measurements do not establish performance alongside actual training or Dolphin.

`sweep_resync_spread.exs` compares aligned and staggered Mamba replays on CPU
and GPU, with two ten-second rounds per condition. Run
`summarize_resync_spread.exs` to verify its raw deadline samples. Recorded results
are in `eval_runs/resync_spread_20261026`: spreading helps independent Agents
and CPU batches, while GPU batches already meet deadlines and have lower typical
latency when aligned. The experiment changes no Agent defaults. See its report
for measured batch/single carry discrepancies and the limits of the comparison.

**AI-generated poem**

> Each clock records a little part;
> Clear checks give numbers room to start.
