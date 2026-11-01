# Complete controller latency, November 1, 2026

**AI-generated measurements and interpretation.**

For this checkpoint, batching eight bots on the GPU met every offered frame
deadline in the measured phases. Eight separate Agents did not. One bot met
every deadline on either CPU or GPU. Staggering replay helps CPU latency, but
does not make every CPU frame meet its deadline.

## What was measured

Checkpoint: `checkpoints/coh_fox_mamba_p314_ep4/model_best_policy.bin`.
This is the latest saved Fox experiment inspected for this task, not a claim
that it is the best policy. It is the existing Mamba architecture, state size
16, two layers, hidden width 512, window 80, with autoregressive controller
heads and previous-action input. This is not a Mamba-2 quality comparison.

Each call starts with a parsed game state and ends with a host-side controller.
It includes Agent dispatch, game-state encoding, recurrent state updates or
history replay, all six autoregressive heads, stochastic sampling, controller
conversion and previous-action bookkeeping. It excludes Dolphin transport,
emulation, rendering and configured game-frame delay. This is warm execution;
compilation and history fill are outside timing and included in `setup_us`.

Hardware: RTX 5090 and AMD Ryzen 9 9950X3D (16 cores, 32 threads). The existing
ExPhil runtime reports Nx/EXLA 1.0.0, Axon 0.9.0 and Edifice 0.3.0. Loaded BEAM
files and the source files recompiled by the bare-VM runner are hashed in
`manifest.json`; version strings alone do not identify the upstream build.
The runner uses highest EXLA precision and disables optional fused custom calls.
No dependency or native-library rebuild was performed.

All phases offer 60 calls per second per worker for ten seconds. Each batch call
returns eight controllers. The deadline is 16.67 ms from the scheduled tick.
One late batch therefore affects eight bots. Separate Agents have separate
worker streams. Late workers keep processing the offered calls, so response
time includes accumulated backlog. Service time is the call's own elapsed time.

The same recorded Fox replay supplies inputs. Each row receives a different
offset, and rows maintain their own previous actions. The competing load is a
repeated 2048-square CUDA matrix product and sum, synchronized after each call.
It is a controlled GPU competitor, not an actual training workload. Browser GPU
activity remained and was recorded. No competing non-desktop GPU process was
observed. These short phases do not certify long-run or live-game deadlines.

## Baseline results

Desktop activity remained; the first table has no synthetic competitor.
P99 means 99% of recorded responses were at or below that duration.

| Execution | Typical call | P99 response | Missed/offered calls |
|---|---:|---:|---:|
| One bot, CPU | 1.37 ms | 12.19 ms | 0/600 |
| One bot, GPU | 2.41 ms | 7.61 ms | 0/600 |
| Eight separate Agents, CPU | 6.34 ms | 74.68 ms | 984/4800 |
| Eight separate Agents, GPU | 12.23 ms | 43.42 ms | 1176/4800 |
| Eight-row batch, CPU | 2.67 ms | 64.51 ms | 80/600 |
| Eight-row batch, GPU | 2.17 ms | 8.27 ms | 0/600 |

Under synthetic GPU load, one CPU bot still had no misses (p99 11.83 ms), and
one GPU bot had no misses (p99 13.51 ms). The eight-row GPU batch had no misses
and p99 9.94 ms. Eight separate GPU Agents missed all 4800 deadlines; their
typical service time was 18.62 ms and backlog grew to about 1.9 seconds. This
does not mean a single inference call took 1.9 seconds.

## Why resync matters

This window-trained checkpoint uses a replay threshold of 30. In this Agent
implementation that produces one replay every 31 calls. The resync experiment
validates the cadence, repeats aligned and staggered schedules twice, and
reverses schedule order in the second round. It uses the existing benchmark
helper's counter assignment; it does not change deployment defaults.

Post-processing classifies each sample using that validated schedule, without
adding instrumentation to the timed call. In round one without synthetic load:

| Eight-row batch | Ordinary call median | Replay-call median | Deadline misses |
|---|---:|---:|---:|
| CPU, aligned | 2.62 ms | 67.05 ms | 76/600 |
| CPU, staggered | 2.62 ms | 12.76 ms | 9/600 |
| GPU, aligned | 2.10 ms | 6.63 ms | 0/600 |
| GPU, staggered | 2.04 ms | 5.73 ms | 0/600 |

The aligned CPU phase had 19 replay calls; every one missed its deadline. Their
backlog caused another 57 ordinary calls to finish late. Staggering replaces
those few large replay batches with more frequent small ones: 154 calls included
some replay, with nine deadline misses and no ordinary-call misses in that phase.

Across the two CPU rounds, staggered batch p99 response was 17.01–17.09 ms
without synthetic load (9 and 11 misses), versus aligned p99 65.44–68.87 ms
(76 misses each). Under the synthetic GPU load, staggered CPU batches had zero
and four misses. Do not infer that load improves CPU performance from this small
sample. GPU batches had zero misses across all ten baseline/resync phases:
6000 batch calls total, including both load conditions and both schedules.

Staggering also reduced the separate GPU Agents' unloaded p99 from roughly
43 ms to 21–25 ms. They still missed 11–22% of deadlines, and all deadlines
under load. Spreading replay does not solve their total throughput shortage.

Batch/single carried-state differences are preserved in the resync validation
JSON. Cadence and finite values were checked; numerical equivalence and gameplay
quality were not established by this timing experiment.

## What to do with this result

Use a shared GPU batch for multiple bots running the same checkpoint when the
caller can collect their states together. Keep a single CPU controller as a
useful option when GPU capacity is needed elsewhere; measure its live-game
budget before choosing a deployment default. CPU resync is the clearest remaining
latency cost here. GPU resync already fits within the measured budget, so another
GPU kernel is not justified by these results alone.

The next model question remains the controlled ExPhil training-quality
comparison. These measurements do not tell us whether canonical Mamba-2 learns
a better policy or which memory-state size it needs.

## Reproduce and inspect

From ExPhil, with no competing compute job and the existing compiled runtime:

```sh
devenv shell -- elixir -pa '_build/dev/lib/*/ebin' scripts/sweep_controller_latency.exs \
  --policy checkpoints/coh_fox_mamba_p314_ep4/model_best_policy.bin \
  --replay /home/blewf/Slippi/2026-10-Mainline/Game_20261022T170737.slp \
  --character fox --seconds 10 --out eval_runs/controller_latency_20261101

devenv shell -- elixir -pa '_build/dev/lib/*/ebin' scripts/analyze_controller_resync.exs \
  eval_runs/controller_latency_20261101
```

Use a new output directory if inputs, measured sources or compiled modules have
changed. The sweep is for periodically resynchronized, non-BPTT checkpoints;
its resync phase explicitly rejects BPTT models rather than resetting their carry.
The runner resumes completed cases only with matching hashes and options.

`TABLES.md` contains all 44 phases. The four report JSON files contain setup,
cadence checks and summary statistics; their `.samples.json.gz` companions hold
raw timestamps and durations. `resync_service.json` contains the replay/ordinary
classification. The sweep recomputes medians, p95, p99, maxima and deadline counts
from raw samples before marking the manifest complete.

**AI-generated poem**

One choice returns before the bell;  
Eight queues have different tales to tell.  
We spread the past across the frame,  
And keep the clock beside the game.
