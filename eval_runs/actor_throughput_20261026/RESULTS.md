# Continuous actor throughput and deadlines — 2026-10-26

**AI-generated report. Status: CPU and CUDA complete.**

All eight profiles (96 phases) pass raw-sample verification. The four CUDA
profiles completed in a clean GPU window. Earlier contaminated attempts remain
excluded. All four eight-row CUDA batches missed zero deadlines, both idle and
under controlled load, in these three-second phases.

| GPU policy | Eight independent Agents, idle actions/s | Eight-row batch, idle actions/s | Batch with load, actions/s |
|---|---:|---:|---:|
| MinGRU | 822 | 7,573 | 3,755 |
| GRU | 1,265 | 7,812 | 3,773 |
| Mamba | 1,061 | 6,442 | 3,403 |
| Mamba-2 | 836 | 6,375 | 3,544 |

Independent CUDA Agents miss deadlines despite sufficient average throughput:
MinGRU misses 68/1,440 idle calls and 814 under load; Mamba misses 102 and 192;
Mamba-2 misses 96 and 421. GRU misses none in either condition. Batching avoids
these misses in this screen, but these short runs are not a long-run guarantee.

The CPU result is clear enough to guide the next experiment. Eight-row batches
deliver 2.6–4.4 times the throughput of eight independent actors on these fixtures.
MinGRU and GRU miss no 60 Hz deadlines. Mamba's eight independent actors miss
280/1,440 idle requests and 296/1,440 requests with GPU load; batching reduces
that to 9/180 batch calls in each condition. A late batch call affects eight bots.
Long Mamba service times recur at measured request indices 2, 33, 64, 95, 126,
and 157. That pattern is consistent with the configured periodic resync and
suggests testing staggered resync work next. This is an inference from timing
and configuration, not a separately instrumented attribution.

This screen compares full stochastic controllers after the exact-stream sampler
optimization. It exercises one Agent, eight independent concurrent Agents, and
one eight-row batch on CPU and GPU. See [TABLES.md](TABLES.md) for all results.
All actors/rows within a profile share one checkpoint; this does not measure
batching different model weights together.

## What is measured

- Four trained fixtures: MinGRU, GRU, Mamba, and archived Mamba-2, using the same
  checkpoint/replay inputs as the sampling-kernel profiles. Different rows read
  replay observations offset by 37 frames; frame numbers increase monotonically.
  Replay content cycles if the maximum-throughput run exhausts the 3,600-frame
  observation bank. This is a computational workload, not a closed-loop game.
- Full public Agent calls include embedding, recurrence, fresh-key stochastic
  heads, transfer, and controller decoding. State evolves continuously; there
  is no snapshot restore between measured calls. Every phase starts from the
  same primed state. Non-BPTT policies use opt-in resync30; BPTT GRU retains carry.
- **Saturated:** each worker requests its next controller immediately after the
  previous response. Throughput counts individual actions, so an eight-row batch
  contributes eight actions per call. Workers are persistent tasks, not a new
  task per request. The independent actors share one BEAM VM and EXLA client.
- **60 Hz:** each worker has a fixed schedule, with one request every 16.67 ms.
  Late workers retain that schedule and drain all offered requests; lag is not
  discarded. Response latency is completion minus the intended request time,
  including scheduling delay and backlog. Service latency starts when the call
  is actually submitted. Each worker offers 180 calls over three seconds.
  Scheduling uses BEAM millisecond timers, whose wakeup jitter is included in
  response latency; this is not a measurement of Dolphin's frame delivery jitter.
  One late batch call makes all eight controller rows late.
- Both cases run idle and with a tight-loop CUDA 2,048 × 2,048 f32 matrix multiply
  plus reduction, synchronizing its result each iteration. The same competitor
  is used with CPU and GPU actors. It represents controlled GPU contention, not
  actual training, emulator load, or training throughput. Competitor iteration
  counts are recorded but no uncontended competitor baseline is measured. It
  shares the BEAM VM and CUDA client; separate-process GPU context contention
  and training memory pressure are not modeled.
- Each phase lasts at least three seconds; a backlog may extend it. These short
  screens locate bottlenecks and do not establish long-run tail guarantees.
  Starts and periodic resyncs are aligned across workers, stressing simultaneous
  demand. Measurements exclude setup/compilation and state restore between
  phases. Warmup is checked to preserve each Agent's public snapshot.

## Reproducibility and validation

Hardware: RTX 5090, Ryzen 9950X3D. EXLA highest precision, portable recurrent
paths (`EDIFICE_FUSED_CUSTOM_CALL=0`). Host and CUDA sweeps run sequentially.
Every profile requires a free GPU before launch, including host profiles because
they exercise the same GPU competitor. A once-per-second process monitor records
unexpected extra GPU compute processes; contaminated profiles fail validation.
This cannot detect every subsecond workload or all CPU interference.
The first Mamba-2 host profile and two MinGRU CUDA attempts overlapped unrelated
GPU processes and were excluded. The Mamba-2 CPU rerun and all four final CUDA
profiles completed cleanly. Original artifacts and monitor evidence are retained
under `contended/`; it contributes no numbers to the primary tables.

The manifest records source/checkpoint/replay hashes, exact commands, GPU model
and driver. Hashes must remain unchanged through the sweep. Summary JSON retains
worker counts, service/response percentiles, deadline misses, elapsed time, and
competitor progress. Compressed raw samples store each worker's ordered
`[submission_offset_us, service_us, response_us]` measurements.
The summarizer independently recomputes percentiles, counts, and deadlines from
those samples and checks consistency with every report. It also checks every
paced worker completed all 180 offered calls.

From ExPhil, with the GPU free:

```sh
devenv shell -- elixir -pa '_build/dev/lib/*/ebin' scripts/sweep_actor_throughput.exs --out /tmp/actor-throughput
devenv shell -- elixir -pa '_build/dev/lib/*/ebin' scripts/summarize_actor_throughput.exs /tmp/actor-throughput
```

The driver uses the bare-VM validation runner, avoiding native library rebuilds.
Change `--seconds` for longer phases and use a new output directory. Inputs are
local checkpoint/replay files named in the manifest. The archived Mamba-2 fixture
retains its prior leaked-label caveat and is not being promoted for play. No
Agent implementation, inference defaults, or training configuration is changed.

The initial CPU profiles used an earlier orchestration helper. After the user's
language preference was clarified, orchestration and reporting were moved to
Elixir. The measured Elixir harness and all inputs remained identical. The
manifest preserves the original metadata in `pre_elixir_manifest.json` and
records which valid CPU profiles were retained. `--import-existing` performs
this narrowly checked migration; normal resumes require all source hashes to
match. Contaminated runs are archived and can be retried by rerunning the command.

To validate the completed CPU subset before resuming:

```sh
devenv shell -- elixir -pa '_build/dev/lib/*/ebin' scripts/summarize_actor_throughput.exs --allow-partial
```

**AI-generated poem**

> Eight clocks can keep an even pace,
> Then pause together in one place.
> Spread out the work and test again;
> The next result will tell us when.
