# Spreading Mamba resync across frames

**AI-generated report. CPU and GPU measurements complete.**

Spreading replay frames helps independent Mamba Agents on both backends. It
also removes almost all CPU batch stalls. For GPU batches, keeping rows aligned
is faster most of the time and already meets every deadline in this test.

| Mode | Aligned missed calls, rounds 1 / 2 | Spread missed calls, rounds 1 / 2 |
|---|---:|---:|
| CPU, independent, idle | 935 / 912 | 139 / 129 |
| CPU, independent, GPU load | 976 / 928 | 197 / 170 |
| CPU, batch, idle | 29 / 24 | 1 / 0 |
| CPU, batch, GPU load | 26 / 25 | 0 / 0 |
| GPU, independent, idle | 462 / 464 | 0 / 0 |
| GPU, independent, GPU load | 844 / 829 | 243 / 299 |
| GPU, batch, idle | 0 / 0 | 0 / 0 |
| GPU, batch, GPU load | 0 / 0 | 0 / 0 |

Each independent-Agent phase offers 4,800 calls; each batch phase offers 600
calls covering 4,800 actions. On idle GPU, independent-Agent response p99 falls
from about 41 ms to 14 ms. Under load it falls from 46 ms to 18 ms, still above
the 16.67 ms frame budget. CPU batch stalls also largely disappear.
See [TABLES.md](TABLES.md) for every phase.

GPU batching has a tradeoff: spreading raises idle response p95 from about
3.6 ms to 7.2–7.6 ms, because more frames now contain a small replay. Both
schedules meet every GPU batch deadline. Do not enable spreading indiscriminately.
The useful next integration is configurable phase assignment for independent
Agents and CPU batches, followed by gameplay and longer contention testing.

Carry comparison maxima are similar in both schedules: CPU absolute error
0.000694 aligned versus 0.000689 spread; GPU 0.000340 versus 0.000345. Worst
errors relative to each tensor's maximum magnitude are approximately 22 parts
per million on CPU and 11 on GPU. These are measured discrepancies, not proof
of exact equivalence. The original elementwise check fails for 135 versus 133
CPU tensor comparisons and 18 versus 19 GPU comparisons. No tolerance in an
existing regression was changed to obtain these timings.

This experiment assigns eight bots different replay frames. Each bot still
rebuilds its complete recent history in one call. It does not split one bot's
replay across several frames or use a partly rebuilt state for an action.

The aligned schedule rebuilds all eight rows together every 31 calls. The
spread schedule separates their replay frames by three or four calls. Both
use ExPhil's existing `stateful_resync: 30` behavior: the counter is checked
before incrementing, so steady replay spacing is 31 calls.

Before assigning phases, every bot performs a fresh replay. Advancing its
counter then makes its first scheduled replay earlier; it never extends the
normal maximum interval. No Agent implementation or deployment default changes.

Outside the timed section, the harness checks two complete replay cycles for
both schedules. It checks every counter transition and compares each batch
row's carried tensors with an independent Agent using the same observations
and schedule. This also compiles partial-row replay before timing it. The trained
model's aligned baseline already exceeds elementwise `atol=rtol=1e-4` for some
carried values (the first observed absolute error was 0.00055). Absolute errors,
tensor scales, and elementwise check outcomes are retained for every row and
frame. Timing results do not certify batch/single numerical parity. Replay
counter transitions, finite carry, and exact 31-call spacing are required.

Measurements use eight independent Agents and one eight-row batch. Each offers
60 calls per second for ten seconds, both idle and with the same synchronized
2,048-square CUDA matrix workload used by the actor throughput screen. Two
rounds reverse schedule order. Each phase restores the same primed state.
Response time includes queueing and missed-frame backlog; one late batch call
delays all eight rows. These are short computational screens, not gameplay
quality tests or long-run deadline guarantees.

All bots share the existing Fox Mamba checkpoint. The replay, checkpoint and
measured sources are hashed in the manifest. Unexpected GPU processes reject
a run. The runner uses existing compiled dependencies and does not rebuild NIFs.

```sh
devenv shell -- elixir -pa '_build/dev/lib/*/ebin' scripts/sweep_resync_spread.exs
devenv shell -- elixir -pa '_build/dev/lib/*/ebin' scripts/summarize_resync_spread.exs
```

**AI-generated poem**

> Eight memories refresh in turn;
> Each keeps the frames it needs to learn.
> A quieter clock is worth the test;
> The measured tails decide the rest.
