# Opt-in resync phases

**AI-generated implementation notes.**

Resync rebuilds a recurrent policy's memory from recent observations. Several
bots doing that on the same frame can cause a latency spike. These options move
their steady replay schedules apart. Each replay still completes in one call;
no action uses a partly rebuilt state.

Defaults are unchanged: phase zero, no automatic spreading. The feature does
not make window-trained policies equivalent to carried-state policies, and it
does not resolve the trained Mamba batch/single numerical discrepancies recorded
in `eval_runs/resync_spread_20261026/RESULTS.md`. BPTT-trained policies reject
periodic resync and non-default phase settings.

## Separate agents

Assign a different `stateful_resync_phase` to each Agent:

```elixir
{:ok, bot} = ExPhil.Agents.Agent.start_link(
  policy: checkpoint,
  stateful_step: true,
  stateful_resync: 30,
  stateful_resync_phase: 7
)
```

For eight bots and interval 30, offsets `[0, 3, 7, 11, 15, 19, 23, 27]` evenly
divide the existing 31-call cycle. Phase is an integer from zero through the
configured interval. It advances the counter once, so larger offsets cause an
earlier second replay. It is not a delay added to every replay.

Both Dolphin launchers forward these CLI options:

```text
--stateful-step --stateful-resync 30 --stateful-resync-phase 7
```

## One batched agent

```elixir
{:ok, bot} = ExPhil.Agents.Agent.start_link(
  policy: checkpoint,
  stateful_step: true,
  stateful_resync: 30,
  stateful_resync_stagger: true
)
:ok = ExPhil.Agents.Agent.batch_init(bot, 8)
```

Row `i` receives `rem(base_phase + div(i * (interval + 1), row_count), interval + 1)`.
The base phase is `stateful_resync_phase`, default zero. More rows than positions
in the cycle necessarily share phases. `--stateful-resync-stagger` exposes the
same setting through the CLI; it only spreads work when the Agent uses batching.

The earlier measurements favor spreading for CPU batches and independent GPU
agents. Aligned GPU batches already met their deadlines and had better typical
latency. Do not enable this universally on GPU batches or infer new performance
numbers from the correctness tests.

## Start, reset, observation and rollback

- Cold padding and real-history requirements are unchanged. The first eligible
  full replay still happens on the old schedule. Only after that fresh replay
  does the counter receive its one-time phase. This means startup can still
  contain a synchronized replay; the option targets steady operation.
- Later uninterrupted replays remain `interval + 1` decision calls apart,
  matching the existing check-before-increment behavior. Phase assignment only
  shortens the first interval after a fresh replay, never lengthens it.
- A single reset rearms its phase. Partial batch resets rearm only those rows
  and require their new history to fill. Other rows keep their counters and state.
  Loading another policy clears the old replay history and rearms the phase;
  a batch must be initialized again for the newly loaded model.
- Observation-only calls keep collecting history and incrementing counters,
  but do not replay, as before. A pending phase is consumed only by an actual
  replay. Pauses, observe-only bursts and independently timed resets can change
  relative alignment; this is not a global wall-clock scheduler.
- `Agent.reconfigure/2` accepts the two options and the interval. Changes rearm
  phase assignment after the next complete replay, without postponing a replay
  already due. Invalid values return an error without altering state. Settings
  may be retained while resync is disabled; they have no effect until enabled.
- Snapshot version 3 includes the interval, phase options, pending assignment,
  counters and history. Stateful batch snapshots also contain row carry, replay
  history, reset flags and controller history. Restoring reproduces the saved
  schedule, even after reconfiguration. Versions 1/2 retain single-agent support
  with phase zero; they cannot restore batch state, which they never recorded.

## Verification

The bare-VM runner reuses compiled dependencies without rebuilding native code:

```sh
RECURRENT_CLIENT=host devenv shell -- elixir -pa '_build/dev/lib/*/ebin' \
  scripts/run_recurrent_validation.exs scripts/test_resync_phases.exs
RECURRENT_CLIENT=cuda devenv shell -- elixir -pa '_build/dev/lib/*/ebin' \
  scripts/run_recurrent_validation.exs scripts/test_resync_phases.exs --phases-only
```

Tests compare MinGRU, Mamba and Mamba-2 batch rows against independent Agents
with matching phases. They check exact counter transitions, unchanged carry
tolerances, partial resets, observation-only calls, rollback before and after
phase assignment, invalid reconfiguration, BPTT rejection, CLI defaults and flags.
They use small synthetic policies; they do not certify trained-policy gameplay.

Validation on October 26: 107 host tests passed; all four phase-specific tests
passed on CUDA (103 unrelated tests excluded). No timing or gameplay claims are
made from these correctness runs.

**AI-generated poem**

> Each bot recalls its frames in turn;
> The clocks shift once, the memories learn.
