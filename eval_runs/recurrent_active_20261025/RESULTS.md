# Active-opponent resync results — October 25, 2026

Keep deployment defaults unchanged. Resync30 is a reasonable **next evaluation
candidate** for these MinGRU/Mamba exports. It is not a demonstrated universal
upgrade. The archived Mamba-2 does not show a quality case for periodic resync;
its windowed baseline remains the conservative choice. BPTT GRU retains carry.

All 52 planned runs completed: 176 environment-minutes, four paired seeds per
arm. MinGRU/Mamba/GRU used eight environments; Mamba-2 used four independent-head
Agents. Every run faced the same windowed MinGRU opponent on FD for 1,800 frames,
with seeded stochastic sampling and synchronous zero-delay inference. These are
NIF simulator screens, not Dolphin/netplay matches or win-rate measurements.
See [protocol](PROTOCOL.md), [raw aggregate and paired differences](summary.json),
and [source/checkpoint/simulator provenance](manifest.json).

## Quality

Mean across the four seed-level runs. Do not compare architectures as a training
bake-off: checkpoint sizes, heads, characters and training histories differ.

| Policy / mode | SD/min | Deaths/min | Kills/min | Damage dealt/min | Damage taken/min |
|---|---:|---:|---:|---:|---:|
| MinGRU windowed | 4.812 | 5.375 | 5.062 | 20.451 | 34.909 |
| MinGRU carried | 4.938 | 5.688 | 4.875 | 22.414 | 25.743 |
| MinGRU resync30 | 4.375 | 5.000 | 5.188 | 21.299 | 26.361 |
| MinGRU resync60 | 4.250 | 4.812 | 4.625 | 27.136 | 34.250 |
| Mamba windowed | 1.312 | 1.438 | 3.875 | 79.887 | 30.451 |
| Mamba carried | 1.500 | 1.625 | 4.188 | 78.264 | 33.730 |
| Mamba resync30 | 1.312 | 1.438 | 4.000 | 80.989 | 32.015 |
| Mamba resync60 | 1.562 | 1.688 | 4.062 | 79.274 | 32.480 |
| Mamba-2 windowed | 0.750 | 0.750 | 1.625 | 53.239 | 41.549 |
| Mamba-2 carried | 1.125 | 1.125 | 1.000 | 60.685 | 46.828 |
| Mamba-2 resync30 | 1.500 | 1.500 | 1.625 | 59.053 | 41.317 |
| Mamba-2 resync60 | 1.125 | 1.250 | 1.375 | 51.326 | 54.757 |
| GRU carried (BPTT control) | 1.125 | 1.250 | 4.000 | 58.112 | 24.269 |

**MinGRU:** resync30 has a balanced mean result relative to windowed: fewer deaths,
similar kills and damage. It reduces deaths in three of four paired seeds, but
damage improves in only two. Resync60 deals more damage in every seed and has
fewer mean deaths, but also fewer mean kills. Neither interval dominates all
metrics. Indefinite carry has more mean deaths and slightly fewer kills than
windowed. The earlier two-seed CPU screen was less favorable to resync30; those
different runs are not pooled here, and the variation argues against promotion.

**Mamba:** resync30 matches windowed death counts in every paired seed, with
damage-dealt differences of -1.000, +0.070, +1.050 and +4.287 per minute. It is the
closest behavioral candidate among the carried modes. Both indefinite carry and
resync60 have somewhat more mean deaths and damage taken. Carry has more kills,
so this remains a tradeoff rather than proof that resync30 is stronger.

**Mamba-2:** resync30 increases deaths in three seeds and ties the fourth; its mean
death rate doubles versus windowed while kills remain equal. Indefinite carry
also loses more stocks and gets fewer kills despite dealing more damage.
Resync60 has more deaths, fewer kills and less damage dealt than windowed on
average. This export has legacy leaked training labels and an impossible causal
reaction-delay target (-1); it is useful for implementation validation, not a
clean test of a correctly trained Mamba-2 policy.

**GRU:** all four carried BPTT controls completed. Resync/window truncation was
intentionally not tested because it violates this checkpoint's training contract.
Its isolated profile and rollback checks are in
[the full-policy report](../recurrent_policy_20261025/RESULTS.md).

## Latency tradeoff

Separate, isolated CUDA probes at the same group sizes, with resident trained
policies and 50 timed samples after warmup. Milliseconds for the whole group:

| Policy (rows) | Windowed median | Ordinary carry median | All-row resync median | Resync30 estimated average | Resync60 estimated average |
|---|---:|---:|---:|---:|---:|
| MinGRU (8, batched) | 1.845 | 0.932 | 2.747 | 0.991 | 0.962 |
| Mamba (8, batched) | 2.491 | 0.989 | 6.753 | 1.175 | 1.083 |
| Mamba-2 (4, sequential Agents) | 10.726 | 5.596 | 15.251 | 5.907 | 5.754 |

The averages are **estimates**, `carry + (resync - carry)/(threshold + 1)`, using
an uninterrupted synchronized schedule. They are not measured rollout averages;
partial resets, staggered histories and runtime noise change actual costs. Each
resync frame still incurs the full replay latency. Sampling in these probes is
deterministic, whereas quality rollouts use seeded stochastic sampling. Neither
the emulator nor simulator/diagnostic overhead is included. Raw results,
including p95 and partial-row cases, are in
`../recurrent_policy_20261025/*_tradeoff.json`.

For MinGRU/Mamba, resync30 retains most of the measured carry speedup while
tracking the windowed quality baseline reasonably in this screen. Mamba-2's
similar compute savings do not rescue its observed quality regressions. Its
single-Agent and eight-Agent profiles show why independent-head batching is a
more direct throughput target than another recurrence optimization.

## Limits and next decision

Four seed-level replicates are a screen, not evidence of statistical superiority.
Environment rows within a run are not counted as independent run-level samples.
The metrics are proxies for play quality, not match wins; especially with low
death counts, a few stocks change the rates substantially. Recovery and controller
behavior metrics remain in the raw JSON. No policy was promoted or retrained.

Before changing deployment settings, evaluate the promising MinGRU/Mamba modes
at their trained delay regime, across stages and stronger/different opponents.
For Mamba-2, prioritize a causal-trained checkpoint for quality work and batched
independent-head inference for throughput; do not infer architecture quality
from this archived export. Keep GRU's carried-state contract intact.

One Mamba attempt failed on an XLA temporary-file disk error and produced no
result. The queue resumed from that arm with compiler scratch in RAM-backed
storage. All 52 final JSON files pass protocol checks; recorded model, source and
simulator-library hashes were unchanged at completion. No failed/partial run was
included. The protocol records the failure and recovery.
