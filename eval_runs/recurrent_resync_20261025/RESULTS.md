# Recurrent resync screen — 2026-10-25

Keep deployment defaults unchanged. Periodic resync helps this MinGRU checkpoint
relative to indefinite carry, but does not recover windowed performance against
an active opponent. These are small screening runs, not promotion evidence.

## Protocol

Four inference modes: windowed, indefinite carry, resync every 30 or 60 frames.
Seeds 1001 and 1002, 1,800 frames per environment, Final Destination. Both
simulator and stochastic sampling use explicit seeds. Highest EXLA host precision,
native CUDA handlers disabled. Eight environments per seed except Mamba-2 (four).
MinGRU and Mamba are Fox; Mamba-2 is Mewtwo against idle Fox. The active opponent
uses the same MinGRU checkpoint in windowed mode. There are 32 runs, totalling
403,200 environment frames (112 environment-minutes). Only two seeds per arm.

Independent-head Mamba-2 uses separate Agents over the simulator's batched rows;
the other policies use batched Agent inference. No head conversion was made.
Raw JSON records checkpoints, seeds, characters and inference driver. Concurrent
host work makes rollout wall time unsuitable as an inference latency benchmark.
The earlier unseeded pilot is under `pilot/` and excluded from `summary.json`.

## Mean results across seeds

Each cell is **self-destructs/min / damage dealt/min**. Lower self-destruct rate
and higher damage are desirable; neither alone establishes gameplay strength.

| Policy / opponent | Windowed | Carry | Resync 30 | Resync 60 |
|---|---:|---:|---:|---:|
| MinGRU / idle | 1.875 / 16.55 | 3.125 / 9.68 | 2.125 / 16.08 | 2.750 / 12.30 |
| Mamba / idle | 0.500 / 41.50 | 0.375 / 44.89 | 0.375 / 44.89 | 0.375 / 44.89 |
| Mamba-2 / idle | 1.000 / 13.25 | 1.250 / 10.64 | 1.000 / 13.72 | 1.250 / 2.98 |
| MinGRU / active | 3.125 / 36.27 | 5.000 / 9.32 | 4.000 / 31.99 | 4.250 / 28.79 |

MinGRU resync30 improves both measures over indefinite carry in these screens,
but remains worse than windowed against the active opponent. Mamba's idle result
shows no clear resync benefit. Mamba-2 resync30 is closest to its windowed baseline;
it still needs active-opponent evaluation and more seeds. Do not infer statistical
significance or cross-architecture quality rankings from different checkpoints.

## Reproduction and next gate

`scripts/compare_recurrent_resync.sh POLICY OUTPUT_DIR 1001 1002` runs all four
modes. Set `ENVS`, `FRAMES`, `CHARACTER`, `OPPONENT_CHARACTER`, and `OPPONENT`
as needed (`OPPONENT` can be a checkpoint path). See `summary.json` for source
files and exact aggregate values. Use a fixed built revision for each comparison.

Next: more seeds and active opponents for all three policies, followed by a
matched delay/stage evaluation before changing deployment settings. Separately,
collect more cold/resync samples. A final idle-GPU full-Agent pass is recorded in
`../recurrent_replay_20261025/*_agent_cuda_idle_timing.json`. Real-checkpoint cold start,
warmup and exact rollback checks passed on both CPU and CUDA; those correctness
checks do not establish equivalence between carried and sliding-window policies.
