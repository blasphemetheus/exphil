# Recurrent replay and embedding profile — 2026-10-25

Three changes are implemented: compile history assembly with replay, replay only
due/reset rows using bounded selection shapes, and assemble batched observations
on the host before one device transfer. No model weights, resync schedule or
inference defaults changed.

## Full Agent measurements

Trained autoregressive policies, batch 8, highest CUDA precision, portable
kernels, 30 samples per case after warmup. Each sample restores the same recorded
history; the timed call includes embedding, recurrent inference, action heads,
and controller transfer. Setup, compilation and fixture restoration are excluded.

Median milliseconds:

| Policy / case | Before | Replay optimization only | Replay + host embedding |
|---|---:|---:|---:|
| MinGRU ordinary frame | 12.460 | 8.438 | 0.917 |
| MinGRU one row resync | 16.582 | 10.213 | 2.828 |
| MinGRU three rows resync | 14.480 | 9.866 | 2.773 |
| MinGRU all eight resync | 14.085 | 10.297 | 2.801 |
| MinGRU one row cold reset | 11.787 | 10.086 | 2.842 |
| Mamba ordinary frame | 8.579 | 8.619 | Pending GPU availability |
| Mamba one row resync | 16.652 | 13.177 | Pending GPU availability |
| Mamba all eight resync | 16.248 | 14.807 | Pending GPU availability |

MinGRU optimized p95: 1.336 ms ordinary, 3.191 ms one-row resync, 3.197 ms
all-row resync. Measurements were sequential on a shared machine; unrelated CPU
checks ran concurrently. The initial ordinary MinGRU baseline varied around
8–12 ms between passes, so do not interpret the table as a precise universal
speedup factor. Replay-only did not consistently improve every dense-reset case
(e.g. three-row Mamba cold reset); see raw JSON rather than extrapolating.

Erlang call-time attribution on 30 ordinary MinGRU frames identifies embedding
as the main remaining cost after replay optimization: **6.755 ms → 0.419 ms**
per batch after host assembly. Sampler attribution was about 0.32–0.35 ms.
Traced callees are accounted separately; these are CPU/dispatch attributions,
not synchronized GPU kernel durations. Profile collection was separate from the
latency samples.

Batch initialization now warms tensor/tuple replay and power-of-two selection
shapes on throwaway state, avoiding first-use JIT during staggered resets.
Measured batch-8 setup was about 2.3 seconds MinGRU and 3.6 seconds Mamba for the
replay-only run. Startup therefore costs more; steady-state carry is untouched.

## Correctness and limitations

- Replay-only MinGRU/Mamba and the complete MinGRU optimization match baseline
  controller structs exactly, and all features/carry within atol=rtol=1e-4.
- 75 targeted integration tests pass, plus two additional pointwise-convolution
  replay cases. Coverage includes six Edifice backbones, nested state, tuple
  history, padded selection, untouched rows, batch sizes 2 and 5, partial resets,
  observe-only frames, toggling resync, warmup-state preservation and rollback.
- Real Mamba and Mamba-2 single-Agent checks pass on CPU after the full update:
  cold action matches windowed, warmup preserves state, rollback is exact.
- A separate Edifice registry-sweep test acquired ~29 GiB GPU memory after an
  idle preflight check during the final Mamba run. XLA autotuning failed; that
  attempt produced no timing artifact and is not included in these results.
  Final full-update Mamba CUDA timing remains pending a free GPU.
- No closed-loop quality claim is made here. The earlier resync/default behavior
  recommendations remain unchanged.

Edifice's batch-32 backbone profile covers MinGRU, GRU, Mamba and Mamba-2 on both
CPU and CUDA. See `edifice/bench/results/recurrent_resync_2026-10-25/README.md`.

## Reproduce

Use `scripts/bench_batched_resync.exs` with a fixed checkpoint and replay,
`--batch 8 --iterations 30 --fixtures /tmp/baseline` on the baseline revision,
then `--reference /tmp/baseline` on the new revision. `--profile` adds a separate
Erlang call-time attribution pass. Checkpoints/replay paths are in each JSON.
The baseline Agent source is ExPhil `4ebb8a04`; all variants use the same compiled
dependencies and precision settings. Local fixtures are temporary and not required
for the normal Agent path. Run with an otherwise idle GPU for clean timings.
