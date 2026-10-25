# Independent-head Mamba-2 batching

The archived Mamba-2 checkpoint now runs eight environments in one Agent batch,
without changing its weights or independent controller heads.

| Full-controller call, eight environments | Sequential Agents median / p95 ms | Batched median / p95 ms |
| --- | ---: | ---: |
| Ordinary | 13.282 / 17.237 | 1.011 / 1.183 |
| One cold row | 14.043 / 16.427 | 3.606 / 4.018 |
| Three cold rows | 18.796 / 20.649 | 4.183 / 4.563 |
| All rows cold | 33.223 / 36.487 | 4.177 / 4.686 |
| One row resync | 14.842 / 16.950 | 3.476 / 3.937 |
| Three rows resync | 18.935 / 21.808 | 4.246 / 4.639 |
| All rows resync | 31.398 / 34.757 | 4.212 / 4.730 |
| Windowed | 21.358 / 23.854 | 2.080 / 2.574 |

RTX 5090, highest EXLA precision, portable kernels, deterministic sampling;
50 samples after five warmups per case. Includes embeddings, recurrence,
heads and controller transfer. Excludes compilation, fixture restoration,
reset API calls and emulator IPC. Baseline is the earlier same-day
`../recurrent_policy_20261025/mamba2_cuda_8.json`; these are separate runs,
not interleaved measurements. No competing GPU compute process was observed.
XLA emitted delay-kernel timer warnings while compiling the windowed graph;
reported timings are warmed wall-clock controller calls, not those autotuner timings.

Validation:

- 61 focused tests passed on both CPU and explicit CUDA, including independent
  and autoregressive state parity across MinGRU, GRU, LSTM, Mamba and Mamba-2,
  partial resets, observations, controller parity and seeded batch rollback.
- Actual Mamba-2 checkpoint: 130 recorded frames, eight rows; exact deterministic
  controller equality against single Agents and carry atol=rtol=1e-4 at every
  frame, including resync and staggered resets. Warmup preserves state.
- Profile asserts that forced resync actually occurs and checks exact rollback
  using an internal complete batch-state fixture. Public snapshot support is
  still single-Agent only.

Independent windowed batches sample the full graph's logits. They reject
`return_features`/`return_sample`, since those outputs promise trunk features.
Independent batch mode-of-N is explicitly rejected instead of accidentally
selecting one action across all rows. Per-row button hysteresis history is
retained and cleared on row reset. Explicit keys reproduce trajectories at a
fixed batch shape; changing shape does not promise the same stochastic draws.

This checkpoint retains its legacy leaked-label limitation. This is a speed
and equivalence result, not a quality improvement or promotion.

Reproduce with `scripts/run_recurrent_validation.exs` and
`scripts/profile_recurrent_policy.exs --policy checkpoints/mewtwo_combo_newera_mamba2_screen_best_policy.bin --replay replays/greg/mewtwo/Game_20240222T222930.slp --character mewtwo --batch 8 --iterations 50 --validate-batch --out OUTPUT.json`.
