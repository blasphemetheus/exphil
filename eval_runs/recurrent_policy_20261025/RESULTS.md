# Trained GRU and Mamba-2 full-policy profile — October 25, 2026

GRU's supported autoregressive batch path is efficient. The archived Mamba-2
export uses independent heads and therefore runs one Agent per environment;
batched independent-head sampling is the clearest remaining throughput target.
No weights, heads, deployment defaults or training contracts were changed.

The GRU export has two 512-wide layers, a 264-wide input and autoregressive heads.
Mamba-2 has two 256-wide layers, a 288-wide input, state size 16, expansion 2 and
legacy independent heads. Both record a 60-frame window; BPTT GRU nevertheless
retains carry rather than truncating at that window. These are different trained
policies, not a parameter-matched architecture comparison.

## CUDA latency

RTX 5090, highest precision, portable kernels, deterministic controllers, 50
samples after five warmups. All numbers are milliseconds for the **whole group**,
including embedding, recurrence, action heads and controller transfer. Row resets,
fixture restoration and JIT compilation are outside the timer. Eight independent
Mamba-2 Agents run sequentially, as in the existing simulator fallback.

| Policy / path | Rows | Median | p95 |
|---|---:|---:|---:|
| GRU ordinary, single Agent | 1 | 0.989 | 1.575 |
| GRU cold reset, single Agent | 1 | 0.951 | 1.455 |
| GRU ordinary, batched | 8 | 0.869 | 1.268 |
| GRU reset one row | 8 | 0.983 | 1.254 |
| GRU reset three rows | 8 | 0.980 | 1.210 |
| GRU reset all rows | 8 | 0.933 | 1.267 |
| Mamba-2 ordinary | 1 | 1.392 | 1.838 |
| Mamba-2 cold reset | 1 | 3.810 | 4.271 |
| Mamba-2 resync | 1 | 3.755 | 4.157 |
| Mamba-2 windowed | 1 | 2.685 | 3.333 |
| Mamba-2 ordinary | 8 | 13.282 | 17.237 |
| Mamba-2 reset one row | 8 | 14.043 | 16.427 |
| Mamba-2 reset three rows | 8 | 18.796 | 20.649 |
| Mamba-2 reset all rows | 8 | 33.223 | 36.487 |
| Mamba-2 resync one row | 8 | 14.842 | 16.950 |
| Mamba-2 resync three rows | 8 | 18.935 | 21.808 |
| Mamba-2 resync all rows | 8 | 31.398 | 34.757 |
| Mamba-2 windowed | 8 | 21.358 | 23.854 |

The independently warmed single- and eight-row runs are not a controlled scaling
curve; do not interpret GRU's small difference as batching making one request
intrinsically faster. The machine is shared, but no competing GPU workload was
observed. A CUDA autotuning timer warning occurred during Mamba-2 windowed setup;
the recorded measurements use host wall clocks after compilation and warmup.

At one Agent, a simple amortization of the measured Mamba-2 costs gives about
1.47 ms/frame at resync30 and 1.43 ms/frame at resync60, versus 1.39 ms indefinite
carry and 2.69 ms windowed. These are estimates, not measured rollout averages;
resync frames still incur their full latency. At eight Agents, synchronized resync
exceeds a 16.67 ms frame budget. Staggering reduces spikes but does not remove the
independent-head throughput limitation.

## Correctness and scope

- Both checkpoints pass public single-Agent snapshot/restore checks: controllers
  and final serialized state are exactly reproduced across a recorded suffix.
  Warmup preserves the initial snapshot. Cross-resync rollback was separately
  checked in `../recurrent_replay_20261025/`; this profile's suffix is short.
- Batched GRU passes an internal state-fixture rollback check. This is explicitly
  **not** a claim that the public single-Agent snapshot API supports batch state.
- GRU was trained with BPTT; cold reset starts fresh carry and consumes the next
  observation once. Window recomputation and periodic resync are intentionally
  not benchmarked as valid GRU modes.
- The harness uses parsed observations, synchronous zero frame/reaction delay,
  and original checkpoint heads. The loader flags this archived Mamba-2 as
  legacy leaked-label training with a trained reaction-delay value of -1. It
  cannot be matched by a causal zero-delay harness. These results characterize
  this checkpoint's implementation, not a promotion candidate or canonical
  Mamba-2 gameplay quality.
- Priming uses unique recorded frame numbers and more than a full window.
  An initial discarded probe had only 53 history entries after 60 inputs and
  never actually resynchronized. The final probe asserts a saturated history and
  verifies that every requested resync row's counter resets. All JSON here comes
  from the corrected probe.

## Reproduce

Use `scripts/run_recurrent_validation.exs` with compiled ExPhil dependencies to
avoid rebuilding a shared native library. It loads the current Agent and Edifice
replay source. Source revisions, checkpoint and replay paths are in each JSON.

```sh
EDIFICE_FUSED_CUSTOM_CALL=0 devenv shell -- elixir -pa '_build/dev/lib/*/ebin' \
  scripts/run_recurrent_validation.exs scripts/profile_recurrent_policy.exs \
  --policy checkpoints/fox_v3_20260917_050901_resume3/model_policy.bin \
  --replay /home/blewf/Slippi/2026-10-Mainline/Game_20261022T170737.slp \
  --character fox --batch 8 --iterations 50 --out /tmp/gru-profile.json
```

For Mamba-2 use `checkpoints/mewtwo_combo_newera_mamba2_screen_best_policy.bin`,
`replays/greg/mewtwo/Game_20240222T222930.slp`, and `--character mewtwo`.
Run each with `--batch 1` and `--batch 8` on an otherwise free GPU.
