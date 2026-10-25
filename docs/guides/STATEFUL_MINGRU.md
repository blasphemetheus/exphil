# MinGRU stateful inference

Temporal MinGRU policies now support `--stateful-step`, or
`stateful_step: true` when starting `ExPhil.Agents.Agent`. This remains opt-in;
loading a policy without the flag retains windowed inference.

The adapter uses `Edifice.Recurrent.MinGRU` with the exported input projection,
recurrent layers and final normalization. State dimensions are inferred from
the trained weights. Independent and autoregressive controller heads use the
existing sampling path. Single-agent reset, snapshot/restore, batched stepping
and per-row resets are covered by synthetic GameState tests.

Carrying state matches evaluating the full prefix from the same initial state.
It does not preserve a window-trained policy's finite-context behavior forever.
The existing cold-start padding and `--stateful-resync N` behavior also apply to
MinGRU. Resync bounds the time since rebuilding the window; it does not make
intervening carried steps identical to freshly evaluated windows. Validate a
trained checkpoint on recorded observations and in closed-loop play before
promoting it. No policy was deployed or retrained by this change.

Validation:

```sh
EXPHIL_GPU=0 EXPHIL_EXLA_PRECISION=highest devenv shell -- mix test \
  test/exphil/agents/agent_stateful_min_gru_test.exs \
  test/exphil/networks/stateful_step_equivalence_test.exs
```

Use highest EXLA precision for strict sequence/step comparisons; f32 tensor
storage does not alone select the GPU's dot-product arithmetic precision.

## Trained-checkpoint validation (2026-10-24)

`scripts/validate_recurrent_carry.exs` compares a trained recurrence against
full-prefix evaluation, then measures continuous carry and a 30-frame resync
counterfactual against fresh sliding windows. It holds recorded observations
and expert previous actions fixed; agreement is not a gameplay quality score.
Results are in `eval_runs/recurrent_carry_20261024/`.

The first 320-frame Fox replay probe uses highest-precision EXLA on CPU:

| Checkpoint | Prefix max absolute feature error | Carry/window full-controller agreement | Resync/window agreement |
|---|---:|---:|---:|
| `coh_prev_q_ck4/model_best_policy.bin` (MinGRU) | 4.77e-7 | 98.34% | 99.59% |
| `coh_fullscale_mamba_v1/model_best_policy.bin` (Mamba) | 1.24e-5 | 100% | 100% |
| `fox_v3_20260917_050901_resume3/model_policy.bin` (GRU) | 0 | 87.74% | 96.17% |

MinGRU's second, midgame excerpt from another replay also gives 98.34%/99.59%
agreement and a 5.96e-7 prefix error. These short excerpts do not establish
full-game equivalence. A third, 1,280-frame midgame excerpt scores 1,201 frames:
carry/window agreement falls to **96.42%**, while resync gives **99.58%**.
Its bounded prefix checks pass with a 9.54e-7 max error. Continuous carry is
numerically correct but observably different from the trained window regime.
MinGRU and Mamba also pass on the RTX 5090 with highest precision (6.26e-7 and
7.63e-6 max prefix errors), with the same controller agreements as the CPU
opening excerpt.
GRU's CUDA prefix comparison also passes (6.88e-6 max error), with the same
87.74%/96.17% carry/resync agreement against truncated windows as CPU.

The GRU checkpoint was trained with BPTT and zero carry:
**continuous carry is its reference behavior**, and resetting windows/resync
discard trained memory. Do not interpret its lower window agreement as a bug
or enable resync for that execution contract.

The archived Mamba-2 checkpoint
`mewtwo_combo_newera_mamba2_screen_best_policy.bin` passes prefix parity on a
320-frame Mewtwo replay excerpt (7.63e-6 max error), with 100% controller
agreement in both modes. All probes check exact snapshot replay and numerical
agreement at resync boundaries. This initial probe calls Edifice directly;
the Mamba-2 Agent adapter is covered by the follow-up below. Older SSD matmul-path exports retain the
compatibility caveat described in Edifice's benchmark notes.

Example (the output path must not already exist):

```sh
devenv shell -- env EXLA_TARGET=host EXPHIL_EXLA_PRECISION=highest mix run \
  scripts/validate_recurrent_carry.exs \
  --policy checkpoints/coh_prev_q_ck4/model_best_policy.bin \
  --replay /path/to/fox.slp --offset 1800 --frames 320 \
  --out /tmp/min_gru_carry.json
```

The probe rejects learned embedding preprocessing and event-head checkpoints
that require additional masking. Prefix checks use three selected lengths;
window comparisons score every frame after one full window. Prefix compilation
is bounded to 320 observed frames by default (`--prefix-limit`), plus cold
padding; long clips still carry and score their entire requested length.
Midgame excerpts
start with fresh state, rather than carrying the omitted replay prefix.

Validation found and fixed a batch-reset bug: a reset row now receives its
first-frame padding before its next real step, while other rows preserve their
carry. Regression tests cover MinGRU, GRU and Mamba; BPTT still consumes each
frame once. The 32 targeted stateful Agent/network tests pass, as do all 11
MinGRU integration/reset tests with the EXLA CUDA client explicitly selected.

## Compiled replay and Mamba-2 adapter (2026-10-25)

Cold-start padding and single-agent resync now use Edifice's compiled replay
loop. Observe-only cold starts and batched cold/reset rows use it too; continuing
batch rows retain their state until the next real frame. `Agent.warmup/1`
compiles replay using throwaway carry so menu-time warmup does not advance the
game's memory. BPTT still consumes each input once and never pads/resyncs.

Both `:mamba_2` and `:mamba_ssd` exports now support `stateful_step: true`, with
their learned projection and SSD blocks. Convolution history is allocated from
the kernel width in the weights, including legacy `:mamba_2` exports whose
builder ignored the metadata's convolution width.

Snapshots are now version 2: they include the resync buffer and counter,
game-frame debounce, airborne state, controller queue and stick-duration state.
These fields are necessary for deterministic replay across resync boundaries.
Version 1 snapshots remain readable with empty/default values for the omitted
fields; they cannot reproduce history they never stored. Stochastic action
sampling is not covered by the deterministic rollback guarantee.

The real `mewtwo_combo_newera_mamba2_screen_best_policy.bin` checkpoint passes
an Agent-level check over 96 recorded Mewtwo frames: first-frame agreement
with windowed inference, warmup preserving the snapshot, and exact actions
and final snapshot after rollback/replay through periodic resync. Result:
`eval_runs/recurrent_replay_20261025/mamba2_agent.json`.

```sh
EXLA_TARGET=host EXPHIL_EXLA_PRECISION=highest mix run scripts/validate_recurrent_agent.exs \
  --policy checkpoints/mewtwo_combo_newera_mamba2_screen_best_policy.bin \
  --replay replays/greg/mewtwo/Game_20240222T222930.slp \
  --character mewtwo --out /tmp/mamba2-agent.json
```

CPU replay-core benchmarks show cold padding dropping from 10.9 to 1.0 ms
for MinGRU, 12.9 to 4.6 ms for GRU, 27.6 to 9.3 ms for Mamba, and 11.1 to
2.8 ms for Mamba-2. These use synthetic resident backbones and exclude input
assembly, heads and first compilation; see Edifice's benchmark report. The GPU
was occupied by the overnight league and has not been timed for this change.

The integration tests now advance game frame numbers. Previously, repeatedly
submitting frame 0 let action caching mask whether subsequent inference and
rollback actually ran. They now check complete post-replay snapshots too.
All 35 targeted CPU tests pass (six Edifice replay tests and 29 Agent tests),
including both Mamba-2 aliases, legacy convolution metadata, partial batch
resets, warmup and resync rollback. Validation used separate Elixir VMs with
the existing compiled dependencies and the changed source modules loaded;
the live overnight queue's native libraries were not rebuilt.

### Batched resync and full Agent timing

`batch_get_controllers` now honors `stateful_resync` independently per row.
Each bot has a history-length counter and resync counter; the shared frame
buffer stores the last window of embeddings. A compiled replay rebuilds the due
rows and only those rows receive the rebuilt carry. A partially reset
batch cannot replay history from a row's previous game: that row must first
accumulate a complete new window. Observe-only calls collect history and
advance the counter, leaving resync to the next decision, just like the single
path. Disabling/re-enabling resync preserves recent history. Counters follow
the single-agent convention: a resync decision resets the counter to zero.

The optimized implementation gathers due rows into a power-of-two bucket capped
at the batch size. Extra bucket rows are unique continuing rows whose carry is
preserved by a runtime count mask. Selection shapes are warmed on throwaway carry
during `batch_init`, so changing due-row counts does not introduce new JIT shapes
during gameplay. All-due resync uses full replay directly. Cold partial resets
use the same selected-row path. History stacking is fused into replay by passing
a chronological tuple of frames instead of eagerly stacking the list.

Batch setup now includes replay compilation (about 2–4 seconds in the measured
batch-8 policies); it does not advance live carry. Batched observation embedding
also matches the single path: host assembly followed by one transfer to the
policy backend. This removes dozens of small eager GPU preprocessing operations.
Full before/after results: `eval_runs/recurrent_resync_profile_20261025/`.

Validation also found that Axon's stored GRU/LSTM initializer RNG key produces
different hidden values and Glorot scale at batch size N. Stateful batch init
and row reset now broadcast the batch-1 initial state, matching independent
bots rather than changing initialization with the number of environments.

`validate_recurrent_agent.exs --benchmark` measures full Agent calls, including
embedding, heads, GenServer dispatch and output transfer. Three measured passes
follow one untimed pass to remove shape compilation. For the real Mamba-2
checkpoint on CPU: windowed median **7.632 ms** (288 samples), ordinary carried
median **0.581 ms** (282 samples), cold **3.078 ms** and resync **3.586 ms**
(only three samples each). Full CPU comparison:

| Checkpoint backbone | Windowed | Ordinary carry | Cold | Resync |
|---|---:|---:|---:|---:|
| MinGRU | 2.457 ms | 0.661 ms | 1.617 ms | 2.562 ms |
| Mamba | 30.569 ms | 1.278 ms | 14.830 ms | 17.653 ms |
| Mamba-2 | 7.632 ms | 0.581 ms | 3.078 ms | 3.586 ms |

Artifacts: `eval_runs/recurrent_replay_20261025/*_agent_host_timing.json`.
These are shared-machine measurements with concurrent CPU work; GPU performance
is not implied. Cold/resync medians have only 3–6 samples, and the benchmark
excludes first compilation and simulator/IPC work beyond the Agent call.
These runs use highest EXLA precision and `EDIFICE_FUSED_CUSTOM_CALL=0`;
they do not validate the separately loaded native CUDA handlers.

Closed-loop comparisons use `scripts/sim_closed_loop.exs --stateful-step
--stateful-resync 30`. The script now fixes both simulation and sampling seeds;
the decode and sampling paths accept an explicit `:key` while retaining fresh
random keys when omitted. `--deterministic` is available for diagnostics, but
the tested MinGRU stood still under argmax, so stochastic comparisons are more
informative. Results are in `eval_runs/recurrent_resync_20261025/`.

The updated targeted CPU suite has 77 passing tests, including Edifice replay,
MinGRU/GRU/LSTM/Mamba/Mamba-2 single-vs-batch carry at window sizes 1 and 8,
batch sizes 2 and 5, padded selections, staggered resets, observe-only frames, runtime resync toggles, stochastic key
reproducibility, decode and sampling regressions. BPTT reconfiguration now
rejects window resync just as initial policy loading does.

CUDA real-checkpoint validation now passes for MinGRU, Mamba and Mamba-2:
cold action matches windowed, warmup leaves state untouched, and rollback across
resync reproduces actions and final state exactly. Contended CUDA medians
(window / carry / cold / resync), in ms: MinGRU 9.896 / 6.548 / 9.086 / 16.835;
Mamba 3.959 / 1.146 / 6.177 / 8.501; Mamba-2 3.365 / 1.690 / 4.139 / 6.463.
Another CUDA test ran concurrently, so these are not idle-device performance
figures. Artifacts: `eval_runs/recurrent_replay_20261025/*_agent_cuda_timing.json`.
A final pass after the queue/tests finished supersedes those contended timings.
No competing GPU process was observed before or after each run. Complete Agent
medians (window / carry / cold / resync), in ms: MinGRU 2.838 / 1.039 / 2.753 /
4.474; Mamba 2.591 / 0.754 / 4.644 / 6.649; Mamba-2 2.224 / 1.070 / 3.780 /
4.481. Ordinary carry p95 is 1.548 / 1.115 / 1.642 ms respectively. Resync p95
is 4.840 / 7.465 / 5.022 ms, but has only 3–6 samples. These use portable
kernels and the compiled main checkout; first JIT and emulator IPC are excluded.
Artifacts: `eval_runs/recurrent_replay_20261025/*_agent_cuda_idle_timing.json`.
A separate fused-on Mamba check also passes, with the handler library confirmed
loaded. Edifice now loads OTP crypto before the handler library; a fresh Erlang
VM regression reproduces the old load-order crash and passes with the fix.

The initial 32 seeded CPU closed-loop screens are summarized in
`eval_runs/recurrent_resync_20261025/RESULTS.md`. Against an active opponent,
MinGRU self-destructs/min were 3.125 windowed, 5.0 carried, 4.0 resync30 and
4.25 resync60. Resync30 helps but does not recover windowed performance.
Keep deployment defaults unchanged. That initial screen only used idle opponents
for Mamba/Mamba-2; the later active-opponent follow-up is linked below.

The integration was fast-forwarded onto main after the walkoff queue finished
on October 25; the application compiled successfully, including Edifice's crypto
loader fix and the press-window/grounded-neutral feature branch. The queue launcher
now stops on merge, compilation or test failures instead of running missing flags.

Larger closed-loop trials against active opponents remain necessary before changing
resync defaults. Train with explicit carried state if indefinite memory is desired.

## Additional full-policy profiles and active opponents (October 25)

`eval_runs/recurrent_policy_20261025/RESULTS.md` profiles trained GRU and Mamba-2
with 50 samples per case, including partial resets and actual resync events.
The two-layer 512-wide BPTT GRU takes 0.869 ms median for eight batched controller
outputs on CUDA; BPTT must retain carry and must not use periodic resync.
The archived independent-head Mamba-2 takes 1.392 ms for one carried Agent but
13.282 ms for eight sequential Agents. Its supported fallback, not its recurrence
alone, limits batched evaluation throughput. Independent-head batching is a useful
next integration target. The profile also checks public single-Agent rollback,
internal batched rollback and warmup-state preservation.

The four-seed CUDA active-opponent comparison is documented in
`eval_runs/recurrent_active_20261025/PROTOCOL.md` and `RESULTS.md`, with raw per-seed
metrics and paired differences. These are NIF simulator screens on FD with
synchronous zero-delay inference, not Dolphin/netplay promotion tests. The archived
Mamba-2 also has legacy leaked training labels; do not generalize its quality to a
correctly trained Mamba-2. Compare modes within a checkpoint, not architectures
against each other. Deployment defaults remain unchanged.
