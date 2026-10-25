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
agreement at resync boundaries. Mamba-2 still needs an ExPhil Agent adapter;
this probe calls Edifice directly. Older SSD matmul-path exports retain the
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

Remaining improvements:

- Compile cold-start/window replay as a single chunk operation, reducing the
  many separate launches on startup and resync.
- Implement per-row resync for batched inference. `stateful_resync` currently
  applies to the single-agent decision path, not `batch_get_controllers`.
- Integrate the validated Mamba-2 step adapter into ExPhil, then measure the
  complete policy with real checkpoint dimensions. These exports use raw
  input widths 264/288, unlike the 516-wide controlled microbenchmark.
- Validate window-trained carry in closed-loop play before changing defaults;
  train with explicit carried state if indefinite memory is desired.
