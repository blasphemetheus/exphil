# Mamba rounding and controller choices

**AI-generated — 2026-10-27.**

The two matching replay checks found no deterministic controller-choice changes.
Shared-key stochastic sampling changed one Mewtwo decision and no Fox decisions.
This supports keeping the production f32 path. It does not establish equal
gameplay strength or identical long games.

| Policy and matching replay | Backend | Decisions | Deterministic changes | Stochastic changes |
| --- | --- | ---: | ---: | ---: |
| Fullscale Fox / Game_20261022T170737 | CUDA | 4,096 | 0 | 0 |
| Full r13 Mewtwo / Game_20240222T222930 | CPU | 4,096 | 0 | 1 |

Each check uses 512 frames for eight rows, offset by 37 replay frames. These
are overlapping excerpts from one replay per policy, not independent games.
Resync is staggered. Both captured Agent features and independently replayed
f32 features produce these counts. Both also pass the original feature bound
`1e-4 + 1e-4 * abs(reference)` on these inputs. This does not reverse the earlier
convolution-history failures or the Mewtwo-on-Fox stress results.

The changed Mewtwo sample is row 1, diagnostic tick 309, on an ordinary frame.
Only `main_x` changes. Exact buckets, probabilities, and logit differences are
stored in the report: the Agent selects bucket 0 while the reference selects
bucket 8. The largest probability change in that head is about 0.00000418
(0.000418 percentage points). A small probability change can therefore produce
a visibly different sampled input on a particular draw. Sampling uses temperature 1 and the same random key
`10000 + tick` on both sides. Only one draw per row/frame was tested.

## What this comparison isolates

The reference runs the Mamba trunk in f64 on the same captured embeddings,
weights, initial state, and reset schedule. Its output is rounded to f32 at
the production controller-head interface. Both sides then use unchanged f32
production heads and sampling functions, with the same batch shape of eight.
This tests the effect of trunk rounding; it is not a full f64 policy reference.
The f64 reference trajectory itself runs as a batch of eight.

Fox's autoregressive head conditions each choice on the preceding choices.
The comparator follows that production sequence, so a changed early choice
could change later logits too. Reported logits are conditional on each side's
own choices. Mewtwo's heads are independent. Its previous-controller input
remains the value saved in the original deterministic Agent capture. Sampled
choices are not fed back into future frames. This deliberately fixes the input
stream; it does not simulate a closed-loop stochastic bot.

We compare eight button bits and five stick/shoulder bucket indices before
conversion to controller values or external overrides. Reference margins are
distance to zero for button logits and the gap between the two largest
categorical logits. Those margins describe deterministic choices, not distance
to a random sampling threshold. Probabilities in changed examples are computed
from the returned logits at temperature 1 for inspection.

No Agent, Edifice runtime, deployment defaults, or native libraries changed.
The tool rejects non-baseline captures, mismatched checkpoint hashes, and
unsupported event/context head configurations. It reads capture metadata to
choose the checkpoint rather than accepting an unrelated checkpoint silently.

## Reproduce

Run from ExPhil using the existing bare-VM runner:

```sh
RECURRENT_CLIENT=cuda ERL_FLAGS='+S 2:2' devenv shell -- \
  elixir -pa '_build/dev/lib/*/ebin' scripts/run_recurrent_validation.exs \
  scripts/compare_mamba_choices.exs \
  --capture eval_runs/mamba_accuracy_extended/timing_baseline/capture.nx \
  --out eval_runs/mamba_controller_choices/fox_cuda.json

RECURRENT_CLIENT=host ERL_FLAGS='+S 2:2' devenv shell -- \
  elixir -pa '_build/dev/lib/*/ebin' scripts/run_recurrent_validation.exs \
  scripts/compare_mamba_choices.exs \
  --capture eval_runs/mamba_accuracy_extended/mewtwo_replay_host/capture.nx \
  --out eval_runs/mamba_controller_choices/mewtwo_host.json
```

The large captures are local artifacts. Capture creation and checkpoint/replay
paths are documented in [Mamba carry resolution](MAMBA_CARRY_RESOLUTION.md).
Raw reports in `eval_runs/mamba_controller_choices/` include checkpoint and replay
hashes, feature errors, component counts, ordinary/resync counts, and examples.
The comparator requires the capture's adjacent `capture_metadata.json`.

Two focused tests cover a button threshold crossing, a categorical tie,
probability differences, row selection, and resync accounting.

**AI-generated poem**

> A rounded sum can shift a draw;
> Count what it changes before changing the law.
