# BPTT evaluation and training delay

## Checkpoint-driven BPTT evaluation

R1/R2 were integrated on 2026-09-11. Evaluate a current exported GRU policy with:

```bash
mix run scripts/eval_model.exs --policy checkpoints/model_policy.bin \
  --replays /path/to/held-out-replays --max-files 10
```

The policy metadata supplies the architecture, head, embedding layout, label
delay, and unroll length. Full `.axon` checkpoints are supported too. Sidecar
JSON fills missing metadata; embedded checkpoint metadata takes precedence.
For an older BPTT export missing its mode stamp, add `--bptt`. This only fills a
missing flag, not an explicit `bptt: false`. Other required architecture/layout
metadata must still be present. Only load trusted checkpoint files.

Evaluation walks each replay in order, resetting GRU carry at replay boundaries
and discontinuities, not at ordinary chunk edges. Every eligible frame is scored
once, including short segments and partial final chunks. Results are frame-weighted
plain cross entropy and per-head accuracy. Autoregressive heads use ground-truth
conditioning: these are **teacher-forced metrics**, not sampled controller or
gameplay performance, and not directly comparable to weighted training loss or
windowed last-frame evaluation. The live sampler probe keeps its sampling semantics.

Current limits are explicit errors: non-GRU BPTT, mixed BPTT/windowed comparison,
queued-action or delay-id inputs, custom discretization, CSV/sequence export,
embedding-width mismatch, and legacy leaked-label evaluation. Default replay
subject is port 1; use `--character` or `--player-port` for a different subject.
Training diagnostics share the forward-input contract; BPTT gradient summaries
are skipped because the old gradient helper is windowed-only.

## One training reaction delay

Use `scripts/train.exs --label-delay K`. The training flags `--frame-delay K`
and `--action-delay K` are compatibility aliases on all replay loaders.
The resolved default is zero. State at frame `t` receives raw Slippi controller
input at `t + 1 + K`: the parser supplies the causal successor, then the loader
adds reaction delay exactly once, per replay. Standard, streaming, BPTT,
validation, and mixed-frame loading use this same value.

Explicit aliases within one source must agree, even when one is zero.
Precedence is CLI > YAML > preset > resume metadata > zero. A higher-precedence
delay overrides all lower aliases together. Conflicting aliases in a lower
source are still rejected rather than hidden by an override.

`Config.defaults/0` leaves delay aliases unset (`nil`) so defaults cannot look
like explicit choices; `Config.parse_args/1` and `LabelDelay.resolve!/1` resolve
all three keys. Library callers should resolve raw keyword options before use.
If modifying an already resolved configuration, update all three keys together
or drop the aliases and resolve a new `label_delay`.

Resume without an explicit delay inherits the checkpoint's reaction delay.
Legacy unstamped delay `d` converts to reaction `d - 1` once. A leaked legacy
delay-zero checkpoint requires an explicit causal choice. Full resume retains
the newly resolved pipeline delay instead of restoring stale alias values.
Exports stamp the canonical delay, causal convention, and executed delay set;
comparability keys consume that canonical value.

Live Dolphin `--frame-delay N` is unchanged: it corresponds to reaction `N - 1`.
Multi-delay drill conditioning and its legacy delay IDs are not renumbered.
Mixed drill exports should be re-exported at the matching reaction delay;
existing warnings for legacy/mismatched exports remain.

Delay augmentation remains an additional jitter on top of the base delay,
supported only by the non-temporal standard loader; temporal, streaming, and
corpus modes reject it rather than ignore it. Nonzero runtime delay on a
precomputed corpus is rejected: its labels must be rebuilt instead.

## Validation scope

The regression suite includes a tiny GRU training step, standalone policy export,
fresh-process real-replay CLI evaluation, carry/reset and chunk-tail equivalence,
windowed input contracts, diagnostics, CLI/YAML/resume resolution, per-loader
successor alignment, mixed frames, discontinuities, and metadata round trips.
These are CPU/native fixture contracts, not production gameplay validation.
