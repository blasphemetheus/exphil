# MLP Backbone

Simple feedforward neural network that processes frames independently.

## Overview

| Property | Value |
|----------|-------|
| Type | Single-frame |
| Inference | 1-2ms |
| 60 FPS Ready | Yes |
| Complexity | O(1) |
| Best For | Baseline, rapid iteration, debugging |

The MLP backbone does not model temporal dependencies. When used with `--temporal`, it extracts only the last frame from each sequence.

## CLI Usage

```bash
# Basic MLP (no temporal)
mix run scripts/train_from_replays.exs \
  --hidden-sizes 256,256 \
  --epochs 10

# MLP with temporal flag (uses last frame only)
mix run scripts/train_from_replays.exs \
  --temporal \
  --backbone mlp \
  --hidden-sizes 256,256
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `hidden_sizes` | `[512, 512]` | List of hidden layer dimensions |
| `activation` | `:relu` | Activation function (`:relu`, `:gelu`, `:silu`) |
| `dropout` | `0.1` | Dropout between layers |
| `layer_norm` | `false` | Apply layer normalization |
| `residual` | `false` | Skip connections between layers |

## Architecture

```
Input: [batch, embed_size]  (or [batch, seq_len, embed_size] → take last)
         │
         ▼
    ┌─────────┐
    │ Dense 1 │ embed_size → hidden_sizes[0]
    └────┬────┘
         │
    ┌────▼────┐
    │  ReLU   │
    └────┬────┘
         │
    ┌────▼────┐
    │ Dropout │
    └────┬────┘
         │
    ┌────▼────┐
    │ Dense 2 │ hidden_sizes[0] → hidden_sizes[1]
    └────┬────┘
         │
        ...
         │
         ▼
   Output: [batch, hidden_sizes[-1]]
```

## Residual Connections

When `--residual` is enabled:

```elixir
# If dimensions match:
output = activation(dense(input)) + input

# If dimensions differ:
output = activation(dense(input)) + project(input)
```

This enables training deeper networks (4+ layers) without gradient degradation.

```bash
# Deep MLP with residuals
mix run scripts/train_from_replays.exs \
  --hidden-sizes 256,256,256,256 \
  --residual \
  --layer-norm
```

## When to Use

**Use MLP when:**
- Rapid prototyping and debugging
- Baseline comparison for temporal models
- Training data is limited (fewer parameters to overfit)
- Inference speed is critical and temporal modeling isn't needed

**Don't use MLP when:**
- Actions depend on recent history (combos, reactions)
- You need to model opponent behavior patterns
- Playing against human-level opponents

## Code Example

```elixir
# Build MLP policy
Policy.build(
  embed_size: 1204,
  hidden_sizes: [256, 256],
  activation: :relu,
  dropout: 0.1,
  residual: true
)

# Output size
Policy.backbone_output_size([256, 256])
# => 256
```

## Memory & Performance

| Metric | Value |
|--------|-------|
| Parameters | ~50K (256,256 layers) |
| VRAM | ~50MB |
| Training Speed | 2-3x faster than temporal |
| Inference | 1-2ms |

With embedding precomputation (`--precompute`, default), MLP training is extremely fast since embeddings are computed once per dataset.

## Implementation

**File:** `lib/exphil/networks/policy.ex`

```elixir
# Lines 552-582
defp build_backbone(input, embed_size, hidden_sizes, activation, opts) do
  Enum.reduce(hidden_sizes, {input, embed_size, 0}, fn size, {layer, prev_size, idx} ->
    layer
    |> Axon.dense(size, name: "backbone_#{idx}")
    |> apply_activation(activation)
    |> maybe_add_residual(prev_size, size, opts[:residual])
    |> Axon.dropout(rate: opts[:dropout])
  end)
end
```

## See Also

- [ARCHITECTURES.md](ARCHITECTURES.md) - Overview of all backbones
- [MAMBA.md](MAMBA.md) - Recommended temporal backbone
