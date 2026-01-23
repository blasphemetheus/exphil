# GRU Backbone

Gated Recurrent Unit - a simpler, faster alternative to LSTM.

## Overview

| Property | Value |
|----------|-------|
| Type | Recurrent |
| Inference | ~150ms |
| 60 FPS Ready | No |
| Complexity | O(L) |
| Val Loss* | 4.48 |
| Best For | Reliable temporal baseline |

*From 3-epoch benchmark (best among recurrent models)

## CLI Usage

```bash
mix run scripts/train_from_replays.exs \
  --temporal \
  --backbone gru \
  --hidden-size 256 \
  --num-layers 2 \
  --window-size 60
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `hidden_size` | `256` | GRU cell hidden dimension |
| `num_layers` | `2` | Stacked GRU layers |
| `dropout` | `0.1` | Dropout between layers |
| `window_size` | `60` | Sequence length (frames) |
| `truncate_bptt` | `nil` | Truncated backprop (nil = full) |

## Architecture

```
Input: [batch, seq_len, embed_size]
         │
         ▼
    ┌────────┐
    │ GRU 1  │ → processes all frames sequentially
    └────┬───┘
         │ [batch, seq_len, hidden_size]
         ▼
    ┌─────────┐
    │ Dropout │
    └────┬────┘
         │
         ▼
    ┌────────┐
    │ GRU 2  │
    └────┬───┘
         │
         ▼
    Take last timestep
         │
         ▼
   Output: [batch, hidden_size]
```

## GRU Cell

GRU uses 2 gates (vs LSTM's 3), with a single hidden state:

```
Gates:
  reset_gate  = σ(W_r · [h_{t-1}, x_t])
  update_gate = σ(W_z · [h_{t-1}, x_t])

Hidden update:
  h_candidate = tanh(W · [reset_gate * h_{t-1}, x_t])
  h_t = (1 - update_gate) * h_{t-1} + update_gate * h_candidate
```

**Key insight:** The update gate controls how much of the old state to keep vs replace, similar to LSTM's forget and input gates combined.

## Why GRU Over LSTM?

| Advantage | Explanation |
|-----------|-------------|
| 25% fewer parameters | 2 gates instead of 3 |
| Faster training | Simpler computation |
| Similar accuracy | For most sequence tasks |
| Single hidden state | Easier to manage |

In our benchmarks, GRU achieved **better validation loss** (4.48) than LSTM (4.75) with faster training.

## Truncated BPTT

Same as LSTM - limits gradient flow for long sequences:

```bash
# Truncate gradients to last 15 frames
mix run scripts/train_from_replays.exs \
  --temporal \
  --backbone gru \
  --truncate-bptt 15
```

## Stateful Inference

```elixir
# Initialize hidden state (single tensor, not tuple)
hidden = Recurrent.initial_hidden(1, hidden_size: 256, cell_type: :gru)

# Single-frame inference
{output, new_hidden} = Recurrent.step(model, frame, hidden)
```

## When to Use

**Use GRU when:**
- You want a reliable temporal baseline
- LSTM is too slow and Mamba isn't converging
- Comparing with recurrent-based prior work
- Memory is constrained (fewer params than LSTM)

**Don't use GRU when:**
- Real-time inference required (150ms > 16.6ms)
- Training speed is critical (Mamba is 6x faster)
- You want best accuracy (Attention is better)

## Benchmark Comparison

| Backbone | Val Loss | Training Time | Inference |
|----------|----------|---------------|-----------|
| **GRU** | **4.48** | 2.6h | 150ms |
| LSTM | 4.75 | 3h | 220ms |
| Mamba | 8.22 | 48min | 8.9ms |
| Attention | 3.68 | 31min | 15ms |

GRU is the best recurrent model but still slower than attention-based approaches.

## Memory & Performance

| Metric | Value |
|--------|-------|
| Parameters | ~150K (256 hidden, 2 layers) |
| VRAM | ~400MB |
| Training Speed | Moderate |
| Inference | ~150ms |

## Implementation

**File:** `lib/exphil/networks/recurrent.ex`

```elixir
def build_gru_backbone(input, opts) do
  hidden_size = opts[:hidden_size] || 256
  num_layers = opts[:num_layers] || 2

  Enum.reduce(1..num_layers, input, fn idx, layer ->
    layer
    |> Axon.gru(hidden_size, name: "gru_#{idx}")
    |> elem(0)
    |> Axon.dropout(rate: opts[:dropout])
  end)
  |> Axon.nx(fn x -> x[[.., -1, ..]] end)
end
```

## See Also

- [ARCHITECTURES.md](ARCHITECTURES.md) - Overview of all backbones
- [LSTM.md](LSTM.md) - More expressive but slower
- [MAMBA.md](MAMBA.md) - Recommended for production
