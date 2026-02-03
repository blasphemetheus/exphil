# LSTM Backbone

Long Short-Term Memory recurrent neural network for sequential processing.

## Overview

| Property | Value |
|----------|-------|
| Type | Recurrent |
| Inference | ~220ms |
| 60 FPS Ready | No |
| Complexity | O(L) |
| Val Loss* | 4.75 |
| Best For | Research, baseline temporal |

*From 3-epoch benchmark

## CLI Usage

```bash
mix run scripts/train_from_replays.exs \
  --temporal \
  --backbone lstm \
  --hidden-size 256 \
  --num-layers 2 \
  --window-size 60
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `hidden_size` | `256` | LSTM cell hidden dimension |
| `num_layers` | `2` | Stacked LSTM layers |
| `dropout` | `0.1` | Dropout between layers |
| `window_size` | `60` | Sequence length (frames) |
| `truncate_bptt` | `nil` | Truncated backprop (nil = full) |

## Architecture

```
Input: [batch, seq_len, embed_size]
         │
         ▼
    ┌─────────┐
    │ LSTM 1  │ → processes all frames sequentially
    └────┬────┘
         │ [batch, seq_len, hidden_size]
         ▼
    ┌─────────┐
    │ Dropout │
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │ LSTM 2  │
    └────┬────┘
         │
         ▼
    Take last timestep
         │
         ▼
   Output: [batch, hidden_size]
```

## LSTM Cell

Each LSTM cell maintains two states:
- **h** (hidden): Short-term memory
- **c** (cell): Long-term memory

```
Gates:
  forget_gate = σ(W_f · [h_{t-1}, x_t] + b_f)
  input_gate  = σ(W_i · [h_{t-1}, x_t] + b_i)
  output_gate = σ(W_o · [h_{t-1}, x_t] + b_o)

Cell update:
  c_t = forget_gate * c_{t-1} + input_gate * tanh(W_c · [h_{t-1}, x_t])
  h_t = output_gate * tanh(c_t)
```

## Truncated BPTT

For long sequences, full backpropagation through time (BPTT) can be slow and memory-intensive. Truncated BPTT limits gradient flow:

```bash
# Truncate gradients to last 15 frames
mix run scripts/train_from_replays.exs \
  --temporal \
  --backbone lstm \
  --truncate-bptt 15
```

| Setting | Memory | Training Speed | Accuracy |
|---------|--------|----------------|----------|
| Full BPTT (nil) | High | Slow | Best |
| Truncate 20 | Medium | 2x faster | -2% |
| Truncate 10 | Low | 3x faster | -5% |

## Stateful Inference

For real-time play, maintain hidden state between frames:

```elixir
# Initialize hidden state
hidden = Recurrent.initial_hidden(1, hidden_size: 256, cell_type: :lstm)
# Returns: {h, c} tensors

# Single-frame inference
{output, new_hidden} = Recurrent.step(model, frame, hidden)
```

## When to Use

**Use LSTM when:**
- You need a well-understood temporal baseline
- Debugging temporal processing issues
- Comparing against published results using LSTM

**Don't use LSTM when:**
- Real-time inference is required (220ms >> 16.6ms frame budget)
- Training time is a constraint (slow)
- You want best accuracy (Attention/Jamba are better)

## Comparison with GRU

| Property | LSTM | GRU |
|----------|------|-----|
| States | 2 (h, c) | 1 (h) |
| Gates | 3 | 2 |
| Parameters | Higher | 25% fewer |
| Speed | Slower | Faster |
| Expressiveness | Higher | Slightly lower |

For most Melee tasks, GRU performs comparably with faster training.

## Memory & Performance

| Metric | Value |
|--------|-------|
| Parameters | ~200K (256 hidden, 2 layers) |
| VRAM | ~500MB |
| Training Speed | Slow (sequential) |
| Inference | ~220ms |

## Implementation

**File:** `lib/exphil/networks/recurrent.ex`

```elixir
# Build LSTM backbone
def build_lstm_backbone(input, opts) do
  hidden_size = opts[:hidden_size] || 256
  num_layers = opts[:num_layers] || 2

  Enum.reduce(1..num_layers, input, fn idx, layer ->
    layer
    |> Axon.lstm(hidden_size, name: "lstm_#{idx}")
    |> elem(0)  # Take output, not hidden state
    |> Axon.dropout(rate: opts[:dropout])
  end)
  |> Axon.nx(fn x -> x[[.., -1, ..]] end)  # Last timestep
end
```

## See Also

- [ARCHITECTURES.md](ARCHITECTURES.md) - Overview of all backbones
- [GRU.md](GRU.md) - Faster recurrent alternative
- [MAMBA.md](MAMBA.md) - Recommended for production
