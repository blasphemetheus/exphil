# Attention Backbone

Sliding window self-attention for efficient local temporal modeling.

## Overview

| Property | Value |
|----------|-------|
| Type | Transformer |
| Inference | ~15ms |
| 60 FPS Ready | Borderline |
| Complexity | O(K²) where K = window |
| Val Loss* | **3.68** (best) |
| Best For | Best accuracy, training |

*From 3-epoch benchmark

## CLI Usage

```bash
mix run scripts/train_from_replays.exs \
  --temporal \
  --backbone attention \
  --window-size 60 \
  --num-layers 2 \
  --num-heads 4
```

Note: `--backbone sliding_window` is an alias for `--backbone attention`.

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `window_size` | `60` | Attention window (frames, 1 sec at 60fps) |
| `num_heads` | `4` | Number of attention heads |
| `head_dim` | `64` | Dimension per head |
| `num_layers` | `2` | Number of attention blocks |
| `ffn_dim` | `256` | Feed-forward hidden dimension |
| `dropout` | `0.1` | Dropout rate |

Output dimension = `num_heads × head_dim` (e.g., 4 × 64 = 256)

## Architecture

```
Input: [batch, seq_len, embed_size]
         │
         ▼
    ┌──────────────┐
    │ Dense → dim  │  Project to hidden_dim
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │  + Pos Enc   │  Sinusoidal positional encoding
    └──────┬───────┘
           │
           ▼
    ╔══════════════════╗
    ║  Attention Block ║ × num_layers
    ╠══════════════════╣
    ║ ┌──────────────┐ ║
    ║ │  Layer Norm  │ ║
    ║ └──────┬───────┘ ║
    ║        ▼         ║
    ║ ┌──────────────┐ ║
    ║ │ Sliding Attn │ ║  Causal + window mask
    ║ └──────┬───────┘ ║
    ║        ▼         ║
    ║     + Residual   ║
    ║        │         ║
    ║ ┌──────▼───────┐ ║
    ║ │  Layer Norm  │ ║
    ║ └──────┬───────┘ ║
    ║        ▼         ║
    ║ ┌──────────────┐ ║
    ║ │     FFN      │ ║  Linear → GELU → Linear
    ║ └──────┬───────┘ ║
    ║        ▼         ║
    ║     + Residual   ║
    ╚════════╪═════════╝
             │
             ▼
    Take last position
             │
             ▼
   Output: [batch, hidden_dim]
```

## Sliding Window Attention

Unlike full attention (O(N²)), sliding window limits each position to attend only to the last K frames:

```
Position i attends to: [max(0, i - window_size + 1), i]

Example with window_size=3:
  Position 0: [0]
  Position 1: [0, 1]
  Position 2: [0, 1, 2]
  Position 3: [1, 2, 3]
  Position 4: [2, 3, 4]
  ...
```

**Complexity:** O(K²) per position where K = window_size (typically 60)

This is ~270x more efficient than full attention for 60-frame sequences.

## Attention Mask

The mask combines:
1. **Causal mask**: Can't attend to future frames
2. **Window mask**: Can only attend to last K frames

```elixir
# Pre-computed at build time
mask = Attention.build_sliding_window_mask(seq_len, window_size)
```

## Multi-Head Attention

```
Q, K, V = split(Linear(x), num_heads)

For each head:
  scores = (Q @ K^T) / sqrt(head_dim)
  scores = scores + mask  # Apply causal + window mask
  weights = softmax(scores)
  output = weights @ V

Concat heads and project
```

## Positional Encoding

Sinusoidal encoding (not learned):

```elixir
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

Added to input embeddings to give the model position information.

## When to Use

**Use Attention when:**
- Training accuracy is the priority
- You have sufficient VRAM (2-3GB)
- Real-time isn't required (or 15ms is acceptable)
- Patterns span multiple seconds (attention captures long-range)

**Don't use Attention when:**
- Strict 60 FPS requirement (15ms is borderline)
- VRAM is limited (<2GB)
- Training speed is critical (Mamba is faster)

## Benchmark Results

Attention achieved the **best validation loss** in our benchmarks:

| Backbone | Val Loss | Speed | Why |
|----------|----------|-------|-----|
| **Attention** | **3.68** | 1.0 b/s | Global context |
| Jamba | 3.87 | 1.7 b/s | Hybrid approach |
| GRU | 4.48 | 0.2 b/s | Sequential |

Attention's ability to attend to any frame in the window (not just sequentially) helps it capture complex temporal patterns.

## Memory & Performance

| Metric | Value |
|--------|-------|
| Parameters | ~400K (2 layers, 4 heads) |
| VRAM | ~2.5GB |
| Training Speed | Fast (parallelizable) |
| Inference | ~15ms |

## Implementation

**File:** `lib/exphil/networks/attention.ex`

```elixir
def build_sliding_window_backbone(input, opts) do
  window_size = opts[:window_size] || 60
  num_heads = opts[:num_heads] || 4
  head_dim = opts[:head_dim] || 64
  hidden_dim = num_heads * head_dim

  input
  |> Axon.dense(hidden_dim, name: "attention_proj")
  |> add_positional_encoding(window_size)
  |> build_attention_layers(opts)
  |> Axon.nx(fn x -> x[[.., -1, ..]] end)
end
```

## Tuning Tips

1. **Window size**: 60 frames (1 second) is usually sufficient. Increase for characters with long recovery animations (Mewtwo, Ganon).

2. **Num heads**: 4 is a good default. More heads = more parallelism but more memory.

3. **Head dim**: 64 is standard. Smaller = faster but less expressive.

4. **Num layers**: 2 is often enough. More layers help with complex patterns but risk overfitting.

## See Also

- [ARCHITECTURES.md](ARCHITECTURES.md) - Overview of all backbones
- [JAMBA.md](JAMBA.md) - Hybrid with Mamba for better speed
- [MAMBA.md](MAMBA.md) - Faster inference, different accuracy trade-off
