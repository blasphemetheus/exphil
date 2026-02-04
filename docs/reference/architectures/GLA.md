# GLA Architecture

**Type:** Gated Linear Attention
**Complexity:** O(n) training and inference
**60 FPS Ready:** Yes (~9ms inference)

## Overview

GLA (Gated Linear Attention) combines the efficiency of linear attention with data-dependent gating for improved expressiveness. It's essentially "linear attention done right" - addressing the known weaknesses of vanilla linear attention while maintaining O(n) complexity.

## Etymology

**GLA** stands for **G**ated **L**inear **A**ttention. The "gated" refers to learnable gates that control information flow, similar to LSTM gates but applied to attention. This gating mechanism makes linear attention competitive with softmax attention.

## Architecture

```
Input x
    │
    ├──► Q = W_q · x            # Query
    ├──► K = W_k · x            # Key
    ├──► V = W_v · x            # Value
    ├──► G = sigmoid(W_g · x)   # Gate (data-dependent!)
    │
    ▼
┌─────────────────────────────────────────┐
│  Gated Linear Attention                 │
│                                         │
│  S_t = G_t ⊙ S_{t-1} + K_t^T V_t       │  (state update)
│  output_t = Q_t · S_t                   │  (query state)
└─────────────────────────────────────────┘
    │
    ▼
Output
```

The gate G controls how much of the previous state to retain, making the model adaptive to input content.

## When to Use

**Choose GLA when:**
- You want linear attention with better quality
- Memory efficiency is critical
- Sequences have varying "forgetting" patterns

**Avoid GLA when:**
- You need bidirectional attention
- Very short sequences (simpler architectures work fine)

## Configuration

```bash
# Basic usage
mix run scripts/train_from_replays.exs --temporal --backbone gla

# With custom settings
mix run scripts/train_from_replays.exs \
  --temporal \
  --backbone gla \
  --hidden-size 256 \
  --num-layers 6 \
  --num-heads 4
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `hidden_size` | 256 | Model dimension |
| `num_layers` | 6 | Number of GLA blocks |
| `num_heads` | 4 | Number of attention heads |
| `expand` | 2 | FFN expansion factor |

## Implementation

```elixir
# lib/exphil/networks/gla.ex
defmodule ExPhil.Networks.GLA do
  @moduledoc """
  Gated Linear Attention: Linear attention with data-dependent gating.
  """

  def build(input, opts \\ []) do
    hidden_size = opts[:hidden_size] || 256
    num_layers = opts[:num_layers] || 6

    Enum.reduce(1..num_layers, input, fn _layer, x ->
      x
      |> gated_linear_attention(hidden_size, opts)
      |> feed_forward(hidden_size, opts)
    end)
  end

  defp gated_linear_attention(x, hidden_size, opts) do
    num_heads = opts[:num_heads] || 4
    head_dim = div(hidden_size, num_heads)

    # Project to Q, K, V, G
    q = Axon.dense(x, hidden_size)
    k = Axon.dense(x, hidden_size)
    v = Axon.dense(x, hidden_size)
    g = Axon.dense(x, hidden_size) |> Axon.sigmoid()  # Gate!

    # Reshape to [batch, seq, heads, head_dim]
    # Apply gated linear attention per head
    # ...
  end
end
```

## Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Training | O(n) | Chunk-wise parallel |
| Inference | O(1) memory | Recurrent formulation |
| Quality | Near-transformer | On most tasks |

### Benchmark (RTX 4090)

| Sequence Length | Training | Inference |
|-----------------|----------|-----------|
| 30 frames | 10ms | 9ms |
| 60 frames | 18ms | 9ms |
| 120 frames | 34ms | 9ms |

## Why Linear Attention Needs Gating

Vanilla linear attention:
```
output = Q · (K^T V)  # Can precompute K^T V
```

Problem: No way to "forget" old information. The state accumulates indefinitely.

GLA solution:
```
S_t = gate_t * S_{t-1} + K_t^T V_t
```

The gate (0-1) controls retention:
- Gate ≈ 1: Remember everything (long-term memory)
- Gate ≈ 0: Forget and focus on current input

This makes the effective context window adaptive and data-dependent.

## Comparison with Related Architectures

| Architecture | Gating | Complexity | Quality |
|--------------|--------|------------|---------|
| Softmax Attention | Implicit (softmax) | O(n²) | Best |
| Linear Attention | None | O(n) | Weak |
| **GLA** | Explicit (learned) | O(n) | Near-best |
| RWKV | Exponential decay | O(n) | Near-best |

## The State Space Duality Connection

GLA is closely related to the State Space Duality (SSD) framework from Mamba-2. Both show that:

1. Linear attention ≈ State space model
2. Gating ≈ Selective state updates
3. The recurrent view enables O(1) inference

GLA can be seen as "attention-flavored SSM" while Mamba-2 is "SSM-flavored attention".

## References

- [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635) - Original paper
- [HGRN2: Gated Linear RNNs with State Expansion](https://arxiv.org/abs/2404.07904) - Related work
- [ExPhil Implementation](../../../lib/exphil/networks/gla.ex)

## See Also

- [RWKV.md](RWKV.md) - Similar linear attention approach
- [MAMBA2_SSD.md](MAMBA2_SSD.md) - State Space Duality framework
- [HGRN.md](HGRN.md) - Hierarchical gated approach
- [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md) - All architectures overview
