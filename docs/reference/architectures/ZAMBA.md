# Zamba Architecture

**Type:** Hybrid (Mamba + Shared Attention)
**Complexity:** O(n) with periodic O(n²) attention
**60 FPS Ready:** Yes (8-12ms inference)

## Overview

Zamba combines Mamba's linear-time state space processing with periodic shared attention layers. Unlike Jamba which alternates Mamba and attention at every layer, Zamba uses a single shared attention block that's applied periodically (e.g., every 6 layers), dramatically reducing the attention overhead while maintaining global context.

## Etymology

"Zamba" is a portmanteau of "Zyphra" (the AI company that created it) and "Mamba". The architecture was introduced in the Zamba paper (2024) as a more efficient hybrid than alternating attention/Mamba.

## Architecture

```
Input → [Mamba Block] × 5 → [Shared Attention] → [Mamba Block] × 5 → [Shared Attention] → Output
                                    ↑                                        ↑
                                    └────── Same weights ──────────────────┘
```

Key insight: A single attention layer with shared weights provides sufficient global context when applied periodically. This reduces parameters significantly compared to per-layer attention.

## When to Use

**Choose Zamba when:**
- You want Mamba's speed with occasional global attention
- Memory efficiency is important (shared attention = fewer parameters)
- Sequences have some long-range dependencies but are mostly local

**Avoid Zamba when:**
- Very short sequences (< 30 frames) - attention overhead not amortized
- Pure local patterns - use plain Mamba instead

## Configuration

```bash
# Basic usage
mix run scripts/train_from_replays.exs --temporal --backbone zamba

# With custom settings
mix run scripts/train_from_replays.exs \
  --temporal \
  --backbone zamba \
  --hidden-size 256 \
  --num-layers 12 \
  --attention-interval 6
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `hidden_size` | 256 | Hidden dimension |
| `num_layers` | 6 | Number of Mamba layers |
| `attention_interval` | 6 | Apply attention every N layers |
| `num_heads` | 4 | Attention heads (in shared block) |
| `expand` | 2 | Mamba expansion factor |

## Implementation

```elixir
# lib/exphil/networks/zamba.ex
defmodule ExPhil.Networks.Zamba do
  @moduledoc """
  Zamba: Mamba with periodic shared attention.
  """

  def build(input, opts \\ []) do
    hidden_size = opts[:hidden_size] || 256
    num_layers = opts[:num_layers] || 6
    attention_interval = opts[:attention_interval] || 6

    # Build shared attention (same weights reused)
    shared_attention = build_shared_attention(hidden_size, opts)

    # Stack Mamba layers with periodic attention
    Enum.reduce(1..num_layers, input, fn layer_idx, x ->
      x = mamba_block(x, hidden_size, opts)

      if rem(layer_idx, attention_interval) == 0 do
        apply_shared_attention(x, shared_attention)
      else
        x
      end
    end)
  end
end
```

## Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Inference (30 frames) | ~10ms | Slightly slower than pure Mamba |
| Memory | O(n) + periodic O(n²) | Attention only at intervals |
| Parameters | Lower than Jamba | Shared attention weights |

## Comparison with Similar Architectures

| Architecture | Attention Frequency | Attention Type | Speed |
|--------------|---------------------|----------------|-------|
| **Zamba** | Every N layers | Shared | Fast |
| Jamba | Every other layer | Per-layer | Medium |
| Attention | Every layer | Per-layer | Slow |
| Mamba | Never | None | Fastest |

## References

- [Zamba: A Compact 7B SSM Hybrid Model](https://arxiv.org/abs/2405.18712) - Original paper
- [Zamba2 Technical Report](https://arxiv.org/abs/2410.18317) - Improved version
- [ExPhil Implementation](../../../lib/exphil/networks/zamba.ex)

## See Also

- [MAMBA.md](MAMBA.md) - Base Mamba architecture
- [JAMBA.md](JAMBA.md) - Alternative Mamba/attention hybrid
- [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md) - All architectures overview
