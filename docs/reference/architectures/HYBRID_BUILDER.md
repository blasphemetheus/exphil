# HybridBuilder

Flexible hybrid architecture builder for combining different layer types.

**CLI:** Not directly exposed (use for experimentation)

## Overview

Unlike fixed hybrids (Jamba, Zamba), HybridBuilder allows arbitrary combinations of layer types for architecture exploration.

```
Pattern: [:mamba, :mamba, :attention, :gla, :mamba, :ffn]
         ↓        ↓         ↓          ↓       ↓       ↓
       M → M → A → G → M → F
```

## Supported Layer Types

| Type | Module | Complexity | Best For |
|------|--------|------------|----------|
| `:mamba` | GatedSSM | O(L) | Long sequences |
| `:attention` | Attention | O(L²) | Global context |
| `:gla` | GLA | O(L) | Fast linear attention |
| `:rwkv` | RWKV | O(L) | Linear RNN |
| `:ffn` | Dense+GELU | O(1) | Feature transform |
| `:kan` | KAN | O(1) | Learnable activations |

## Usage

```elixir
alias ExPhil.Networks.HybridBuilder

# Custom hybrid pattern
pattern = [:mamba, :mamba, :attention, :mamba, :gla, :ffn]
model = HybridBuilder.build(pattern,
  embed_size: 287,
  hidden_size: 256,
  seq_len: 60
)

# Predefined patterns
model = HybridBuilder.build_pattern(:jamba_like, 6, embed_size: 287)
model = HybridBuilder.build_pattern(:mamba_gla, 6, embed_size: 287)
```

## Predefined Patterns

| Pattern | Description | Example (6 layers) |
|---------|-------------|-------------------|
| `:jamba_like` | Interleaved Mamba + Attention | M, A, M, A, M, A |
| `:zamba_like` | All Mamba (use with shared_layers) | M, M, M, M, M, M |
| `:mamba_gla` | Mamba + periodic GLA | M, M, G, M, M, G |
| `:rwkv_attention` | RWKV + sparse attention | R, R, R, A, R, R |
| `:full_hybrid` | Diverse mix | M, A, G, R, M, F |
| `:ssm_stack` | Pure SSM | M, M, M, M, M, M |

## Layer-Specific Options

```elixir
HybridBuilder.build(pattern,
  embed_size: 287,
  hidden_size: 256,

  # Mamba options
  mamba_state_size: 16,
  mamba_expand_factor: 2,

  # Attention options
  attention_num_heads: 4,
  attention_head_dim: 64,
  attention_window_size: 60,

  # GLA options
  gla_num_heads: 4,
  gla_head_dim: 64,

  # RWKV options
  rwkv_head_size: 64,

  # KAN options
  kan_grid_size: 5
)
```

## Visualization

```elixir
pattern = [:mamba, :attention, :gla, :ffn]
IO.puts(HybridBuilder.visualize(pattern))
# Layer pattern: M → A → G → F
# Legend: M=mamba, A=attention, G=gla, F=ffn
```

## Parameter Estimation

```elixir
count = HybridBuilder.param_count(pattern, embed_size: 287, hidden_size: 256)
# Returns approximate parameter count
```

## For Melee

Different layer combinations may work better for different aspects:
- **Mamba**: Efficient temporal processing
- **Attention**: Complex decision points (edgeguards, tech situations)
- **GLA**: Fast alternative to attention
- **FFN**: Pure feature transformation

## Architecture Exploration

Use HybridBuilder to explore which combinations work best:

```elixir
# Compare different hybrids
patterns = [
  {:mamba_only, [:mamba, :mamba, :mamba, :mamba]},
  {:mamba_attention, [:mamba, :attention, :mamba, :attention]},
  {:mamba_gla, [:mamba, :gla, :mamba, :gla]},
]

for {name, pattern} <- patterns do
  model = HybridBuilder.build(pattern, embed_size: 287)
  params = HybridBuilder.param_count(pattern, embed_size: 287)
  IO.puts("#{name}: #{params} params")
end
```
