# Mamba Backbone

Selective State Space Model (S6) for efficient, production-grade temporal modeling.

## Overview

| Property | Value |
|----------|-------|
| Type | State Space Model |
| Inference | **8.9ms** |
| 60 FPS Ready | **Yes** |
| Complexity | O(L) |
| Val Loss* | 8.22 |
| Best For | Production, real-time play |

*From 3-epoch benchmark (needs more epochs to converge)

## CLI Usage

```bash
mix run scripts/train_from_replays.exs \
  --temporal \
  --backbone mamba \
  --hidden-size 256 \
  --num-layers 2 \
  --state-size 16 \
  --window-size 60
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `hidden_size` | `256` | Main model dimension |
| `state_size` | `16` | SSM latent dimension (N) |
| `expand_factor` | `2` | Inner dim = hidden × expand |
| `conv_size` | `4` | Causal conv kernel size |
| `num_layers` | `2` | Number of Mamba blocks |
| `dropout` | `0.1` | Dropout rate |
| `gradient_checkpoint` | `false` | Memory-efficient training |
| `checkpoint_every` | `1` | Checkpoint interval |
| `window_size` | `60` | Sequence length |

## Architecture

```
Input: [batch, seq_len, embed_size]
         │
         ▼
    ┌──────────────┐
    │ Dense → dim  │
    └──────┬───────┘
           │
           ▼
    ╔════════════════════════════════════╗
    ║         Mamba Block × N            ║
    ╠════════════════════════════════════╣
    ║                                    ║
    ║  ┌────────────────────────────┐    ║
    ║  │       Layer Norm           │    ║
    ║  └─────────────┬──────────────┘    ║
    ║                │                   ║
    ║  ┌─────────────▼──────────────┐    ║
    ║  │  Dense → 2 × inner_size    │    ║
    ║  └──────┬─────────────┬───────┘    ║
    ║         │             │            ║
    ║    X branch      Z branch          ║
    ║         │             │            ║
    ║  ┌──────▼───────┐  ┌──▼───┐        ║
    ║  │ Causal Conv  │  │ SiLU │        ║
    ║  │    + SiLU    │  └──┬───┘        ║
    ║  └──────┬───────┘     │            ║
    ║         │             │            ║
    ║  ┌──────▼───────┐     │            ║
    ║  │ Selective SSM │    │            ║
    ║  └──────┬───────┘     │            ║
    ║         │             │            ║
    ║         └─────×───────┘            ║
    ║               │                    ║
    ║        ┌──────▼───────┐            ║
    ║        │  Dense → dim │            ║
    ║        └──────┬───────┘            ║
    ║               │                    ║
    ║           + Residual               ║
    ╚═══════════════╪════════════════════╝
                    │
                    ▼
           Take last position
                    │
                    ▼
          Output: [batch, hidden_size]
```

## Selective State Space Model (S6)

The core innovation: input-dependent state transitions.

### Traditional SSM (S4)
```
h(t) = A · h(t-1) + B · x(t)
y(t) = C · h(t)

A, B, C are fixed parameters
```

### Selective SSM (S6 / Mamba)
```
h(t) = A(x) · h(t-1) + B(x) · x(t)
y(t) = C(x) · h(t)

A, B, C are computed FROM the input x
```

This "selectivity" allows the model to:
- **Remember** important information (high dt)
- **Forget** irrelevant context (low dt)
- **Focus** on specific input features

### Parameter Flow

```
x: [batch, seq_len, inner_size]
         │
    ┌────▼────┐
    │ x_proj  │ → B, C parameters
    └────┬────┘
         │
    ┌────▼────┐
    │ dt_proj │ → discretization step
    └────┬────┘
         │
    ┌────▼────┐
    │   SSM   │ → selective state update
    └─────────┘

dt (delta): Controls how much current input vs hidden state matters
  - High dt: Focus on current input
  - Low dt: Rely on accumulated state
```

## Why Mamba for Melee?

1. **60 FPS ready**: 8.9ms inference fits in 16.6ms frame budget
2. **Linear complexity**: O(L) vs O(L²) for attention
3. **Selective memory**: Can learn to "remember" relevant game states
4. **Efficient training**: Parallelizable unlike RNNs

## Gradient Checkpointing

For large models or limited VRAM:

```bash
mix run scripts/train_from_replays.exs \
  --temporal \
  --backbone mamba \
  --num-layers 6 \
  --gradient-checkpoint \
  --checkpoint-every 2
```

| Setting | VRAM | Training Speed |
|---------|------|----------------|
| No checkpoint | 2.5GB | Baseline |
| Every layer | 0.8GB | -30% |
| Every 2 layers | 1.2GB | -15% |

## Convergence Notes

In our 3-epoch benchmark, Mamba achieved higher validation loss (8.22) than attention-based models. This is likely due to:

1. **Slower convergence**: SSMs often need more epochs
2. **Hyperparameter sensitivity**: May need tuning for Melee-specific patterns
3. **Local vs global**: Mamba excels at local patterns; Melee may need more global context

**Recommendations:**
- Train for more epochs (10-20+)
- Try Jamba for hybrid approach
- Increase state_size for more expressiveness

## Memory & Performance

| Metric | Value |
|--------|-------|
| Parameters | ~500K (256 hidden, 2 layers) |
| VRAM | ~800MB |
| VRAM (checkpointed) | ~300MB |
| Training Speed | Fast (6x vs recurrent) |
| Inference | **8.9ms** |

## Comparison

| Model | Inference | 60 FPS | Memory |
|-------|-----------|--------|--------|
| **Mamba** | **8.9ms** | **Yes** | 800MB |
| Jamba | 12ms | Yes | 1.2GB |
| Attention | 15ms | Borderline | 2.5GB |
| LSTM | 220ms | No | 500MB |

## Stateful Inference

For real-time play, maintain SSM state between frames:

```elixir
# Initialize state
state = Mamba.initial_state(batch_size: 1, hidden_size: 256, state_size: 16)

# Single-frame inference
{output, new_state} = Mamba.step(model, frame, state)
```

## Implementation

**File:** `lib/exphil/networks/mamba.ex`

```elixir
def build_mamba_block(input, opts) do
  hidden = opts[:hidden_size] || 256
  inner = hidden * (opts[:expand_factor] || 2)
  state_size = opts[:state_size] || 16

  # Project to 2x inner (x and z branches)
  projected = Axon.dense(input, 2 * inner)

  # Split into x and z
  {x, z} = split_branches(projected)

  # X branch: conv + SSM
  x = x
  |> causal_conv1d(opts[:conv_size])
  |> silu()
  |> selective_ssm(state_size)

  # Z branch: gating
  z = silu(z)

  # Combine and project out
  Nx.multiply(x, z)
  |> Axon.dense(hidden)
end
```

## Tuning Tips

1. **state_size**: Start with 16. Increase to 32 for more expressiveness (more params).

2. **expand_factor**: 2 is standard. Increase for more capacity.

3. **num_layers**: 2-4 for most tasks. More layers help with complex patterns.

4. **Training epochs**: Plan for 10-20+ epochs. Mamba converges slower than attention.

5. **Learning rate**: May need lower LR than attention (try 1e-4 instead of 3e-4).

## See Also

- [ARCHITECTURES.md](ARCHITECTURES.md) - Overview of all backbones
- [JAMBA.md](JAMBA.md) - Hybrid Mamba + Attention
- [Mamba Paper](https://arxiv.org/abs/2312.00752) - Original research
