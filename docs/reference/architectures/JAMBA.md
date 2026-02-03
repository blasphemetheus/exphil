# Jamba Backbone

Hybrid architecture combining Mamba's efficiency with Attention's long-range capabilities.

## Overview

| Property | Value |
|----------|-------|
| Type | Hybrid (Mamba + Attention) |
| Inference | ~12ms |
| 60 FPS Ready | **Yes** |
| Complexity | O(L) |
| Val Loss* | 3.87 |
| Best For | Production with high accuracy |

*From 3-epoch benchmark

## CLI Usage

```bash
mix run scripts/train_from_replays.exs \
  --temporal \
  --backbone jamba \
  --num-layers 6 \
  --attention-every 3 \
  --hidden-size 256 \
  --window-size 60
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `num_layers` | `6` | Total number of layers |
| `attention_every` | `3` | Insert attention every N layers |
| `hidden_size` | `256` | Main model dimension |
| **Mamba options** | | |
| `state_size` | `16` | SSM latent dimension |
| `expand_factor` | `2` | Inner dim expansion |
| `conv_size` | `4` | Conv kernel size |
| **Attention options** | | |
| `num_heads` | `4` | Attention heads |
| `head_dim` | `64` | Dimension per head |
| `window_size` | `60` | Sliding window size |
| `use_sliding_window` | `true` | Use sliding vs full attention |

## Architecture

Interleaved Mamba and Attention layers:

```
Input: [batch, seq_len, embed_size]
         │
         ▼
    ┌──────────────┐
    │ Dense → dim  │
    └──────┬───────┘
         │
         ▼
    ┌─────────────┐
    │ Mamba Block │  Layer 1
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │ Mamba Block │  Layer 2
    └──────┬──────┘
           │
    ┌──────▼──────────┐
    │ Attention Block │  Layer 3 (attention_every=3)
    │    + FFN        │
    └──────┬──────────┘
           │
    ┌──────▼──────┐
    │ Mamba Block │  Layer 4
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │ Mamba Block │  Layer 5
    └──────┬──────┘
           │
    ┌──────▼──────────┐
    │ Attention Block │  Layer 6 (attention_every=3)
    │    + FFN        │
    └──────┬──────────┘
           │
           ▼
    Take last position
           │
           ▼
   Output: [batch, hidden_size]
```

## Layer Pattern

```elixir
Hybrid.layer_pattern(num_layers: 6, attention_every: 3)
# => [:mamba, :mamba, :attention, :mamba, :mamba, :attention]

Hybrid.layer_pattern(num_layers: 6, attention_every: 2)
# => [:mamba, :attention, :mamba, :attention, :mamba, :attention]
```

## Why Hybrid?

| Component | Strength | Limitation |
|-----------|----------|------------|
| **Mamba** | O(L) efficiency, local patterns | Limited global context |
| **Attention** | Global context, long-range | O(K²) cost |
| **Jamba** | Both strengths | Slightly more complex |

**Mamba layers:** Efficiently process local temporal patterns (combos, tech chases)

**Attention layers:** Capture long-range dependencies (stage positioning, stock leads)

## Attention Block Details

Each attention layer includes:

```
┌─────────────────────────────────────┐
│ Pre-norm (Layer Norm)               │
│           ↓                         │
│ Optional projection to attn_dim     │
│           ↓                         │
│ Sliding Window Self-Attention       │
│   - Causal mask                     │
│   - Window mask (last K frames)     │
│           ↓                         │
│ + Residual                          │
│           ↓                         │
│ Post-norm (Layer Norm)              │
│           ↓                         │
│ FFN (Linear → GELU → Linear, 4x)    │
│           ↓                         │
│ + Residual                          │
└─────────────────────────────────────┘
```

## Tuning `attention_every`

| Value | Mamba:Attn Ratio | Trade-off |
|-------|------------------|-----------|
| 2 | 1:1 | More attention, better long-range, slower |
| 3 | 2:1 | Balanced (recommended) |
| 4 | 3:1 | More Mamba, faster, local focus |
| 6 | 5:1 | Mostly Mamba with rare attention |

**For Melee:**
- Most patterns are local (combos, reactions) → Mamba handles well
- Some patterns are long-range (stage control, conditioning) → Attention helps
- `attention_every: 3` is a good starting point

## Benchmark Results

Jamba achieved second-best validation loss with good speed:

| Backbone | Val Loss | Speed | Inference |
|----------|----------|-------|-----------|
| Attention | 3.68 | 1.0 b/s | 15ms |
| **Jamba** | **3.87** | **1.7 b/s** | **12ms** |
| GRU | 4.48 | 0.2 b/s | 150ms |
| Mamba | 8.22 | 1.3 b/s | 8.9ms |

**Key insight:** Jamba is 70% faster training than pure attention with only 5% higher loss. For production with real-time requirements, this is an excellent trade-off.

## Memory & Performance

| Metric | Value |
|--------|-------|
| Parameters | ~1.2M (6 layers) |
| VRAM | ~1.2GB |
| Training Speed | Fast |
| Inference | ~12ms |

## When to Use

**Use Jamba when:**
- You need both speed AND accuracy
- Real-time inference is required (60 FPS)
- You want the benefits of both Mamba and Attention
- Training time is a concern (faster than attention)

**Don't use Jamba when:**
- Absolute best accuracy is required (use Attention)
- Minimal latency is critical (use pure Mamba)
- Memory is very limited (use Mamba with checkpointing)

## Configuration Examples

```bash
# Balanced (recommended)
mix run scripts/train_from_replays.exs \
  --temporal \
  --backbone jamba \
  --num-layers 6 \
  --attention-every 3

# More attention (better accuracy, slower)
mix run scripts/train_from_replays.exs \
  --temporal \
  --backbone jamba \
  --num-layers 6 \
  --attention-every 2

# More Mamba (faster, good for local patterns)
mix run scripts/train_from_replays.exs \
  --temporal \
  --backbone jamba \
  --num-layers 8 \
  --attention-every 4

# Character with long recovery (more attention helpful)
mix run scripts/train_from_replays.exs \
  --temporal \
  --backbone jamba \
  --num-layers 6 \
  --attention-every 2 \
  --window-size 90 \
  --character mewtwo
```

## Implementation

**File:** `lib/exphil/networks/hybrid.ex`

```elixir
def build_jamba_backbone(input, opts) do
  num_layers = opts[:num_layers] || 6
  attention_every = opts[:attention_every] || 3

  Enum.reduce(1..num_layers, input, fn idx, layer ->
    if rem(idx, attention_every) == 0 do
      build_attention_layer(layer, opts)
    else
      build_mamba_layer(layer, opts)
    end
  end)
  |> Axon.nx(fn x -> x[[.., -1, ..]] end)
end
```

## Origin

Named after AI21's [Jamba architecture](https://www.ai21.com/jamba) which pioneered interleaving Mamba with attention layers for efficient long-context modeling.

## Training Stability (NaN Prevention)

Jamba can be prone to NaN loss during training due to the interaction between Mamba and attention layers. If you encounter NaN losses, try these fixes in order:

### Implemented Fixes

| Fix | Status | Description | Flag/Option |
|-----|--------|-------------|-------------|
| **1. Pre-LayerNorm** | ✅ | Move norm before blocks (more stable gradients) | `--pre-norm` |
| **2. LR Warmup** | ✅ | Linear warmup for first N steps | `--warmup-steps 500` |
| **3. Gradient Clipping** | ✅ | Clip gradients to prevent explosion | `--max-grad-norm 0.5` |
| **4. Lower LR** | ✅ | Jamba needs lower LR than pure Mamba | `--learning-rate 1e-5` |
| **5. QK LayerNorm** | ✅ | Normalize Q/K before attention | `--qk-layernorm` |

### Recommended Jamba Settings

```bash
# Stable Jamba training (recommended)
mix run scripts/train_from_replays.exs \
  --temporal \
  --backbone jamba \
  --num-layers 4 \
  --attention-every 2 \
  --learning-rate 1e-5 \
  --max-grad-norm 0.5 \
  --warmup-steps 500 \
  --pre-norm \
  --qk-layernorm
```

### Potential Future Fixes

If training still diverges, these additional techniques may help:

| Fix | Complexity | Description |
|-----|------------|-------------|
| **6. Separate LRs** | Medium | Lower LR for attention layers vs Mamba (10x difference) |
| **7. Attention Dropout** | Easy | Increase dropout specifically in attention (0.2-0.3) |
| **8. Weight Init** | Medium | Xavier/Glorot for attention, smaller scale for Mamba |
| **9. Mixed Precision Off** | Easy | BF16/FP16 can destabilize attention; try FP32 |
| **10. Gradient Accumulation** | Easy | Smaller effective batch through accumulation |
| **11. Reduce Model Size** | Easy | Train smaller model first, then scale up |
| **12. Remove Attention** | Diagnostic | Test pure Mamba stack, add attention back incrementally |

### Why Jamba is Unstable

1. **Gradient scale mismatch**: Mamba layers have O(L) gradients, attention has O(L²)
2. **Interaction effects**: Mamba's selective scan + attention's softmax can amplify errors
3. **Initialization**: Default init may not be optimal for hybrid architectures
4. **Learning dynamics**: Attention converges faster than Mamba, causing imbalance

### Debugging NaN

```bash
# Enable gradient debugging (very verbose)
mix run scripts/train_from_replays.exs --backbone jamba --debug-gradients

# Check which layer causes NaN
mix run scripts/train_from_replays.exs --backbone jamba --epochs 1 --batch-size 1
```

If NaN occurs:
- First batch → initialization issue (try `--pre-norm`)
- After N steps → gradient explosion (reduce LR, add clipping)
- After epoch → learning rate too high (reduce by 10x)

## See Also

- [ARCHITECTURES.md](ARCHITECTURES.md) - Overview of all backbones
- [MAMBA.md](MAMBA.md) - Pure Mamba (faster inference)
- [ATTENTION.md](ATTENTION.md) - Pure Attention (best accuracy)
