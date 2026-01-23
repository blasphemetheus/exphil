# ExPhil Backbone Architectures

ExPhil supports 7 backbone architectures for policy networks, ranging from simple MLPs to state-of-the-art sequence models.

## Quick Comparison

| Backbone | Type | Inference | 60 FPS | Val Loss* | Best For |
|----------|------|-----------|--------|-----------|----------|
| [MLP](MLP.md) | Single-frame | 1-2ms | Yes | - | Baseline, rapid iteration |
| [LSTM](LSTM.md) | Recurrent | 220ms | No | 4.75 | Legacy, research |
| [GRU](GRU.md) | Recurrent | 150ms | No | 4.48 | Faster recurrent |
| [Attention](ATTENTION.md) | Transformer | 15ms | Borderline | **3.68** | Best accuracy |
| [Mamba](MAMBA.md) | State Space | **8.9ms** | **Yes** | 8.22 | Production (60 FPS) |
| [Jamba](JAMBA.md) | Hybrid | 12ms | Yes | 3.87 | Accuracy + speed |

*Val loss from 3-epoch benchmark on RTX 4090 (lower is better)

## Architecture Selection

```
Need real-time (60 FPS)?
├─ YES → Mamba or Jamba
│        ├─ Best accuracy? → Jamba (3.87 val loss)
│        └─ Fastest? → Mamba (8.9ms)
└─ NO → Training/research?
         ├─ Best accuracy? → Attention (3.68 val loss)
         ├─ Temporal baseline? → GRU (reliable, well-understood)
         └─ Rapid iteration? → MLP (fastest training)
```

## Benchmark Results (2026-01-22)

Full benchmark on RTX 4090, 64GB RAM, 3 epochs each:

| Rank | Architecture | Val Loss | Speed | Training Time |
|------|-------------|----------|-------|---------------|
| 1 | **Attention** | **3.68** | 1.0 b/s | 31min |
| 2 | Jamba | 3.87 | 1.7 b/s | 72min |
| 3 | GRU | 4.48 | 0.2 b/s | 2.6h |
| 4 | LSTM | 4.75 | 0.2 b/s | 3h |
| 5 | Mamba | 8.22 | 1.3 b/s | 48min |

**Key findings:**
- Attention achieves best validation loss while being 5x faster than GRU
- Jamba (hybrid) is nearly as good as pure attention but 70% faster
- Mamba needs more epochs to converge but has fastest inference

## CLI Usage

```bash
# Enable temporal processing with a specific backbone
mix run scripts/train_from_replays.exs \
  --temporal \
  --backbone mamba \
  --window-size 60 \
  --num-layers 2

# Available backbones
--backbone mlp           # MLP (temporal extracts last frame)
--backbone lstm          # LSTM recurrent
--backbone gru           # GRU recurrent
--backbone attention     # Sliding window attention
--backbone sliding_window # Same as attention
--backbone mamba         # Mamba SSM (recommended)
--backbone jamba         # Hybrid Mamba + Attention
```

## Training Presets

```bash
# Quick iteration (MLP, no temporal)
mix run scripts/train_from_replays.exs --preset gpu_mlp_quick

# Standard training (Mamba)
mix run scripts/train_from_replays.exs --preset gpu_standard

# Production quality (Mamba + all features)
mix run scripts/train_from_replays.exs --preset production
```

## Memory & Compute Requirements

| Backbone | VRAM | RAM | Complexity | Parallelizable |
|----------|------|-----|------------|----------------|
| MLP | 50MB | 1GB | O(1) | Yes |
| LSTM | 500MB | 2GB | O(L) | No |
| GRU | 400MB | 2GB | O(L) | No |
| Attention | 2.5GB | 4GB | O(K²) | Yes |
| Mamba | 800MB | 2GB | O(L) | Partial |
| Jamba | 1.2GB | 3GB | O(L) | Partial |

L = sequence length, K = window size (typically 60)

## Individual Architecture Docs

- [MLP](MLP.md) - Simple feedforward, single-frame processing
- [LSTM](LSTM.md) - Long Short-Term Memory recurrent
- [GRU](GRU.md) - Gated Recurrent Unit
- [Attention](ATTENTION.md) - Sliding window self-attention
- [Mamba](MAMBA.md) - Selective State Space Model
- [Jamba](JAMBA.md) - Hybrid Mamba + Attention

## Implementation Files

| Backbone | Primary File | Policy Integration |
|----------|-------------|-------------------|
| MLP | `policy.ex:552-582` | `build_backbone/5` |
| LSTM | `recurrent.ex:183-211` | `build_recurrent_backbone/3` |
| GRU | `recurrent.ex:502-519` | `build_recurrent_backbone/3` |
| Attention | `attention.ex:333-409` | `build_sliding_window_backbone/2` |
| Mamba | `mamba.ex:97-150` | `build_mamba_backbone/2` |
| Jamba | `hybrid.ex:118-204` | `build_jamba_backbone/2` |

All in `lib/exphil/networks/`.
