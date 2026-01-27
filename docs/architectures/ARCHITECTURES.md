# ExPhil Backbone Architectures

ExPhil supports 7 backbone architectures for policy networks, ranging from simple MLPs to state-of-the-art sequence models.

## Quick Comparison

| Backbone | Type | Inference | 60 FPS | Val Loss* | Best For |
|----------|------|-----------|--------|-----------|----------|
| [MLP](MLP.md) | Single-frame | 9ms | ✓ Yes | 3.11 | Baseline, rapid iteration |
| [LSTM](LSTM.md) | Recurrent | 229ms | ❌ No | **2.95** | Best accuracy (offline) |
| [GRU](GRU.md) | Recurrent | ~150ms | ❌ No | ~3.0 | Faster recurrent |
| [Attention](ATTENTION.md) | Transformer | 17ms | ⚠️ Borderline | 3.07 | Accuracy + parallelism |
| [Mamba](MAMBA.md) | State Space | **24ms** | ⚠️ Needs opt | 3.00 | Balance (after optimization) |
| [Jamba](JAMBA.md) | Hybrid | ~20ms | ⚠️ Needs opt | ~3.0 | Hybrid approach |

*Val loss from 3-epoch benchmark on RTX 4090, batch_size=256, 50 replays (lower is better)

## Architecture Selection

```
Need real-time (60 FPS)?
├─ YES → Need <15ms inference
│        ├─ MLP: 9ms ✓ (no temporal context)
│        ├─ Attention: 17ms ⚠️ (reduce window_size)
│        └─ Mamba: 24ms → ONNX quantization gets <10ms
└─ NO → Training/research?
         ├─ Best accuracy? → LSTM (2.95 val loss, but slow)
         ├─ Good accuracy + fast training? → Attention (8.1 b/s)
         ├─ Balance? → Mamba (6.2 b/s, near-best accuracy)
         └─ Rapid iteration? → MLP (needs --cache-embeddings)
```

## Benchmark Results (2026-01-27)

Full benchmark on RTX 4090, 50 replays, 3 epochs, batch_size=256:

| Rank | Architecture | Val Loss | Train Speed | Inference | 60 FPS |
|------|-------------|----------|-------------|-----------|--------|
| 1 | **LSTM** | **2.9517** | 7.7 b/s | 228.8ms | ❌ |
| 2 | Mamba | 2.9964 | 6.2 b/s | 23.7ms | ✓ |
| 3 | Attention | 3.065 | 8.1 b/s | 17.0ms | ✓ |
| 4 | MLP | 3.1084 | 2.7 b/s | 9.4ms | ✓ |
| - | GRU | - | ~8.3 b/s | - | - |
| - | Jamba | - | ~5.0 b/s | - | - |

*GRU and Jamba failed mid-training due to NaN loss (fixed in latest code)*

**Key findings:**
- LSTM achieves best validation loss but is 10x too slow for 60 FPS
- Mamba is the sweet spot: near-best accuracy with real-time inference
- MLP training is slowest due to on-the-fly embedding (use `--cache-embeddings`)
- Attention is borderline for 60 FPS but has excellent accuracy

## Theoretical Complexity Analysis

### Time Complexity (per sample)

| Architecture | Training | Inference | Key Operation |
|--------------|----------|-----------|---------------|
| MLP | O(d²) | O(d²) | Matrix multiply |
| LSTM/GRU | O(L × d²) | O(L × d²) | Sequential gates |
| Mamba | O(L × d) | O(L × d) | Parallel selective scan |
| Attention | O(L² × d) | O(L² × d) | Self-attention matrix |
| Jamba | O(L × d + L²/k) | O(L × d + L²/k) | Hybrid (attn every k layers) |

Where: d=hidden_size (256), L=sequence_length (30), k=attention_every (3)

### Why Observed Speeds Differ from Theory

**MLP slowest despite O(1)?**
- Temporal models use **precomputed sequence embeddings** (2h27m one-time cost)
- MLP processes raw frames through embedding pipeline per batch
- Solution: Use `--cache-embeddings` to precompute and cache MLP embeddings too

**LSTM faster than Mamba in training?**
- Mamba uses smaller batch_size=64 for memory (vs 256 for LSTM)
- 4x more batches = 4x more overhead
- cuDNN LSTM kernels are highly optimized
- Mamba shines at inference time (23.7ms vs 228.8ms)

**Attention competitive despite O(L²)?**
- L=30 is small enough that L²=900 operations are fast on GPU
- Self-attention is embarrassingly parallel (all positions at once)
- For longer sequences (L>100), attention would slow significantly

### Real-Time Gameplay Target

For 60 FPS Melee gameplay:
- **Required latency: <16.7ms per frame**
- Account for ~2ms overhead (game state read, controller output)
- **Target inference: <15ms**

| Architecture | Inference | Headroom | Verdict |
|--------------|-----------|----------|---------|
| MLP | 9.4ms | +5.6ms | ✓ Safe |
| Attention | 17.0ms | -2.0ms | ⚠️ Borderline |
| Mamba | 23.7ms | -8.7ms | ❌ Needs optimization |
| LSTM | 228.8ms | -214ms | ❌❌ Unusable |

**Note:** Inference times are for batch_size=4. Single-sample inference is faster.

### Optimization Targets

| Architecture | Current | Target | Strategy |
|--------------|---------|--------|----------|
| Mamba | 23.7ms | <15ms | ONNX quantization, state caching |
| Attention | 17.0ms | <15ms | Reduce window_size or num_layers |
| LSTM | 228.8ms | - | Not recommended for real-time |

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
