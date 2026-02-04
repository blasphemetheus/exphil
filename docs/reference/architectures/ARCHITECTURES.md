# ExPhil Backbone Architectures

ExPhil supports 15 backbone architectures for policy networks, ranging from simple MLPs to state-of-the-art sequence models.

> **New to neural network architectures?** See [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md) for beginner-friendly explanations of what each architecture is, why it's named that way, and when to use it.

## Quick Comparison

### Production Backbones (Benchmarked)

| Backbone | Type | Inference | 60 FPS | Val Loss* | Best For |
|----------|------|-----------|--------|-----------|----------|
| [MLP](MLP.md) | Single-frame | 9ms | ✓ Yes | 3.11 | Baseline, rapid iteration |
| [LSTM](LSTM.md) | Recurrent | 229ms | ❌ No | **2.95** | Best accuracy (offline) |
| [GRU](GRU.md) | Recurrent | ~150ms | ❌ No | ~3.0 | Faster recurrent |
| [Attention](ATTENTION.md) | Transformer | 17ms | ⚠️ Borderline | 3.07 | Accuracy + parallelism |
| [Mamba](MAMBA.md) | State Space | **24ms** | ⚠️ Needs opt | 3.00 | Balance (after optimization) |
| [Jamba](JAMBA.md) | Hybrid | ~20ms | ⚠️ Needs opt | ~3.0 | Hybrid approach |

### New Architectures (2026-02)

| Backbone | Type | Complexity | Best For |
|----------|------|------------|----------|
| [Zamba](ZAMBA.md) | Hybrid SSM | O(L) | Shared attention efficiency |
| [Mamba-2 SSD](MAMBA2_SSD.md) | State Space | O(L) | Tensor core training |
| [RWKV-7](RWKV.md) | Linear RNN | O(L) | O(1) memory inference |
| [GLA](GLA.md) | Linear Attention | O(L) | Short sequence speed |
| [HGRN-2](HGRN.md) | Gated RNN | O(L) | Hierarchical patterns |
| [Griffin](GRIFFIN.md) | RG-LRU + Attention | O(L) | Simple gating, stable training |
| [xLSTM](XLSTM.md) | Extended LSTM | O(L) | Exponential gating, matrix memory |
| [RetNet](RETNET.md) | Retention | O(L²)/O(1) | Parallel training, O(1) inference |
| [KAN](KAN.md) | Learnable activations | O(L) | Interpretable, symbolic patterns |
| [Decision Transformer](DECISION_TRANSFORMER.md) | Return-conditioned | O(L²) | Goal-directed behavior |
| [S5](S5.md) | Simplified SSM | O(L) | MIMO state space |
| [Liquid](LIQUID.md) | Neural ODE | O(L×steps) | Adaptive dynamics |
| [GatedSSM](GATED_SSM.md) | Simple SSM | O(L) | Fast baseline |

See also: [Policy Types](../POLICY_TYPES.md) for non-backbone policy architectures (Diffusion, ACT, Flow Matching).

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
| Zamba | O(L × d + L²) | O(L × d + L²) | Shared attention reuse |
| Mamba-2 SSD | O(L × d) | O(L × d) | SSD matmul / Blelloch scan |
| RWKV-7 | O(L × d) | O(d) per step | WKV linear attention |
| GLA | O(L × d) | O(L × d) | Gated linear attention |
| HGRN-2 | O(L × d) | O(L × d) | Hierarchical gating |
| Decision Transformer | O(L² × d) | O(L² × d) | Causal self-attention |
| S5 | O(L × N²) | O(L × N²) | MIMO state space |
| Liquid | O(L × d × s) | O(L × d × s) | ODE integration (s=steps) |

Where: d=hidden_size (256), L=sequence_length (30), k=attention_every (3), N=state_size, s=integration_steps

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

### FAQ: Is Flash Attention a Separate Architecture?

**No.** Flash Attention is an **optimization technique**, not a backbone architecture.

It optimizes how attention is computed without changing what is computed:
- **Standard attention**: Materializes full n×n attention matrix → O(n²) memory
- **Flash attention**: Computes in tiles without full matrix → O(n) memory

Flash attention would be an option for the existing `Attention` backbone:
```bash
# Future flag (not yet implemented)
mix run scripts/train_from_replays.exs --backbone attention --flash-attention
```

Benefits for Attention backbone:
- Enable 120+ frame sequences (vs ~90 currently)
- 30-50% memory reduction
- Same accuracy (mathematically equivalent)

See [GPU_OPTIMIZATIONS.md](../GPU_OPTIMIZATIONS.md#flash-attention-medium-priority) for implementation status.

## CLI Usage

```bash
# Enable temporal processing with a specific backbone
mix run scripts/train_from_replays.exs \
  --temporal \
  --backbone mamba \
  --window-size 60 \
  --num-layers 2

# Available backbones (15 total)
# --- Production (benchmarked) ---
--backbone mlp              # MLP (temporal extracts last frame)
--backbone lstm             # LSTM recurrent
--backbone gru              # GRU recurrent
--backbone attention        # Sliding window attention
--backbone sliding_window   # Same as attention
--backbone mamba            # Mamba SSM (recommended)
--backbone jamba            # Hybrid Mamba + Attention

# --- New architectures (2026-02) ---
--backbone zamba            # Shared attention + Mamba hybrid
--backbone mamba_ssd        # Mamba-2 with SSD algorithm
--backbone rwkv             # RWKV-7 linear RNN
--backbone gla              # Gated Linear Attention
--backbone hgrn             # Hierarchical Gated RNN
--backbone decision_transformer  # Return-conditioned transformer
--backbone s5               # Simplified State Space
--backbone liquid           # Liquid Neural Networks (ODE-based)
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
| LSTM | 500MB | **13GB+** | O(L) | No |
| GRU | 400MB | **13GB+** | O(L) | No |
| Attention | 2.5GB | **13GB+** | O(K²) | Yes |
| Mamba | 800MB | **13GB+** | O(L) | Partial |
| Jamba | 1.2GB | **15GB+** | O(L) | Partial |
| Zamba | 1.0GB | **13GB+** | O(L) | Partial |
| Mamba-2 SSD | 800MB | **13GB+** | O(L) | Yes (training) |
| RWKV-7 | 600MB | **13GB+** | O(L) | Partial |
| GLA | 700MB | **13GB+** | O(L) | Yes |
| HGRN-2 | 600MB | **13GB+** | O(L) | Partial |
| Decision Transformer | 1.5GB | **13GB+** | O(L²) | Yes |
| S5 | 500MB | **13GB+** | O(L) | Yes |
| Liquid | 800MB | **13GB+** | O(L×s) | Partial |

L = sequence length, K = window size (typically 60), s = ODE integration steps

**⚠️ RAM Warning for Temporal Architectures:**

The benchmark script pre-builds all sequence embeddings for efficiency. With 50 replays (~380K sequences):
- Each sequence: 30 frames × 287 dims × 4 bytes = 34 KB
- Total: 380K × 34 KB ≈ **13 GB RAM**

If your machine has <16 GB RAM, use `--max-files 25` or fewer replays.

MLP avoids this by using frame embeddings directly (no sequence pre-building).

## Individual Architecture Docs

### Production Backbones
- [MLP](MLP.md) - Simple feedforward, single-frame processing
- [LSTM](LSTM.md) - Long Short-Term Memory recurrent
- [GRU](GRU.md) - Gated Recurrent Unit
- [Attention](ATTENTION.md) - Sliding window self-attention
- [Mamba](MAMBA.md) - Selective State Space Model
- [Jamba](JAMBA.md) - Hybrid Mamba + Attention

### New Architectures (2026-02)

- [Zamba](ZAMBA.md) - Shared attention Mamba hybrid
- [Mamba-2 SSD](MAMBA2_SSD.md) - State Space Duality for tensor cores
- [RWKV-7](RWKV.md) - Linear RNN with O(1) inference memory
- [GLA](GLA.md) - Gated Linear Attention
- [HGRN-2](HGRN.md) - Hierarchical Gated RNN
- [Decision Transformer](DECISION_TRANSFORMER.md) - Return-conditioned transformer
- [S5](S5.md) - Simplified State Space
- [Liquid](LIQUID.md) - Neural ODE with adaptive dynamics
- [GatedSSM](GATED_SSM.md) - Simple gated state space (baseline)

## Implementation Files

| Backbone | Primary File | Policy Integration |
|----------|-------------|-------------------|
| MLP | `policy.ex` | `build_backbone/5` |
| LSTM | `recurrent.ex` | `build_recurrent_backbone/3` |
| GRU | `recurrent.ex` | `build_recurrent_backbone/3` |
| Attention | `attention.ex` | `build_sliding_window_backbone/2` |
| Mamba | `mamba.ex` | `build_mamba_backbone/2` |
| Jamba | `hybrid.ex` | `build_jamba_backbone/2` |
| Zamba | `zamba.ex` | `build_zamba_backbone/2` |
| Mamba-2 SSD | `mamba_ssd.ex` | `build_mamba_ssd_backbone/2` |
| RWKV-7 | `rwkv.ex` | `build_rwkv_backbone/2` |
| GLA | `gla.ex` | `build_gla_backbone/2` |
| HGRN-2 | `hgrn.ex` | `build_hgrn_backbone/2` |
| Decision Transformer | `decision_transformer.ex` | `build_decision_transformer_backbone/2` |
| S5 | `s5.ex` | `build_s5_backbone/2` |
| Liquid | `liquid.ex` | `build_liquid_backbone/2` |

All in `lib/exphil/networks/`.

## New Architecture Details

### Zamba - Shared Attention Hybrid

Uses 6 Mamba blocks with a single shared attention layer called at configurable intervals.

```bash
mix run scripts/train_from_replays.exs --backbone zamba --attention-interval 3
```

### Mamba-2 SSD - Training Optimized

Dual-mode operation: SSD matmul for training (tensor cores), Blelloch scan for inference.

```bash
# Training mode (automatic)
mix run scripts/train_from_replays.exs --backbone mamba_ssd

# Inference uses fast scan automatically
```

### RWKV-7 - Linear Complexity RNN

Time-mixing + channel-mixing blocks with O(1) memory inference.

```bash
mix run scripts/train_from_replays.exs --backbone rwkv --num-layers 4
```

### GLA - Gated Linear Attention

Data-dependent gating on linear attention for short-sequence efficiency.

```bash
mix run scripts/train_from_replays.exs --backbone gla --num-heads 4
```

### HGRN-2 - Hierarchical Gating

Multi-resolution temporal modeling with state expansion.

```bash
mix run scripts/train_from_replays.exs --backbone hgrn --expand-ratio 2
```

### Decision Transformer - Goal-Conditioned

Return-conditioned transformer for target-driven behavior.

```bash
mix run scripts/train_from_replays.exs --backbone decision_transformer
# Note: Requires return-to-go in training data for full functionality
```

### S5 - Simplified State Space

Single MIMO SSM for ablation studies comparing to Mamba's SISO approach.

```bash
mix run scripts/train_from_replays.exs --backbone s5 --state-size 64
```

### Liquid - Neural ODE

Continuous-time dynamics with multiple ODE solvers.

```bash
# Default (RK4 solver)
mix run scripts/train_from_replays.exs --backbone liquid

# High accuracy (DOPRI5 adaptive)
mix run scripts/train_from_replays.exs --backbone liquid --solver dopri5
```

Available solvers: `:euler`, `:midpoint`, `:rk4`, `:dopri5`

## Advanced Features

### HybridBuilder - Custom Hybrid Architectures

Flexible builder for combining different layer types in arbitrary patterns.

```elixir
alias ExPhil.Networks.HybridBuilder

# Custom pattern
pattern = [:mamba, :mamba, :attention, :gla, :mamba, :ffn]
model = HybridBuilder.build(pattern, embed_size: 287)

# Predefined patterns
model = HybridBuilder.build_pattern(:mamba_gla, 6, embed_size: 287)
```

See [HYBRID_BUILDER.md](HYBRID_BUILDER.md) for full documentation.

### Speculative Decoding - Fast Inference

Use a fast draft model to propose actions, verify with accurate target model.

```elixir
alias ExPhil.Networks.SpeculativeDecoding

decoder = SpeculativeDecoding.create(
  draft_fn: &mlp_predict/2,
  target_fn: &mamba_batch_predict/2,
  lookahead: 4
)

{actions, count, decoder} = SpeculativeDecoding.generate(...)
```

See [SPECULATIVE_DECODING.md](SPECULATIVE_DECODING.md) for full documentation.

### Mixture of Experts (MoE) - Adaptive Expert Selection

Route inputs to specialized expert networks for increased capacity.

```elixir
alias ExPhil.Networks.MoE

# Replace FFN layers with MoE
model = MoE.build_moe_backbone(
  embed_size: 287,
  num_layers: 6,
  moe_every: 2,
  num_experts: 8,
  top_k: 2,
  backbone: :mamba
)
```

See [MOE.md](MOE.md) for full documentation.

### World Model - Planning & Imagination

Predict future states for model-based planning.

```elixir
alias ExPhil.Networks.WorldModel

model = WorldModel.build(
  state_dim: 287,
  action_dim: 13,
  predict_reward: true,
  residual_prediction: true
)

# Model predictive control
actions = WorldModel.mpc(world_model, state, goal, horizon: 10)
```

See [WORLD_MODEL.md](WORLD_MODEL.md) for full documentation.
