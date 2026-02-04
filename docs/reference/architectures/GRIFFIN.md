# Griffin

Hybrid RG-LRU + Local Attention architecture from Google DeepMind.

**CLI:** `--backbone griffin`

## Overview

Griffin combines Real-Gated Linear Recurrent Units (RG-LRU) with local attention, providing:
- Simpler gating than Mamba's SSM
- Numerically stable long-range modeling
- Efficient inference with O(1) memory per token

## Architecture

```
Input [batch, seq_len, embed_size]
      │
      ▼
┌─────────────────────────────────────┐
│     RG-LRU Block (2x)               │
│  RMSNorm → RG-LRU → Residual        │
│  RMSNorm → Gated MLP → Residual     │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│     Local Attention Block (1x)      │
│  RMSNorm → LocalAttn → Residual     │
│  RMSNorm → Gated MLP → Residual     │
└─────────────────────────────────────┘
      │ (repeat pattern)
      ▼
Output [batch, hidden_size]
```

## RG-LRU (Real-Gated Linear Recurrent Unit)

```
r_t = σ(W_a x_t + b_a)           # Recurrence gate
i_t = σ(W_x x_t + b_x)           # Input gate
a_t = a^(c · r_t)                # Gated decay (a = σ(Λ), c = 8)
h_t = a_t ⊙ h_{t-1} + √(1-a_t²) ⊙ (i_t ⊙ x_t)
```

The `√(1-a_t²)` term ensures hidden state norm is preserved (like a rotation), enabling stable training at long sequences.

## Usage

```bash
mix run scripts/train_from_replays.exs \
  --temporal --backbone griffin \
  --num-layers 6 \
  --window-size 60
```

## Comparison with Mamba

| Aspect | Mamba | Griffin |
|--------|-------|---------|
| Recurrence | SSM with A,B,C,Δ | Simple gated RNN |
| Parallel scan | Required | Optional |
| Long-range | Pure recurrence | Hybrid with local attention |
| Parameters | Higher | Lower |

## References

- [Griffin Paper](https://arxiv.org/abs/2402.19427) - De et al., 2024
- Hawk: RG-LRU only variant (no local attention)
