# RetNet

Retentive Network - A Successor to Transformer with O(1) inference.

**CLI:** `--backbone retnet`

## Overview

RetNet replaces attention with "retention" - a decay-based mechanism that supports three computation modes with the same weights:

| Mode | Time | Memory | Best For |
|------|------|--------|----------|
| Parallel | O(L²) | O(L²) | Training |
| Recurrent | O(1) | O(1) | Inference |
| Chunkwise | O(L) | O(C) | Long sequences |

## Retention Mechanism

```
Parallel:   Y = (Q ⊙ Θ) · D · (K ⊙ Θ)^T · V
Recurrent:  s_n = γ·s_{n-1} + K_n^T·V_n; o_n = Q_n·s_n
```

Where D is a decay matrix: `D[n,m] = γ^(n-m) if n≥m, else 0`

## Multi-Scale Retention (MSR)

Different heads use different decay rates for multi-scale modeling:
- `γ_h = 1 - 2^(-5-h)` for head h
- GroupNorm instead of LayerNorm
- SiLU gating: `Y = SiLU(X·W_G) ⊙ Retention(X)·W_O`

## Architecture

```
Input [batch, seq_len, embed_size]
      │
      ▼
┌─────────────────────────────────────┐
│       RetNet Block                   │
│  LayerNorm → MSR → Residual         │
│  LayerNorm → FFN → Residual         │
└─────────────────────────────────────┘
      │ (repeat for num_layers)
      ▼
Output [batch, hidden_size]
```

## Usage

```bash
mix run scripts/train_from_replays.exs \
  --temporal --backbone retnet \
  --num-layers 6 \
  --num-heads 4
```

## For Melee

RetNet's decay-based memory is well-suited for:
- Gradual forgetting of old frame data
- Fast O(1) inference for 60 FPS gameplay
- Efficient handling of reaction windows

## References

- [RetNet Paper](https://arxiv.org/abs/2307.08621) - Sun et al., Microsoft 2023
- [Official Code](https://github.com/microsoft/unilm/tree/master/retnet)
