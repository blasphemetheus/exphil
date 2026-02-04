# KAN

Kolmogorov-Arnold Networks with learnable activation functions.

**CLI:** `--backbone kan`

## Overview

Unlike MLPs with fixed activations on nodes, KAN has learnable activations on edges:

```
MLP:  y = W₂ · σ(W₁ · x)           # Fixed σ (ReLU, etc.)
KAN:  y = Σⱼ Φⱼ(xⱼ)               # Learnable Φⱼ per edge
```

Based on the Kolmogorov-Arnold representation theorem: any multivariate continuous function can be represented as compositions of univariate functions.

## Edge Activations

Each edge activation is parameterized as:
```
Φ(x) = w_base · SiLU(x) + w_spline · spline(x)
```

## Basis Function Options

| Basis | Formula | Speed | Best For |
|-------|---------|-------|----------|
| `:sine` (default) | Σ A·sin(ω·x + φ) | Fast | Periodic patterns |
| `:chebyshev` | Σ c·Tₙ(x) | Fast | Smooth functions |
| `:fourier` | Σ (a·cos + b·sin) | Medium | Frequency analysis |
| `:rbf` | Σ w·exp(-‖x-μ‖²) | Medium | Local features |

## Architecture

```
Input [batch, seq_len, embed_size]
      │
      ▼
┌─────────────────────────────────────┐
│       KAN Block                      │
│  LayerNorm → KAN Layer → Residual   │
│  LayerNorm → KAN Layer → Residual   │
└─────────────────────────────────────┘
      │ (repeat for num_layers)
      ▼
Output [batch, hidden_size]
```

## Usage

```bash
mix run scripts/train_from_replays.exs \
  --temporal --backbone kan \
  --num-layers 4
```

## Comparison with MLP

| Aspect | MLP | KAN |
|--------|-----|-----|
| Activation | Fixed on nodes | Learnable on edges |
| Interpretability | Low | High (visualizable) |
| Parameters | O(n²) | O(n²g) where g=grid |
| Best for | General tasks | Symbolic/scientific |

## For Melee

KAN's learnable activations could help model:
- Non-linear damage scaling
- Complex hitbox/hurtbox interactions
- Character-specific move properties

## References

- [KAN Paper](https://arxiv.org/abs/2404.19756) - Liu et al., 2024
- [SineKAN](https://www.frontiersin.org/articles/10.3389/frai.2024.1462952)
- [GitHub](https://github.com/KindXiaoming/pykan)
