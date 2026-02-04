# xLSTM

Extended Long Short-Term Memory with exponential gating and matrix memory.

**CLI:** `--backbone xlstm`, `--backbone xlstm_slstm`, `--backbone xlstm_mlstm`

## Overview

xLSTM addresses three LSTM limitations:
1. **Inability to revise storage** → Exponential gating
2. **Limited storage capacity** → Matrix memory (mLSTM)
3. **Lack of parallelizability** → mLSTM covariance update

## Two Variants

### sLSTM (Scalar LSTM)

- Exponential gating: `i_t = exp(W_i x_t + R_i h_{t-1} + b_i)`
- Normalizer state prevents overflow
- Sequential processing with memory mixing
- Good for **state-tracking** tasks

### mLSTM (Matrix LSTM)

- Matrix memory cell: `C_t = f_t ⊙ C_{t-1} + i_t ⊙ (v_t k_t^T)`
- Key-value storage similar to attention
- Fully parallelizable during training
- Good for **memorization** tasks

## Architecture

```
Input [batch, seq_len, embed_size]
      │
      ▼
┌─────────────────────────────────────┐
│         xLSTM Block                  │
│  LayerNorm → sLSTM/mLSTM            │
│  LayerNorm → Feedforward            │
│  Residual Connection                │
└─────────────────────────────────────┘
      │ (repeat for num_layers)
      ▼
Output [batch, hidden_size]
```

## Usage

```bash
# sLSTM only (state tracking)
mix run scripts/train_from_replays.exs --temporal --backbone xlstm_slstm

# mLSTM only (memorization)
mix run scripts/train_from_replays.exs --temporal --backbone xlstm_mlstm

# Mixed (alternating, default)
mix run scripts/train_from_replays.exs --temporal --backbone xlstm
```

## For Melee

- **sLSTM**: Good for tracking opponent habits and combo state
- **mLSTM**: Good for memorizing common patterns and frame data
- **Mixed**: Best of both worlds

## References

- [xLSTM Paper](https://arxiv.org/abs/2405.04517) - Beck et al., NeurIPS 2024
- [Official Code](https://github.com/NX-AI/xlstm)
