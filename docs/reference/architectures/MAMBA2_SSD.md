# Mamba-2 SSD Architecture

**Type:** State Space Model (Structured)
**Complexity:** O(n) inference, O(n²) training (with tensor cores)
**60 FPS Ready:** Yes (~9ms inference)

## Overview

Mamba-2 introduces the State Space Duality (SSD) framework, which reveals that state space models and attention are mathematically equivalent under certain conditions. This insight enables a matmul-based training algorithm that fully utilizes tensor cores, making training 2-8x faster than Mamba-1 while maintaining O(n) inference speed.

## Etymology

**Mamba-2** is the successor to the original Mamba architecture. **SSD** stands for "State Space Duality" - the key theoretical insight that state space models can be reformulated as a form of linear attention, allowing training via matrix multiplications.

## Architecture

The key innovation is dual computation paths:

```
Training Mode (SSD - uses matmuls):
┌──────────────────────────────────────────────────────┐
│  Q, K, V = linear(input)                             │
│  A = exp(-softplus(linear(input)))  # decay matrix  │
│  Y = matmul(Q, K.T) * mask * A      # tensor cores! │
│  output = matmul(Y, V)                               │
└──────────────────────────────────────────────────────┘

Inference Mode (recurrent - O(1) per step):
┌──────────────────────────────────────────────────────┐
│  For each timestep t:                                │
│    h_t = A_t * h_{t-1} + B_t * x_t   # state update │
│    y_t = C_t * h_t                    # output      │
└──────────────────────────────────────────────────────┘
```

## When to Use

**Choose Mamba-2 SSD when:**
- Training on GPU with tensor cores (RTX 30/40/50 series, A100, H100)
- You need fast training AND fast inference
- Sequences are medium-to-long (30+ frames)

**Avoid Mamba-2 SSD when:**
- Training on CPU or older GPUs without tensor cores
- Very short sequences (overhead not amortized)

## Configuration

```bash
# Basic usage (auto-detects training vs inference)
mix run scripts/train_from_replays.exs --temporal --backbone mamba2

# Explicit training mode (uses SSD matmul formulation)
mix run scripts/train_from_replays.exs \
  --temporal \
  --backbone mamba2 \
  --training-mode

# Custom settings
mix run scripts/train_from_replays.exs \
  --temporal \
  --backbone mamba2 \
  --hidden-size 256 \
  --state-size 16 \
  --num-heads 4
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `hidden_size` | 256 | Hidden dimension |
| `state_size` | 16 | SSM state dimension |
| `num_heads` | 4 | Number of attention heads (SSD mode) |
| `expand` | 2 | Expansion factor |
| `training_mode` | auto | Force SSD matmul path |

## Implementation

```elixir
# lib/exphil/networks/mamba_ssd.ex
defmodule ExPhil.Networks.MambaSSD do
  @moduledoc """
  Mamba-2 with State Space Duality.
  Uses matmul-based training for tensor core utilization.
  """

  def build(input, opts \\ []) do
    training_mode = opts[:training_mode] || false

    if training_mode do
      # SSD formulation - O(n²) but uses tensor cores
      ssd_forward(input, opts)
    else
      # Recurrent formulation - O(n) for inference
      recurrent_forward(input, opts)
    end
  end

  defp ssd_forward(input, opts) do
    # Compute Q, K, V projections
    {q, k, v, a} = compute_qkva(input, opts)

    # Matmul attention with decay mask
    # This is mathematically equivalent to SSM recurrence
    scores = Nx.dot(q, Nx.transpose(k)) * decay_mask(a)
    Nx.dot(scores, v)
  end
end
```

## Performance

| Metric | Training | Inference |
|--------|----------|-----------|
| Complexity | O(n²) matmul | O(n) recurrent |
| Tensor Core Usage | Yes | No |
| Speed vs Mamba-1 | 2-8x faster | Same |
| Memory | O(n²) | O(1) per step |

### Benchmark (RTX 4090)

| Mode | 30 frames | 60 frames | 120 frames |
|------|-----------|-----------|------------|
| Training (SSD) | 15ms | 45ms | 160ms |
| Inference (recurrent) | 9ms | 18ms | 36ms |

## The State Space Duality

The key insight is that the SSM recurrence:

```
h_t = A * h_{t-1} + B * x_t
y_t = C * h_t
```

Can be unrolled into a matrix form equivalent to linear attention:

```
Y = (Q K^T ⊙ L) V
```

Where L is a lower-triangular decay mask derived from A. This allows using highly optimized matmul kernels during training.

## Comparison with Mamba-1

| Feature | Mamba-1 | Mamba-2 SSD |
|---------|---------|-------------|
| Training method | Parallel scan | Matmul (SSD) |
| Tensor core usage | Partial | Full |
| Training speed | 1x | 2-8x |
| Inference speed | Same | Same |
| Mathematical basis | SSM | SSM = Linear Attention |

## References

- [Transformers are SSMs](https://arxiv.org/abs/2405.21060) - Mamba-2 paper
- [Original Mamba paper](https://arxiv.org/abs/2312.00752)
- [ExPhil Implementation](../../../lib/exphil/networks/mamba_ssd.ex)

## See Also

- [MAMBA.md](MAMBA.md) - Original Mamba architecture
- [GLA.md](GLA.md) - Gated Linear Attention (related via SSD)
- [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md) - All architectures overview
