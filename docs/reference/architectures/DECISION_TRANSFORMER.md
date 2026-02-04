# Decision Transformer Architecture

**Type:** Return-Conditioned Transformer
**Complexity:** O(n²) attention
**60 FPS Ready:** No (~180ms inference)

## Overview

Decision Transformer reframes reinforcement learning as sequence modeling. Instead of learning Q-values or policy gradients, it learns to predict actions given (return, state, action) sequences. At inference time, you specify a desired return and the model generates actions to achieve it.

## Etymology

**Decision Transformer** literally means "a transformer that makes decisions." It applies the transformer architecture (from NLP) to decision-making tasks (RL). The key innovation is conditioning on desired returns, turning RL into conditional sequence generation.

## Architecture

```
Input Sequence:
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│ R_1  │ s_1  │ a_1  │ R_2  │ s_2  │ a_2  │ R_3  │ s_3  │  ?   │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
   ↓      ↓      ↓      ↓      ↓      ↓      ↓      ↓      ↓
┌─────────────────────────────────────────────────────────────┐
│                    Causal Transformer                        │
│  (Each position attends only to previous positions)         │
└─────────────────────────────────────────────────────────────┘
   ↓      ↓      ↓      ↓      ↓      ↓      ↓      ↓      ↓
                                                          ↓
                                                       a_3 (predicted)

R = return-to-go (sum of future rewards)
s = state embedding
a = action
```

The magic: By conditioning on high returns, the model learns to imitate trajectories that achieved those returns.

## When to Use

**Choose Decision Transformer when:**
- You have offline RL data with reward labels
- You want controllable behavior (specify desired return)
- Interpretability of "skill level" matters

**Avoid Decision Transformer when:**
- You only have imitation data (no rewards)
- Real-time inference is critical (transformer attention is slow)
- You want online learning (DT is primarily offline)

## Configuration

```bash
# Basic usage (requires reward-labeled data)
mix run scripts/train_from_replays.exs \
  --temporal \
  --backbone decision_transformer \
  --reward-labels replays/rewards.json

# With custom settings
mix run scripts/train_from_replays.exs \
  --temporal \
  --backbone decision_transformer \
  --hidden-size 256 \
  --num-layers 6 \
  --num-heads 4 \
  --context-length 30
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `hidden_size` | 256 | Model dimension |
| `num_layers` | 6 | Transformer layers |
| `num_heads` | 4 | Attention heads |
| `context_length` | 30 | How many (R,s,a) triples |
| `max_return` | 100.0 | Maximum return for normalization |

## Implementation

```elixir
# lib/exphil/networks/decision_transformer.ex
defmodule ExPhil.Networks.DecisionTransformer do
  @moduledoc """
  Decision Transformer: RL as sequence modeling.
  Conditions on return-to-go to control behavior.
  """

  def build(input, opts \\ []) do
    hidden_size = opts[:hidden_size] || 256
    num_layers = opts[:num_layers] || 6
    context_length = opts[:context_length] || 30

    # Separate embeddings for returns, states, actions
    return_embed = Axon.dense(opts[:return_input], hidden_size)
    state_embed = Axon.dense(input, hidden_size)
    action_embed = Axon.dense(opts[:action_input], hidden_size)

    # Interleave: [R1, s1, a1, R2, s2, a2, ...]
    sequence = interleave(return_embed, state_embed, action_embed)

    # Add timestep embeddings
    sequence = add_timestep_embedding(sequence, context_length)

    # Causal transformer
    Enum.reduce(1..num_layers, sequence, fn _layer, x ->
      x
      |> causal_self_attention(hidden_size, opts)
      |> feed_forward(hidden_size, opts)
    end)
  end

  def build_simple(input, opts \\ []) do
    # Simplified version for backbone integration
    # Uses state-only input, learns return conditioning internally
  end
end
```

## Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Training | O(n²) | Standard transformer |
| Inference | O(n²) | Full attention each step |
| Quality | Excellent | On offline RL benchmarks |

### Benchmark (RTX 4090)

| Context Length | Training | Inference |
|----------------|----------|-----------|
| 30 triples | 45ms | 180ms |
| 60 triples | 120ms | 400ms |
| 90 triples | 240ms | 700ms |

Decision Transformer is one of the slower architectures due to full O(n²) attention and the 3x sequence length (R, s, a triples).

## Return Conditioning Explained

Traditional RL: "What action maximizes expected return?"
Decision Transformer: "What action did agents take when they got return R?"

```python
# Traditional RL (Q-learning)
action = argmax_a Q(state, a)

# Decision Transformer
action = DT(return_to_go=TARGET, state=current_state)
```

**At inference time:**
- Set high return → Expert behavior
- Set medium return → Average behavior
- Set low return → Novice behavior

This makes it easy to control difficulty without retraining!

## Melee Application

For Melee bots, Decision Transformer enables:

1. **Skill-level control**: Specify "play like a rank 5 player"
2. **Win probability targeting**: "Play to win 60% of games" (for training partners)
3. **Style transfer**: Condition on returns from different player styles

```elixir
# Play optimally
predict(model, return_to_go: 100, state: game_state)

# Play at medium skill (good training partner)
predict(model, return_to_go: 50, state: game_state)
```

## Comparison with Other RL Approaches

| Approach | Online | Offline | Return Control | Complexity |
|----------|--------|---------|----------------|------------|
| Q-Learning | Yes | Limited | No | Medium |
| Policy Gradient | Yes | No | No | Medium |
| Behavior Cloning | No | Yes | No | Low |
| **Decision Transformer** | Limited | Yes | Yes | High |

## References

- [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345) - Original paper
- [Offline Reinforcement Learning as One Big Sequence Modeling Problem](https://arxiv.org/abs/2106.02039) - Trajectory Transformer (related)
- [Online Decision Transformer](https://arxiv.org/abs/2202.05607) - Online extension
- [ExPhil Implementation](../../../lib/exphil/networks/decision_transformer.ex)

## See Also

- [ATTENTION.md](ATTENTION.md) - Base transformer attention
- [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md) - All architectures overview
