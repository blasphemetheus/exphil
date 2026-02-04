# Speculative Decoding

Fast inference using draft model speculation and target model verification.

## Overview

Speculative decoding accelerates inference by using a small, fast "draft" model to propose multiple actions that a larger "target" model can verify in parallel.

```
Draft Model (MLP, ~9ms)     Target Model (Mamba, ~24ms)
       │                            │
       ▼                            │
  [a1, a2, a3, a4] (proposed)       │
       │                            │
       └──────────► Parallel ◄──────┘
                    Verify
                       │
                       ▼
              Accept [a1, a2] ✓
              Reject [a3, a4] ✗
                       │
                       ▼
              Return [a1, a2, target_sample]
```

## Key Insight

Verifying K actions in parallel is often faster than generating K actions sequentially:

| Approach | Time for 4 actions |
|----------|-------------------|
| Sequential (Mamba) | 4 × 24ms = 96ms |
| Speculative (50% accept) | 4×9ms + 24ms = 60ms |

## Usage

```elixir
alias ExPhil.Networks.SpeculativeDecoding

# Create decoder
decoder = SpeculativeDecoding.create(
  draft_fn: &draft_model_predict/2,
  target_fn: &target_model_batch_predict/2,
  lookahead: 4,
  acceptance_threshold: 0.8
)

# Generate actions
{actions, acceptance_count, decoder} =
  SpeculativeDecoding.generate(
    decoder,
    draft_params,
    target_params,
    current_state
  )
```

## Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `:lookahead` | 4 | Number of actions to speculate |
| `:temperature` | 1.0 | Sampling temperature |
| `:acceptance_threshold` | 0.8 | Min probability ratio to accept |

## Speedup Estimation

```elixir
# Estimate speedup before implementing
speedup = SpeculativeDecoding.estimate_speedup(
  0.5,    # acceptance_rate
  4,      # lookahead
  9.0,    # draft_speed_ms (MLP)
  24.0    # target_speed_ms (Mamba)
)
# Returns ~1.3x speedup
```

## For Melee

Real-time 60 FPS gameplay requires <16.7ms per frame. Speculative decoding helps by:

1. **Draft model (MLP)**: Fast single-frame prediction (~9ms)
2. **Target model (Mamba)**: Accurate but slower temporal model (~24ms)
3. **Combined**: Get temporal quality with better throughput

## Statistics Tracking

```elixir
# Track acceptance rate over time
stats = SpeculativeDecoding.get_stats(decoder)
IO.puts("Acceptance rate: #{stats.acceptance_rate * 100}%")

# Reset for new session
decoder = SpeculativeDecoding.reset_stats(decoder)
```

## When to Use

| Scenario | Recommendation |
|----------|----------------|
| Single action needed | Use target directly |
| Action sequence needed | Use speculative |
| Draft much slower than target | Use target directly |
| High acceptance rate (>60%) | Speculative excels |
| Low acceptance rate (<30%) | May not help |

## Training the Draft Model

For best results, train the draft model to approximate the target:

```bash
# Train draft model with knowledge distillation
mix run scripts/train_from_replays.exs \
  --backbone mlp \
  --teacher-checkpoint target_model.axon \
  --distillation-weight 0.5
```

## Future Improvements

- **Adaptive lookahead**: Adjust based on acceptance rate
- **Parallel draft**: Generate multiple speculation branches
- **Token-level speculation**: Speculate individual controller inputs
