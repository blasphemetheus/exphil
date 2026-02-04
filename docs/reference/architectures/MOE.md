# Mixture of Experts (MoE)

Adaptive expert selection for specialized sub-networks.

## Overview

MoE routes each input to a subset of specialized "expert" networks based on a learned routing function. This allows larger model capacity while maintaining fast inference (only K experts active per input).

```
Input x
    │
    ▼
┌─────────────────┐
│     Router      │ → Selects top-K experts
│  (softmax gate) │
└────────┬────────┘
         │
  ┌──────┼──────┬──────┬──────┐
  ▼      ▼      ▼      ▼      ▼
┌───┐  ┌───┐  ┌───┐  ┌───┐  ┌───┐
│E1 │  │E2 │  │E3 │  │E4 │  │E5 │
└─┬─┘  └─┬─┘  └───┘  └───┘  └───┘
  │      │       (inactive)
  ▼      ▼
 weighted sum
    │
    ▼
 Output
```

## Key Benefit

8 experts with top-2 routing gives 8x capacity with only 2x compute:

| Configuration | Capacity | Active Compute |
|---------------|----------|----------------|
| Standard FFN (256 hidden) | 1x | 1x |
| MoE 8 experts, top-2 | 8x | 2x |
| MoE 16 experts, top-2 | 16x | 2x |

## Usage

### Single MoE Layer

```elixir
alias ExPhil.Networks.MoE

model = MoE.build(
  input_size: 256,
  hidden_size: 512,
  num_experts: 8,
  top_k: 2,
  routing: :top_k
)
```

### MoE Block (with residual)

```elixir
input = Axon.input("x", shape: {nil, seq_len, 256})

output = MoE.build_block(input,
  hidden_size: 256,
  num_experts: 8,
  top_k: 2
)
```

### MoE-Enhanced Backbone

```elixir
# Replace FFN layers with MoE at intervals
model = MoE.build_moe_backbone(
  embed_size: 287,
  hidden_size: 256,
  num_layers: 6,
  moe_every: 2,       # MoE every 2nd layer
  num_experts: 8,
  top_k: 2,
  backbone: :mamba    # or :attention
)
```

## Routing Strategies

| Strategy | Description | Load Balance |
|----------|-------------|--------------|
| `:top_k` | Select K highest-scoring experts | Requires aux loss |
| `:switch` | Route to single best expert | Best balance |
| `:soft` | Weighted sum (approximated) | Most expensive |
| `:hash` | Deterministic based on input | Perfect balance |

## Expert Types

| Type | Description | Best For |
|------|-------------|----------|
| `:ffn` | Standard FFN (default) | General use |
| `:glu` | Gated Linear Unit | Smoother gradients |
| `:mamba` | Mamba block | Temporal experts |

## Load Balancing

Without load balancing, routers tend to collapse to using few experts. The auxiliary loss encourages uniform utilization:

```elixir
# Compute auxiliary loss
aux_loss = MoE.compute_aux_loss(router_probs, expert_mask,
  load_balance_weight: 0.01
)

# Add to training loss
total_loss = main_loss + aux_loss
```

## Speedup Estimation

```elixir
# Estimate FLOPs reduction
speedup = MoE.estimate_speedup(
  8,    # num_experts
  2,    # top_k
  0.5   # expert_fraction (50% of model is expert layers)
)
# Returns ~1.6x speedup
```

## For Melee

Different experts can specialize on different game situations:

| Expert | Specialization |
|--------|----------------|
| Expert 1 | Neutral game (spacing, pokes) |
| Expert 2 | Advantage state (combos, edgeguards) |
| Expert 3 | Disadvantage (escape, recovery) |
| Expert 4 | Tech situations (techchase, ledge) |
| Expert 5 | Character-specific (projectiles, unique moves) |
| Expert 6 | Stage-specific (platforms, hazards) |
| Expert 7 | Defensive options (shield, roll, spot dodge) |
| Expert 8 | Aggressive options (approach, pressure) |

## Configuration

```elixir
defaults = MoE.melee_defaults()
# Returns:
# [
#   num_experts: 8,
#   top_k: 2,
#   routing: :top_k,
#   expert_type: :ffn,
#   capacity_factor: 1.25,
#   load_balance_weight: 0.01,
#   moe_every: 2
# ]
```

## Training Considerations

1. **Start with load balancing**: Use `load_balance_weight: 0.01` initially
2. **Monitor expert utilization**: Check that all experts get used
3. **Gradual capacity increase**: Start with fewer experts, scale up
4. **Expert dropout**: During training, randomly drop experts for robustness

## References

- [Switch Transformers](https://arxiv.org/abs/2101.03961) - Google, 2021
- [GShard](https://arxiv.org/abs/2006.16668) - Scaling MoE
- [Mixtral](https://arxiv.org/abs/2401.04088) - MoE for LLMs
