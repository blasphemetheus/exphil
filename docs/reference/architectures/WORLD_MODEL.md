# World Model

Learned environment model for planning and imagination.

## Overview

The world model predicts how the game state evolves given an action:

```
(state, action) → WorldModel → (next_state, reward, done)
```

This enables:
- **Model Predictive Control (MPC)**: Plan optimal action sequences
- **Imagination-based learning**: Train policy on imagined trajectories
- **Uncertainty estimation**: Ensemble disagreement for exploration

## Architecture

```
State [batch, state_dim]    Action [batch, action_dim]
         │                           │
         └─────────┬─────────────────┘
                   │
                   ▼
            ┌─────────────┐
            │   Encoder   │
            │  (MLP x N)  │
            └──────┬──────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
   ┌────────┐ ┌────────┐ ┌────────┐
   │ State  │ │ Reward │ │  Done  │
   │  Head  │ │  Head  │ │  Head  │
   └───┬────┘ └───┬────┘ └───┬────┘
       │          │          │
       ▼          ▼          ▼
  next_state   reward      done
```

## Usage

### Basic World Model

```elixir
alias ExPhil.Networks.WorldModel

model = WorldModel.build(
  state_dim: 287,
  action_dim: 13,
  hidden_size: 512,
  num_layers: 3,
  predict_reward: true,
  predict_done: true,
  residual_prediction: true
)

{init_fn, predict_fn} = Axon.build(model)
params = init_fn.(templates, Axon.ModelState.empty())

output = predict_fn.(params, %{
  "state" => current_state,
  "action" => action
})
# output.next_state, output.reward, output.done
```

### Ensemble World Model

```elixir
ensemble = WorldModel.build_ensemble(
  state_dim: 287,
  action_dim: 13,
  num_models: 5
)

# Get uncertainty from ensemble disagreement
{mean_pred, std_pred} = ensemble.aggregate_fn.(predictions, :mean_std)
```

### Recurrent World Model

```elixir
model = WorldModel.build_recurrent(
  state_dim: 287,
  action_dim: 13,
  hidden_size: 256,
  rnn_type: :gru
)
# Maintains hidden state across predictions
```

## Melee Configuration

```elixir
defaults = WorldModel.melee_defaults()
# Returns:
# [
#   state_dim: 287,
#   action_dim: 13,
#   hidden_size: 512,
#   num_layers: 3,
#   predict_reward: true,
#   predict_done: false,  # Melee doesn't end from single actions
#   residual_prediction: true
# ]
```

## Model Predictive Control

```elixir
# Find optimal action sequence to reach goal
actions = WorldModel.mpc(
  world_model,
  current_state,
  goal_state,
  horizon: 10,
  num_samples: 100
)
```

## Rollout (Imagination)

```elixir
# Imagine trajectory from starting state
{states, rewards, dones} = WorldModel.rollout(
  world_model,
  params,
  initial_state,
  actions,  # sequence of actions
  horizon: 20
)
```

## Training

```elixir
# Compute loss for world model training
loss = WorldModel.compute_loss(predictions, targets,
  state_weight: 1.0,
  reward_weight: 0.1,
  done_weight: 0.1
)
```

## For Melee

World models are particularly useful for:

| Use Case | Benefit |
|----------|---------|
| Combo planning | Simulate hit sequences |
| Recovery | Plan optimal recovery path |
| Edgeguarding | Predict opponent recovery |
| Tech chasing | Anticipate tech options |
| Spacing | Simulate approach outcomes |

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `:state_dim` | 287 | Input state dimension |
| `:action_dim` | 13 | Action dimension |
| `:hidden_size` | 512 | Hidden layer size |
| `:num_layers` | 3 | Number of encoder layers |
| `:predict_reward` | true | Include reward prediction |
| `:predict_done` | true | Include termination prediction |
| `:residual_prediction` | true | Predict state delta, not absolute |

## Residual Prediction

When `residual_prediction: true`, the model predicts:

```
next_state = current_state + model_output
```

This helps with:
- Numerical stability
- Learning identity for no-change scenarios
- Faster convergence

## References

- [World Models](https://worldmodels.github.io/) - Ha & Schmidhuber, 2018
- [Dreamer](https://arxiv.org/abs/1912.01603) - Hafner et al., 2019
- [IRIS](https://arxiv.org/abs/2209.00588) - Autoregressive world models
