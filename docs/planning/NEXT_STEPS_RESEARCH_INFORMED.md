# Research-Informed Next Steps for ExPhil

> **Purpose:** Concrete implementation roadmap based on insights from WORLD_MODELS_AND_SCALING.md and PIVOTAL_RESEARCHERS.md. Each item includes rationale, implementation details, and success criteria.

**Last Updated:** 2026-02-03

---

## Overview

These next steps are prioritized by the **Bitter Lesson filter**: does this scale with compute, or does it encode our assumptions? Items that scale get priority.

| Priority | Item | Effort | Scales? | Rationale |
|----------|------|--------|---------|-----------|
| 1 | Run Self-Play at Scale | Low | ✓ | Validate existing infrastructure |
| 2 | Sparse Reward Experiment | Low | ✓ | Test if shaped rewards help or hurt |
| 3 | Implement GRPO | Medium | ✓ | Simpler than PPO, same results |
| 4 | State Prediction Loss | Medium | ✓ | Free world model during BC |
| 5 | Hierarchical Policy | High | ✓ | Multi-timescale planning |
| 6 | MPC Planning | High | ✓ | Lookahead with world model |

---

## 1. Run Self-Play Infrastructure at Scale

### Rationale

You have complete GenServer-based self-play infrastructure (supervisor, game runners, matchmaker, Elo) but it's only been tested with short episodes and random policies. Before adding new features, validate that the existing system works at scale.

**Research basis:** AlphaStar and OpenAI Five both required extensive infrastructure debugging before meaningful training could begin. Better to find issues now.

### Current State

From `docs/planning/GOALS.md`:
- ✅ GamePoolSupervisor + GameRunner GenServers
- ✅ PopulationManager with configurable history
- ✅ Matchmaker with skill-based pairing
- ✅ Mock environment with physics
- ✅ Elo rating system (26 tests passing)

### Implementation

**Phase 1: Mock Environment Stress Test**

```bash
# Start small, verify stability
mix run scripts/train_self_play.exs \
  --game-type mock \
  --num-games 4 \
  --timesteps 10000 \
  --max-episode-frames 600 \
  --track-elo \
  --verbose

# Scale up gradually
mix run scripts/train_self_play.exs \
  --game-type mock \
  --num-games 8 \
  --timesteps 50000 \
  --max-episode-frames 1800 \
  --track-elo

# Full scale test
mix run scripts/train_self_play.exs \
  --game-type mock \
  --num-games 16 \
  --timesteps 200000 \
  --max-episode-frames 3600 \
  --track-elo \
  --checkpoint-interval 10000
```

**Phase 2: With Pretrained Policy**

```bash
# Use BC-trained policy as starting point
mix run scripts/train_self_play.exs \
  --game-type mock \
  --pretrained checkpoints/bc_mamba_mewtwo.axon \
  --num-games 8 \
  --timesteps 100000 \
  --track-elo \
  --save-best
```

**Phase 3: Dolphin Integration (if mock works)**

```bash
# Real Melee via Dolphin
source .venv/bin/activate
mix run scripts/train_self_play.exs \
  --game-type dolphin \
  --dolphin ~/.config/Slippi\ Launcher/netplay \
  --iso ~/melee.iso \
  --num-games 2 \
  --timesteps 50000
```

### Metrics to Track

| Metric | Target | Why |
|--------|--------|-----|
| Games/hour | 100+ (mock) | Throughput |
| Memory usage | Stable over time | No leaks |
| Elo variance | Increasing | Learning signal |
| Policy entropy | Decreasing then stable | Convergence |
| GenServer crashes | 0 | Stability |

### Success Criteria

- [ ] 16 parallel games run for 200k timesteps without crashes
- [ ] Elo ratings show meaningful differentiation (not all draws)
- [ ] Memory usage stable (no growth over time)
- [ ] Checkpoints save/load correctly mid-training

### Potential Issues & Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| Memory leak | RAM grows over time | Check for retained tensors in GenServer state |
| Elo stagnation | All games draw | Increase episode length, check reward signal |
| Slow throughput | <50 games/hour | Profile, check for blocking calls |
| GenServer timeout | `:timeout` errors | Increase timeout, make inference async |

---

## 2. Sparse Reward Experiment

### Rationale

**DeepSeek-R1's key finding:** Skipping supervised fine-tuning and using pure RL with sparse rewards enabled "emergent development of advanced reasoning patterns, such as self-reflection, verification, and dynamic strategy adaptation."

**The Bitter Lesson argument:** Shaped rewards encode our assumptions about "good play." With sufficient compute, the model should discover what matters from win/loss alone.

**Counter-argument:** Melee games are long (minutes), rewards are sparse (stocks). TD learning may struggle with credit assignment. This experiment quantifies the tradeoff.

### Current State

From `lib/exphil/rewards/`:
- `standard.ex` - Stock differential + damage
- `shaped.ex` - Approach, combo, edgeguard bonuses

### Implementation

**Step 1: Implement Sparse Reward Module**

```elixir
# lib/exphil/rewards/sparse.ex
defmodule ExPhil.Rewards.Sparse do
  @moduledoc """
  Pure win/loss reward signal.

  Based on DeepSeek-R1 and Sutton's reward hypothesis:
  given sufficient compute, sparse rewards should discover
  optimal behavior without encoding our assumptions.
  """

  @doc """
  Returns reward based solely on game outcome.

  ## Returns
    - `+1.0` if player won (took last stock)
    - `-1.0` if player lost
    - `0.0` otherwise (mid-game)
  """
  def compute(game_state, player_port) do
    cond do
      game_over?(game_state) and winner?(game_state, player_port) -> 1.0
      game_over?(game_state) -> -1.0
      true -> 0.0
    end
  end

  defp game_over?(%{players: players}) do
    Enum.any?(players, fn {_port, player} -> player.stock == 0 end)
  end

  defp winner?(%{players: players}, player_port) do
    case players[player_port] do
      %{stock: stock} when stock > 0 -> true
      _ -> false
    end
  end
end
```

**Step 2: Add CLI Flag**

```elixir
# In lib/exphil/training/config.ex, add to @valid_flags:
"--reward-type"

# In defaults/0:
reward_type: :shaped

# In parse/1:
defp parse_reward_type("sparse"), do: :sparse
defp parse_reward_type("shaped"), do: :shaped
defp parse_reward_type("standard"), do: :standard
```

**Step 3: Integrate into Self-Play**

```elixir
# In lib/exphil/self_play/game_runner.ex
defp compute_reward(game_state, player_port, opts) do
  case Keyword.get(opts, :reward_type, :shaped) do
    :sparse -> ExPhil.Rewards.Sparse.compute(game_state, player_port)
    :shaped -> ExPhil.Rewards.Shaped.compute(game_state, player_port)
    :standard -> ExPhil.Rewards.Standard.compute(game_state, player_port)
  end
end
```

### Experiment Protocol

**Experiment A: Sparse vs Shaped (Fixed Compute)**

```bash
# Shaped rewards (control)
mix run scripts/train_self_play.exs \
  --reward-type shaped \
  --timesteps 500000 \
  --num-games 8 \
  --seed 42 \
  --name shaped_500k

# Sparse rewards (treatment)
mix run scripts/train_self_play.exs \
  --reward-type sparse \
  --timesteps 500000 \
  --num-games 8 \
  --seed 42 \
  --name sparse_500k
```

**Experiment B: Sparse with More Compute**

```bash
# Sparse rewards with 2x compute
mix run scripts/train_self_play.exs \
  --reward-type sparse \
  --timesteps 1000000 \
  --num-games 8 \
  --seed 42 \
  --name sparse_1M
```

### Metrics

| Metric | Shaped | Sparse | Sparse 2x | Notes |
|--------|--------|--------|-----------|-------|
| Win rate vs CPU L9 | ? | ? | ? | Primary metric |
| Win rate vs BC baseline | ? | ? | ? | Improvement over imitation |
| Strategy diversity | ? | ? | ? | Action entropy per situation |
| Training time | ? | ? | ? | Wall clock |
| Sample efficiency | ? | ? | ? | Timesteps to 50% win rate |

### Success Criteria

- [ ] Both configurations train without divergence
- [ ] Clear winner emerges (or evidence they're equivalent)
- [ ] Strategy diversity measured and compared
- [ ] Results documented with statistical significance

### Expected Outcomes

**If shaped wins:** Continue using shaped rewards, but document the compute level where this was tested. May flip at higher compute.

**If sparse wins:** Major simplification. Remove reward shaping code, focus on scale.

**If equivalent:** Use sparse (simpler, fewer hyperparameters).

---

## 3. Implement GRPO (Group Relative Policy Optimization)

### Rationale

**From DeepSeek-R1 paper:** GRPO was proposed to "simplify the training process and reduce the resource consumption of PPO." It eliminates the critic network by using the group of sampled responses as a baseline.

**Key insight:** Instead of learning a value function V(s) to compute advantages, GRPO computes advantages relative to the mean reward of a batch of samples from the same state.

### How GRPO Differs from PPO

| Aspect | PPO | GRPO |
|--------|-----|------|
| Baseline | Learned value function V(s) | Mean of group rewards |
| Networks | Policy + Critic | Policy only |
| Advantage | A = R - V(s) | A = R - mean(R_group) |
| Compute | ~2x (critic forward/backward) | 1x |
| Stability | GAE smoothing | Group statistics |

### Mathematical Formulation

**PPO Advantage:**
```
A_t = R_t + γV(s_{t+1}) - V(s_t)  (with GAE smoothing)
```

**GRPO Advantage:**
```
Given state s, sample K actions {a_1, ..., a_K}
Execute each, get rewards {r_1, ..., r_K}
A_i = r_i - mean({r_1, ..., r_K})

Optionally normalize:
A_i = (r_i - mean) / (std + ε)
```

### Implementation

**Step 1: GRPO Module**

```elixir
# lib/exphil/training/grpo.ex
defmodule ExPhil.Training.GRPO do
  @moduledoc """
  Group Relative Policy Optimization.

  Simplified PPO that uses group statistics instead of a learned
  critic for advantage estimation. Based on DeepSeek-R1.

  Key differences from PPO:
  - No value network (saves ~40% compute)
  - Advantages computed from batch statistics
  - Requires multiple samples per state for stable gradients
  """

  import Nx.Defn

  @default_opts [
    clip_epsilon: 0.2,
    group_size: 8,           # Samples per state for advantage estimation
    normalize_advantages: true,
    entropy_coef: 0.01,
    max_grad_norm: 0.5
  ]

  @doc """
  Compute GRPO loss for a batch of experiences.

  ## Parameters
    - policy_params: Current policy network parameters
    - policy_fn: Function (params, states) -> action_logits
    - batch: Map with :states, :actions, :rewards, :old_log_probs
    - opts: Training options

  ## Returns
    {loss, metrics}
  """
  def compute_loss(policy_params, policy_fn, batch, opts \\ []) do
    opts = Keyword.merge(@default_opts, opts)

    %{
      states: states,
      actions: actions,
      rewards: rewards,
      old_log_probs: old_log_probs
    } = batch

    # Get current policy log probs
    logits = policy_fn.(policy_params, states)
    log_probs = compute_log_probs(logits, actions)

    # Compute group-relative advantages
    advantages = compute_group_advantages(rewards, opts)

    # PPO-style clipped objective
    ratio = Nx.exp(log_probs - old_log_probs)
    clipped_ratio = Nx.clip(ratio, 1 - opts[:clip_epsilon], 1 + opts[:clip_epsilon])

    policy_loss = -Nx.mean(
      Nx.min(
        ratio * advantages,
        clipped_ratio * advantages
      )
    )

    # Entropy bonus for exploration
    entropy = compute_entropy(logits)
    entropy_loss = -opts[:entropy_coef] * entropy

    total_loss = policy_loss + entropy_loss

    metrics = %{
      policy_loss: Nx.to_number(policy_loss),
      entropy: Nx.to_number(entropy),
      mean_advantage: Nx.to_number(Nx.mean(advantages)),
      clip_fraction: compute_clip_fraction(ratio, opts[:clip_epsilon])
    }

    {total_loss, metrics}
  end

  defnp compute_group_advantages(rewards, opts) do
    # Group-relative advantage: A_i = r_i - mean(r)
    mean_reward = Nx.mean(rewards)
    advantages = rewards - mean_reward

    if opts[:normalize_advantages] do
      std = Nx.standard_deviation(rewards)
      advantages / (std + 1.0e-8)
    else
      advantages
    end
  end

  defnp compute_log_probs(logits, actions) do
    # For autoregressive heads, sum log probs across action components
    # This is simplified - actual implementation needs per-head handling
    log_softmax = Axon.Activations.log_softmax(logits)
    Nx.take_along_axis(log_softmax, Nx.new_axis(actions, -1), axis: -1)
    |> Nx.squeeze(axes: [-1])
    |> Nx.sum(axes: [-1])
  end

  defnp compute_entropy(logits) do
    probs = Axon.Activations.softmax(logits)
    log_probs = Axon.Activations.log_softmax(logits)
    -Nx.sum(probs * log_probs) / Nx.axis_size(logits, 0)
  end

  defp compute_clip_fraction(ratio, epsilon) do
    clipped = Nx.logical_or(
      Nx.less(ratio, 1 - epsilon),
      Nx.greater(ratio, 1 + epsilon)
    )
    Nx.to_number(Nx.mean(Nx.as_type(clipped, :f32)))
  end
end
```

**Step 2: GRPO Trainer**

```elixir
# lib/exphil/training/grpo_trainer.ex
defmodule ExPhil.Training.GRPOTrainer do
  @moduledoc """
  Training loop for GRPO.

  Simpler than PPO trainer since there's no critic to train.
  """

  alias ExPhil.Training.{GRPO, Output}

  def train(policy, env, opts \\ []) do
    opts = Keyword.merge(default_opts(), opts)

    {init_fn, predict_fn} = Axon.build(policy, mode: :train)
    params = init_fn.(Nx.template({1, opts[:state_dim]}, :f32), %{})

    optimizer = Polaris.Optimizers.adam(learning_rate: opts[:learning_rate])
    opt_state = Polaris.Optimizers.init(optimizer, params)

    state = %{
      params: params,
      opt_state: opt_state,
      step: 0,
      metrics_history: []
    }

    Enum.reduce(1..opts[:num_iterations], state, fn iter, state ->
      # Collect experiences
      batch = collect_batch(state.params, predict_fn, env, opts)

      # Compute gradients and update
      {loss, grads, metrics} = compute_grads(state.params, predict_fn, batch, opts)

      {new_params, new_opt_state} = Polaris.Optimizers.update(
        optimizer,
        grads,
        state.params,
        state.opt_state
      )

      # Log progress
      if rem(iter, opts[:log_interval]) == 0 do
        Output.puts("Iter #{iter}: loss=#{Float.round(loss, 4)}, entropy=#{Float.round(metrics.entropy, 4)}")
      end

      %{state |
        params: new_params,
        opt_state: new_opt_state,
        step: state.step + 1,
        metrics_history: [metrics | state.metrics_history]
      }
    end)
  end

  defp collect_batch(params, predict_fn, env, opts) do
    # Collect opts[:batch_size] transitions
    # For GRPO, we want multiple samples per state for group statistics
    # ...
  end

  defp compute_grads(params, predict_fn, batch, opts) do
    grad_fn = fn params ->
      {loss, metrics} = GRPO.compute_loss(params, predict_fn, batch, opts)
      {loss, metrics}
    end

    {{loss, metrics}, grads} = Nx.Defn.value_and_grad(grad_fn, &elem(&1, 0)).(params)
    {Nx.to_number(loss), grads, metrics}
  end

  defp default_opts do
    [
      learning_rate: 3.0e-4,
      batch_size: 2048,
      num_iterations: 1000,
      log_interval: 10,
      state_dim: 287,
      clip_epsilon: 0.2,
      entropy_coef: 0.01
    ]
  end
end
```

**Step 3: CLI Integration**

```bash
# Add --algorithm flag
mix run scripts/train_self_play.exs \
  --algorithm grpo \
  --timesteps 100000 \
  --group-size 8
```

### Comparison Experiment

```bash
# PPO baseline
mix run scripts/train_self_play.exs \
  --algorithm ppo \
  --timesteps 200000 \
  --seed 42 \
  --name ppo_baseline

# GRPO
mix run scripts/train_self_play.exs \
  --algorithm grpo \
  --timesteps 200000 \
  --seed 42 \
  --name grpo_comparison
```

### Success Criteria

- [ ] GRPO trains without divergence
- [ ] Achieves similar win rate to PPO with less compute
- [ ] Training is ~30-40% faster (no critic updates)
- [ ] Gradient variance is manageable (may need tuning group_size)

---

## 4. State Prediction Auxiliary Loss (JEPA-Style)

### Rationale

**From LeCun's JEPA:** The key insight is predicting in representation space, not observation space. By adding a state prediction objective, we:
1. Learn better representations (encoder must capture dynamics-relevant features)
2. Get a world model "for free" during BC training
3. Enable future MPC planning

**This is self-supervised learning:** No new labels needed, just predict next state from current state + action.

### Architecture

```
Current BC Pipeline:
  state_t → Encoder → repr_t → Policy → action_t
  Loss: CE(action_t, true_action_t)

With State Prediction:
  state_t → Encoder → repr_t → Policy → action_t
                  ↓
            repr_t + action_t → Predictor → pred_repr_{t+1}
  state_{t+1} → Encoder → repr_{t+1}

  Loss: CE(action_t, true_action_t) + λ * MSE(pred_repr_{t+1}, repr_{t+1})
```

### Implementation

**Step 1: World Model Module**

```elixir
# lib/exphil/networks/world_model.ex
defmodule ExPhil.Networks.WorldModel do
  @moduledoc """
  JEPA-style world model that predicts next state representations.

  The predictor learns dynamics in representation space:
    pred_repr_{t+1} = Predictor(repr_t, action_t)

  This enables:
  1. Better representation learning (dynamics-aware)
  2. Future MPC planning (simulate action consequences)
  3. Uncertainty estimation (prediction error as confidence)
  """

  @doc """
  Build encoder + predictor architecture.

  ## Options
    - state_dim: Input state dimension (default: 287)
    - repr_dim: Representation dimension (default: 256)
    - action_dim: Action embedding dimension (default: 64)
    - predictor_hidden: Predictor hidden size (default: 512)
  """
  def build(opts \\ []) do
    state_dim = Keyword.get(opts, :state_dim, 287)
    repr_dim = Keyword.get(opts, :repr_dim, 256)
    action_dim = Keyword.get(opts, :action_dim, 64)
    predictor_hidden = Keyword.get(opts, :predictor_hidden, 512)

    # State encoder (shared for current and target states)
    encoder = build_encoder(state_dim, repr_dim)

    # Action embedder
    action_embedder = build_action_embedder(action_dim)

    # Predictor: (repr_t, action_embed) -> pred_repr_{t+1}
    predictor = build_predictor(repr_dim, action_dim, predictor_hidden)

    %{
      encoder: encoder,
      action_embedder: action_embedder,
      predictor: predictor
    }
  end

  defp build_encoder(state_dim, repr_dim) do
    Axon.input("state", shape: {nil, state_dim})
    |> Axon.dense(repr_dim * 2, activation: :gelu, name: "encoder_dense1")
    |> Axon.layer_norm(name: "encoder_ln1")
    |> Axon.dense(repr_dim, activation: :gelu, name: "encoder_dense2")
    |> Axon.layer_norm(name: "encoder_ln2")
  end

  defp build_action_embedder(action_dim) do
    # Embed discrete actions into continuous space
    # 8 buttons + 4 stick axes + 1 shoulder = 13 dims
    Axon.input("action", shape: {nil, 13})
    |> Axon.dense(action_dim, activation: :gelu, name: "action_embed")
  end

  defp build_predictor(repr_dim, action_dim, hidden_dim) do
    repr_input = Axon.input("repr", shape: {nil, repr_dim})
    action_input = Axon.input("action_embed", shape: {nil, action_dim})

    Axon.concatenate([repr_input, action_input])
    |> Axon.dense(hidden_dim, activation: :gelu, name: "predictor_dense1")
    |> Axon.layer_norm(name: "predictor_ln1")
    |> Axon.dense(hidden_dim, activation: :gelu, name: "predictor_dense2")
    |> Axon.layer_norm(name: "predictor_ln2")
    |> Axon.dense(repr_dim, name: "predictor_output")  # No activation - predict repr directly
  end

  @doc """
  Compute prediction loss between predicted and actual next representations.

  Uses cosine similarity + MSE for stable training (following V-JEPA).
  """
  def prediction_loss(pred_repr, target_repr, opts \\ []) do
    # Stop gradient on target (target encoder is EMA or frozen)
    target_repr = Nx.stop_gradient(target_repr)

    # MSE loss
    mse = Nx.mean(Nx.pow(pred_repr - target_repr, 2))

    # Cosine similarity loss (helps with scale invariance)
    pred_norm = Nx.sqrt(Nx.sum(Nx.pow(pred_repr, 2), axes: [-1], keep_axes: true) + 1.0e-8)
    target_norm = Nx.sqrt(Nx.sum(Nx.pow(target_repr, 2), axes: [-1], keep_axes: true) + 1.0e-8)
    cosine_sim = Nx.sum(pred_repr * target_repr, axes: [-1]) / (pred_norm * target_norm)
    cosine_loss = 1 - Nx.mean(cosine_sim)

    # Combined loss
    alpha = Keyword.get(opts, :cosine_weight, 0.5)
    alpha * cosine_loss + (1 - alpha) * mse
  end
end
```

**Step 2: Integrate into Training**

```elixir
# In lib/exphil/training/imitation.ex, modify train_step:

defp train_step_with_world_model(params, batch, opts) do
  %{
    states: states,          # {batch, state_dim}
    next_states: next_states, # {batch, state_dim}
    actions: actions,         # {batch, action_dim}
    targets: targets          # {batch, target_dim} (for policy)
  } = batch

  grad_fn = fn params ->
    # Forward pass through encoder
    repr = encoder_forward(params.encoder, states)
    next_repr = encoder_forward(params.encoder, next_states)

    # Policy loss (existing BC loss)
    policy_logits = policy_forward(params.policy, repr)
    policy_loss = compute_policy_loss(policy_logits, targets, opts)

    # World model loss (new)
    action_embed = action_embed_forward(params.action_embedder, actions)
    pred_repr = predictor_forward(params.predictor, repr, action_embed)
    world_model_loss = WorldModel.prediction_loss(pred_repr, next_repr)

    # Combined loss
    lambda = Keyword.get(opts, :world_model_weight, 0.1)
    total_loss = policy_loss + lambda * world_model_loss

    {total_loss, %{policy_loss: policy_loss, world_model_loss: world_model_loss}}
  end

  {{loss, metrics}, grads} = Nx.Defn.value_and_grad(grad_fn, &elem(&1, 0)).(params)
  {loss, grads, metrics}
end
```

**Step 3: CLI Flag**

```bash
mix run scripts/train_from_replays.exs \
  --temporal \
  --backbone mamba \
  --world-model \
  --world-model-weight 0.1 \
  --epochs 20
```

### Experiment: Does State Prediction Improve BC?

```bash
# Baseline (no world model)
mix run scripts/train_from_replays.exs \
  --backbone mamba \
  --epochs 20 \
  --seed 42 \
  --name bc_baseline

# With world model auxiliary loss
mix run scripts/train_from_replays.exs \
  --backbone mamba \
  --world-model \
  --world-model-weight 0.1 \
  --epochs 20 \
  --seed 42 \
  --name bc_world_model
```

### Metrics

| Metric | Baseline | With World Model |
|--------|----------|------------------|
| BC validation loss | ? | ? |
| Action accuracy | ? | ? |
| Representation quality* | ? | ? |
| Win rate vs CPU | ? | ? |

*Representation quality: linear probe accuracy on game events (stock taken, hit landed, etc.)

### Success Criteria

- [ ] World model loss decreases during training
- [ ] BC loss not significantly worse (within 5%)
- [ ] Representations transfer better to RL fine-tuning
- [ ] Enables future MPC experiments

---

## 5. Hierarchical Policy (H-JEPA Style)

### Rationale

**From LeCun's H-JEPA proposal:** Different levels of abstraction enable different prediction horizons. Melee naturally decomposes into timescales:

| Level | Abstraction | Time Horizon | Melee Context |
|-------|-------------|--------------|---------------|
| High | Strategic | 60+ frames | Win the game, control stage |
| Mid | Tactical | 10-60 frames | Win the exchange, execute combo |
| Low | Execution | 1-10 frames | Input the technique |

**Why this helps:**
- High-level policy sets goals, doesn't worry about execution
- Mid-level policy translates goals into action sequences
- Low-level policy handles frame-perfect inputs
- Each level operates at appropriate temporal resolution

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Hierarchical Policy                       │
│                                                             │
│  Every 60 frames:                                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  High-Level Policy                                   │   │
│  │  Input: game_state (stocks, percent, positions)      │   │
│  │  Output: strategic_goal (64-dim embedding)           │   │
│  │  Examples: "edgeguard", "neutral_control", "recover" │   │
│  └──────────────────────────┬──────────────────────────┘   │
│                             │                               │
│  Every 10 frames:           ▼                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Mid-Level Policy                                    │   │
│  │  Input: game_state + strategic_goal                  │   │
│  │  Output: tactical_goal (64-dim embedding)            │   │
│  │  Examples: "approach_aerial", "shield_pressure"      │   │
│  └──────────────────────────┬──────────────────────────┘   │
│                             │                               │
│  Every frame:               ▼                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Low-Level Policy                                    │   │
│  │  Input: game_state + tactical_goal                   │   │
│  │  Output: controller_action (buttons, sticks)         │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Implementation

**Step 1: Hierarchical Policy Module**

```elixir
# lib/exphil/networks/hierarchical_policy.ex
defmodule ExPhil.Networks.HierarchicalPolicy do
  @moduledoc """
  H-JEPA-style hierarchical policy with three temporal levels.

  - High-level: Strategic decisions (every 60 frames)
  - Mid-level: Tactical decisions (every 10 frames)
  - Low-level: Frame-by-frame execution

  Goals are represented as learned embeddings, not discrete labels.
  This allows the model to discover its own goal space.
  """

  @high_level_interval 60
  @mid_level_interval 10
  @goal_dim 64

  def build(opts \\ []) do
    state_dim = Keyword.get(opts, :state_dim, 287)
    hidden_dim = Keyword.get(opts, :hidden_dim, 256)

    %{
      high_level: build_high_level(state_dim, hidden_dim),
      mid_level: build_mid_level(state_dim, hidden_dim),
      low_level: build_low_level(state_dim, hidden_dim)
    }
  end

  defp build_high_level(state_dim, hidden_dim) do
    # Strategic policy: state -> goal embedding
    Axon.input("state", shape: {nil, state_dim})
    |> Axon.dense(hidden_dim, activation: :gelu)
    |> Axon.layer_norm()
    |> Axon.dense(hidden_dim, activation: :gelu)
    |> Axon.dense(@goal_dim, name: "strategic_goal")  # Goal embedding
  end

  defp build_mid_level(state_dim, hidden_dim) do
    # Tactical policy: state + strategic_goal -> tactical_goal
    state_input = Axon.input("state", shape: {nil, state_dim})
    goal_input = Axon.input("strategic_goal", shape: {nil, @goal_dim})

    Axon.concatenate([state_input, goal_input])
    |> Axon.dense(hidden_dim, activation: :gelu)
    |> Axon.layer_norm()
    |> Axon.dense(hidden_dim, activation: :gelu)
    |> Axon.dense(@goal_dim, name: "tactical_goal")
  end

  defp build_low_level(state_dim, hidden_dim) do
    # Execution policy: state + tactical_goal -> action
    state_input = Axon.input("state", shape: {nil, state_dim})
    goal_input = Axon.input("tactical_goal", shape: {nil, @goal_dim})

    Axon.concatenate([state_input, goal_input])
    |> Axon.dense(hidden_dim, activation: :gelu)
    |> Axon.layer_norm()
    |> Axon.dense(hidden_dim, activation: :gelu)
    |> build_controller_heads()  # Existing autoregressive heads
  end

  @doc """
  Forward pass with goal caching.

  Goals are only recomputed at their designated intervals.
  """
  def forward(model, params, state, cache, frame) do
    # Update high-level goal every 60 frames
    strategic_goal = if rem(frame, @high_level_interval) == 0 or cache.strategic_goal == nil do
      predict_high_level(model.high_level, params.high_level, state)
    else
      cache.strategic_goal
    end

    # Update mid-level goal every 10 frames
    tactical_goal = if rem(frame, @mid_level_interval) == 0 or cache.tactical_goal == nil do
      predict_mid_level(model.mid_level, params.mid_level, state, strategic_goal)
    else
      cache.tactical_goal
    end

    # Always compute low-level action
    action = predict_low_level(model.low_level, params.low_level, state, tactical_goal)

    new_cache = %{
      strategic_goal: strategic_goal,
      tactical_goal: tactical_goal,
      last_frame: frame
    }

    {action, new_cache}
  end

  defp predict_high_level(model, params, state) do
    Axon.predict(model, params, %{"state" => state})
  end

  defp predict_mid_level(model, params, state, strategic_goal) do
    Axon.predict(model, params, %{
      "state" => state,
      "strategic_goal" => strategic_goal
    })
  end

  defp predict_low_level(model, params, state, tactical_goal) do
    Axon.predict(model, params, %{
      "state" => state,
      "tactical_goal" => tactical_goal
    })
  end
end
```

**Step 2: Hierarchical Training**

Training hierarchical policies is challenging. Options:

**Option A: End-to-End (simplest)**
Train all levels together with a single reward signal. Gradients flow through all levels.

**Option B: Feudal (hierarchical reward)**
High-level gets game outcome reward. Mid/low-level get intrinsic reward for achieving goals set by higher levels.

**Option C: Pre-train then fine-tune**
1. Pre-train low-level on BC (learn execution)
2. Pre-train mid-level on BC with goal labels derived from game events
3. Fine-tune all levels together with RL

### Success Criteria

- [ ] Hierarchical policy trains end-to-end
- [ ] High-level goals show interpretable clustering (visualize with t-SNE)
- [ ] Performance improves on long-horizon tasks (edgeguarding, tech chasing)
- [ ] Inference latency still under 16.7ms (goal caching helps)

---

## 6. MPC Planning with World Model

### Rationale

**From V-JEPA 2-AC:** Instead of reactive policy execution, use the world model for lookahead planning:

1. Generate K candidate action sequences
2. Simulate N steps with world model
3. Score trajectories with value function
4. Execute first action of best trajectory
5. Repeat (receding horizon)

**Why this helps for Melee:**
- Can "pre-react" to opponent actions by simulating possibilities
- Better for situations requiring planning (edgeguards, tech chases)
- Naturally handles uncertainty (simulate multiple futures)

### Architecture

```
At each decision point:

1. Current state s_t
          │
          ▼
2. Generate K=64 action sequences of length N=10
   [{a_0, a_1, ..., a_9}_1, ..., {a_0, ..., a_9}_64]
          │
          ▼
3. Simulate each sequence with world model:
   For each sequence k:
     repr_0 = Encoder(s_t)
     for i in 0..N-1:
       repr_{i+1} = Predictor(repr_i, a_i)
     value_k = ValueHead(repr_N)
          │
          ▼
4. Select best trajectory:
   k* = argmax_k(value_k)
          │
          ▼
5. Execute first action:
   a* = a_0 of sequence k*
          │
          ▼
6. Observe next state s_{t+1}, goto 1
```

### Implementation

```elixir
# lib/exphil/planning/mpc.ex
defmodule ExPhil.Planning.MPC do
  @moduledoc """
  Model Predictive Control using learned world model.

  At each timestep:
  1. Sample K candidate action sequences
  2. Simulate with world model predictor
  3. Score with value function
  4. Execute best first action

  Based on V-JEPA 2-AC robotics planning.
  """

  @default_opts [
    num_candidates: 64,      # K: number of action sequences to sample
    horizon: 10,             # N: planning horizon in frames
    temperature: 1.0,        # Action sampling temperature
    top_k: 8,                # For CEM-style refinement (optional)
    iterations: 1            # CEM iterations (1 = random shooting)
  ]

  @doc """
  Plan best action given current state using world model.

  ## Parameters
    - world_model: %{encoder, predictor, value_head}
    - params: Trained parameters for all components
    - state: Current game state tensor
    - policy: Policy network for action proposal (optional, else random)
    - opts: Planning options

  ## Returns
    {best_action, planning_info}
  """
  def plan(world_model, params, state, policy \\ nil, opts \\ []) do
    opts = Keyword.merge(@default_opts, opts)

    # Encode current state
    repr = encode_state(world_model.encoder, params.encoder, state)

    # Generate candidate action sequences
    action_sequences = generate_candidates(
      policy,
      params.policy,
      state,
      opts[:num_candidates],
      opts[:horizon],
      opts[:temperature]
    )

    # Simulate each sequence and score
    values = simulate_and_score(
      world_model,
      params,
      repr,
      action_sequences,
      opts[:horizon]
    )

    # Select best trajectory
    best_idx = Nx.argmax(values) |> Nx.to_number()
    best_sequence = Enum.at(action_sequences, best_idx)
    best_action = hd(best_sequence)

    info = %{
      best_value: Nx.to_number(values[best_idx]),
      mean_value: Nx.to_number(Nx.mean(values)),
      value_std: Nx.to_number(Nx.standard_deviation(values))
    }

    {best_action, info}
  end

  defp encode_state(encoder, params, state) do
    Axon.predict(encoder, params, %{"state" => state})
  end

  defp generate_candidates(nil, _params, _state, k, horizon, _temp) do
    # Random action sampling (uniform over action space)
    for _ <- 1..k do
      for _ <- 1..horizon do
        sample_random_action()
      end
    end
  end

  defp generate_candidates(policy, params, state, k, horizon, temp) do
    # Policy-guided sampling with temperature
    for _ <- 1..k do
      # Could use autoregressive sampling from policy
      # with added noise/temperature for diversity
      for _ <- 1..horizon do
        sample_from_policy(policy, params, state, temp)
      end
    end
  end

  defp simulate_and_score(world_model, params, initial_repr, action_sequences, horizon) do
    # Batch simulate all sequences in parallel
    # Returns tensor of shape {num_candidates}

    sequences_tensor = action_sequences_to_tensor(action_sequences)

    # Unroll world model for horizon steps
    final_reprs = Enum.reduce(0..(horizon-1), initial_repr, fn step, repr ->
      actions_at_step = sequences_tensor[[.., step, ..]]
      action_embed = embed_actions(world_model.action_embedder, params.action_embedder, actions_at_step)
      predict_next(world_model.predictor, params.predictor, repr, action_embed)
    end)

    # Score final representations with value head
    Axon.predict(world_model.value_head, params.value_head, %{"repr" => final_reprs})
    |> Nx.squeeze()
  end

  defp sample_random_action do
    # Sample random controller state
    %{
      buttons: for(_ <- 1..8, do: :rand.uniform() < 0.1),  # Low button probability
      main_x: :rand.uniform() * 2 - 1,
      main_y: :rand.uniform() * 2 - 1,
      c_x: :rand.uniform() * 2 - 1,
      c_y: :rand.uniform() * 2 - 1,
      shoulder: :rand.uniform()
    }
  end
end
```

### Latency Considerations

MPC adds computation at inference time. Budget:
- 16.7ms total (60 FPS)
- Current policy: 8.9ms (Mamba)
- Available for MPC: ~7ms

With 64 candidates, 10-step horizon:
- 640 predictor forward passes
- Need batched, JIT-compiled predictor
- May need to reduce candidates (32) or horizon (5) for real-time

### Success Criteria

- [ ] MPC planning runs under 16.7ms total
- [ ] Win rate improves on planning-heavy tasks (edgeguarding)
- [ ] Graceful degradation if planning budget exceeded (fall back to reactive)
- [ ] Value function quality sufficient for trajectory scoring

---

## Timeline Summary

```
Month 1:
├── Week 1-2: Self-play stress testing (#1)
├── Week 3: Sparse reward experiment (#2)
└── Week 4: Begin GRPO implementation (#3)

Month 2:
├── Week 1-2: Complete GRPO, run comparison (#3)
├── Week 3: State prediction auxiliary loss (#4)
└── Week 4: Evaluate world model quality (#4)

Month 3:
├── Week 1-2: Hierarchical policy architecture (#5)
├── Week 3-4: Train hierarchical, compare to flat (#5)

Month 4+:
├── MPC planning implementation (#6)
├── Real-time optimization (#6)
└── Integration and polish
```

---

## References

- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) - GRPO algorithm
- [A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf) - H-JEPA
- [V-JEPA 2](https://arxiv.org/abs/2506.09985) - MPC with world models
- [MuZero](https://arxiv.org/abs/1911.08265) - Learned dynamics + planning
- [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) - Scale over engineering
