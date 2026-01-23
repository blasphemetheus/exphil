# Training Approaches for Melee AI

This document compares different training methodologies for Melee AI, analyzing their strengths, weaknesses, and practical considerations.

## Overview of Approaches

| Approach | Data Source | Compute | Human-Like | Ceiling |
|----------|-------------|---------|------------|---------|
| Pure RL | Self-play | Very High | No | Superhuman |
| Behavioral Cloning | Replays | Low | Yes | Human |
| BC + RL | Replays + Self-play | Medium | Initially | Beyond Human |
| Transformer IL | Replays | Medium | Yes | Human |
| Rule-Based | Expert Knowledge | None | Tunable | Fixed |

## Pure Reinforcement Learning

### Overview

Train from scratch using only self-play, without human demonstrations.

```
Random Policy → Self-Play → Better Policy → Self-Play → ...
```

### Architecture (Phillip)

```
┌─────────────────────────────────────────┐
│              Actor-Critic               │
├─────────────────────────────────────────┤
│ Actor: State → Action Distribution      │
│ Critic: State → Value Estimate          │
├─────────────────────────────────────────┤
│ Algorithm: A3C / PPO                    │
│ Exploration: Entropy Regularization     │
└─────────────────────────────────────────┘
```

### Training Loop

```python
while not converged:
    # Collect experience from parallel environments
    trajectories = []
    for env in environments:
        trajectory = rollout(policy, env)
        trajectories.append(trajectory)

    # Compute advantages
    advantages = compute_gae(trajectories, value_function)

    # Policy gradient update
    for batch in batches(trajectories):
        actor_loss = -mean(log_prob(actions) * advantages)
        critic_loss = mean((values - returns) ** 2)
        entropy_loss = -mean(entropy(policy))

        loss = actor_loss + critic_weight * critic_loss - entropy_weight * entropy_loss
        update(loss)
```

### Advantages

1. **No data dependency**: Can train any character without replays
2. **Discovers novel strategies**: Not limited to human play
3. **Potentially superhuman**: No ceiling from demonstrations
4. **Self-improving**: Gets better opponents as it improves

### Disadvantages

1. **Sample inefficiency**: Billions of frames needed
2. **Compute intensive**: Months of training, distributed systems
3. **Exploration challenges**: Massive action space makes exploration hard
4. **Unnatural play**: May develop alien strategies

### When to Use

- Unlimited compute budget
- Novel character/matchup with no replay data
- Research into optimal play beyond human metagame
- Characters with simple movesets (fewer exploration challenges)

## Behavioral Cloning

### Overview

Supervised learning from human demonstrations.

```
Human Replays → (state, action) pairs → Supervised Learning → Policy
```

### Architecture

```
┌─────────────────────────────────────────┐
│          Imitation Network              │
├─────────────────────────────────────────┤
│ Input: Game State                       │
│ Output: Action Probabilities            │
├─────────────────────────────────────────┤
│ Loss: Cross-Entropy (actions)           │
│       + MSE (continuous components)     │
└─────────────────────────────────────────┘
```

### Training Loop

```python
for epoch in range(num_epochs):
    for batch in replay_dataset:
        states, actions = batch

        # Forward pass
        predicted_actions = policy(states)

        # Compute loss
        button_loss = cross_entropy(predicted_actions.buttons, actions.buttons)
        stick_loss = mse(predicted_actions.stick, actions.stick)
        loss = button_loss + stick_loss

        # Update
        optimizer.step(loss)

    # Evaluate
    accuracy = evaluate_on_test_set(policy)
```

### Advantages

1. **Fast training**: Days vs months
2. **Human-like play**: Inherits human style and technique
3. **Stable training**: Supervised learning is well-understood
4. **Data efficient**: Works with tens of thousands of games

### Disadvantages

1. **Bounded by data quality**: Can't exceed human demonstrations
2. **Compounding errors**: Small mistakes accumulate
3. **Distribution mismatch**: Training states ≠ deployment states
4. **No adaptation**: Can't improve from self-play

### When to Use

- Limited compute budget
- Want human-like aesthetics
- Character has abundant high-quality replays
- Baseline model before RL refinement

## BC + RL (Two-Stage)

### Overview

Combine strengths: BC for initialization, RL for refinement.

```
Stage 1: Replays → BC → Initial Policy
Stage 2: Initial Policy → Self-Play RL → Refined Policy
```

### Architecture (slippi-ai)

```
┌─────────────────────────────────────────┐
│              Stage 1: BC                │
├─────────────────────────────────────────┤
│ Train policy on replay data             │
│ Also train value function on returns    │
├─────────────────────────────────────────┤
│              Stage 2: RL                │
├─────────────────────────────────────────┤
│ Freeze BC policy as "teacher"           │
│ Train new policy with PPO               │
│ KL regularization to teacher            │
└─────────────────────────────────────────┘
```

### KL Regularization

**Purpose**: Prevent policy from diverging too far from human-like play

```python
# PPO loss with KL penalty
ppo_loss = clip_surrogate_objective(...)
kl_loss = kl_divergence(current_policy, teacher_policy)

total_loss = ppo_loss + kl_weight * kl_loss
```

**Why it matters**:
- Pure RL from BC init can collapse to degenerate strategies
- KL keeps play recognizable and stable
- Gradually reduce weight to allow innovation

### Training Loop

```python
# Stage 1: BC
bc_policy = train_bc(replay_dataset, epochs=50)
bc_value = train_value(replay_dataset, discount=0.994)

# Stage 2: RL
teacher = freeze(bc_policy)
student = copy(bc_policy)

while not converged:
    # Collect rollouts
    trajectories = collect_rollouts(student, environments)

    # Compute advantages
    advantages = compute_gae(trajectories, bc_value)

    # PPO update with KL penalty
    for batch in batches(trajectories):
        ppo_loss = ppo_objective(student, batch)
        kl_loss = kl_divergence(student, teacher)

        loss = ppo_loss + kl_weight * kl_loss
        update(loss)

    # Optional: decay KL weight
    kl_weight *= kl_decay
```

### Advantages

1. **Best of both worlds**: Human-like start, superhuman ceiling
2. **Practical compute**: Weeks vs months
3. **Stable training**: BC provides strong initialization
4. **Proven results**: slippi-ai competitive with top players

### Disadvantages

1. **Complexity**: Two-stage pipeline, more hyperparameters
2. **Still needs data**: Relies on BC quality
3. **KL tuning**: Wrong weight causes collapse or stagnation
4. **Self-play challenges**: Still needs careful opponent selection

### When to Use

- Targeting competitive play against humans
- Have good replay data AND compute for RL
- Want to exceed human performance ceiling
- Primary recommended approach for ExPhil

## Transformer-Based Imitation

### Overview

Treat action prediction as next-token prediction (GPT-style).

```
State Sequence → Transformer → Next Action
```

### Architecture (Eric Gu)

```
┌─────────────────────────────────────────┐
│        Decoder-Only Transformer         │
├─────────────────────────────────────────┤
│ Input: [s_0, a_0, s_1, a_1, ..., s_t]  │
│ Output: a_t distribution                │
├─────────────────────────────────────────┤
│ Parameters: ~20M                        │
│ Context: ~1000 frames                   │
└─────────────────────────────────────────┘
```

### Training

```python
# Causal attention mask
def attention_mask(seq_len):
    return tril(ones(seq_len, seq_len))

# Training loop
for batch in dataset:
    states, actions = batch  # Interleaved sequence

    # Predict next action at each step
    logits = transformer(concat(states, actions), mask=attention_mask)

    # Cross-entropy loss on action positions
    loss = cross_entropy(logits[action_positions], actions)

    update(loss)
```

### Key Findings (Eric Gu)

1. **All-character > single-character**: General model outperforms specialists
2. **Scaling helps**: 20M params > 5M params
3. **Long context helps**: 1000 frames > 100 frames
4. **Cheap training**: $5, 5 hours on 2× 3090s

### Advantages

1. **Simple objective**: Just next-token prediction
2. **Scales with data**: More replays → better performance
3. **Long context**: Can model long-term strategies
4. **Multi-character**: Single model for all characters

### Disadvantages

1. **Pure imitation ceiling**: Can't exceed demonstration quality
2. **No RL refinement**: Standard approach doesn't include self-play
3. **Attention cost**: O(n²) limits sequence length
4. **Less explored**: Fewer published results than BC+RL

### When to Use

- Want simplest possible training setup
- Have massive amounts of replay data
- Multi-character model is priority
- Not targeting superhuman play

## Self-Play Considerations

### The Rock-Paper-Scissors Problem

**Single opponent self-play** leads to:
```
Policy A beats Policy B
Policy B' beats Policy A
Policy A' beats Policy B'
→ Oscillating, never converging
```

### Solutions

#### 1. Historical Sampling

```python
# Maintain pool of past checkpoints
checkpoint_pool = []

# Every N steps, save checkpoint
if step % save_interval == 0:
    checkpoint_pool.append(copy(policy))

# Sample opponent from pool
opponent = random.choice(checkpoint_pool)
```

#### 2. Population-Based Training

```python
# Maintain population of diverse agents
population = [Policy() for _ in range(pop_size)]

# Each agent plays against population
for agent in population:
    opponents = sample(population, k=3)
    for opponent in opponents:
        trajectory = play(agent, opponent)
        agent.update(trajectory)

# Selection and mutation
population = evolve(population)
```

#### 3. League System (AlphaStar)

```
Main Agents: Optimize for overall performance
League Exploiters: Find weaknesses in main agents
Main Exploiters: Beat league exploiters

Cascading improvement with stability
```

### Reward Shaping

**Sparse rewards** (win/loss only):
```python
reward = 1 if opponent_died else (-1 if self_died else 0)
```

**Dense rewards** (shaped):
```python
reward = (
    1.0 * stock_diff +
    0.01 * damage_diff +
    0.001 * spacing_reward +
    -0.001 * stalling_penalty
)
```

**Trade-offs**:
- Sparse: No human bias, but slow learning
- Dense: Faster learning, but may encode suboptimal strategies

## Practical Recommendations

### For ExPhil

**Recommended approach**: BC + RL with Mamba backbone

```
Stage 1: BC with Mamba
- Train on low-tier character replays
- Use learned action embeddings (smaller model)
- Target: 80%+ action accuracy

Stage 2: PPO Self-Play
- Use historical sampling
- KL regularization to BC teacher
- Character-specific reward shaping
- Target: Competitive with top online players
```

### Training Schedule

```
Week 1: Data preparation
- Filter replays for target characters
- Compute embedding statistics
- Cache embeddings for fast training

Week 2-3: BC training
- Train Mamba backbone
- Hyperparameter tuning (LR, batch size)
- Evaluate action accuracy

Week 4+: RL refinement
- Set up self-play infrastructure
- Tune KL weight
- Monitor for policy collapse
```

### Compute Budget

| Stage | GPU Hours | Notes |
|-------|-----------|-------|
| BC | 24-72 | Single GPU sufficient |
| RL | 200-500 | Parallel envs help |
| Evaluation | 10-20 | Per checkpoint |

### Metrics to Track

**BC Stage**:
- Action accuracy (per component)
- Loss curve convergence
- Validation vs training gap

**RL Stage**:
- Win rate vs teacher
- Win rate vs CPU levels
- KL divergence from teacher
- Entropy of policy
- Episode length

## References

- [PPO Paper](https://arxiv.org/abs/1707.06347) - Schulman et al.
- [AlphaStar](https://www.nature.com/articles/s41586-019-1724-z) - League training
- [Phillip Paper](https://arxiv.org/abs/1702.06230) - Pure RL for Melee
- [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) - Scale vs induction
