# Reward Shaping Deep Dive

This document provides a comprehensive guide to reward shaping for Melee AI training, covering theory, practical implementations, character-specific considerations, and failure modes.

## Why Reward Shaping Matters

Melee presents a classic **sparse reward** problem:
- A stock KO may take 30+ seconds of continuous play
- At 60 FPS, that's 1800+ frames between meaningful feedback
- Pure RL without shaping requires billions of samples to learn

Reward shaping provides **dense training signal** by rewarding intermediate behaviors that correlate with good outcomes.

---

## Standard Reward Functions

### Stock-Based Rewards (slippi-ai, Phillip)

The fundamental reward signal in all Melee AI projects:

```python
# slippi-ai reward.py
stock_diff = opponent_stocks_lost - self_stocks_lost
reward = stock_diff  # +1 for taking stock, -1 for losing
```

**Properties**:
- Sparse but unambiguous
- No reward hacking possible
- Slow learning without shaping

### Damage-Based Rewards

All major projects add damage as a denser signal:

```python
# Phillip approach
damage_diff = damage_dealt - damage_taken
reward = stocks + damage_ratio * damage_diff
```

**slippi-ai configuration**:
```python
@dataclasses.dataclass
class RewardConfig:
    damage_ratio: float = 0.01       # Damage weight vs KO
    ledge_grab_penalty: float = 0    # Discourage defensive ledge
    approaching_factor: float = 0    # Reward closing distance
    stalling_penalty: float = 0      # Penalize offstage camping
    stalling_threshold: float = 20   # Distance to trigger
    nana_ratio: float = 0.5          # Ice Climbers partner weight
```

### ExPhil Standard Implementation

ExPhil separates rewards into components for flexibility:

```elixir
# ExPhil.Rewards.Standard
%{
  stock: compute_stock_reward(...),   # +1/-1 for stock changes
  damage: compute_damage_reward(...), # Damage dealt - taken
  win: compute_win_reward(...)        # Game end bonus
}
```

**Key design choice**: Return components separately so training can weight them dynamically.

---

## Potential-Based Reward Shaping (PBRS)

### Theory

[Potential-based reward shaping](https://medium.com/@sophiezhao_2990/potential-based-reward-shaping-in-reinforcement-learning-05da05cfb84a) is the gold standard for adding shaped rewards without changing optimal policy.

**Definition**:
```
F(s, a, s') = γ × Φ(s') - Φ(s)
```

Where:
- `Φ(s)` is a potential function over states
- `γ` is the discount factor
- `F` is the shaping reward added to the true reward

**Key theorem** (Ng et al., 1999): PBRS is **necessary and sufficient** for policy invariance - the optimal policy under shaped rewards equals the optimal policy under true rewards.

### Why PBRS Matters

Without PBRS guarantees, shaped rewards can:
1. Create local optima that don't align with true objectives
2. Cause the agent to prioritize shaping over actual goals
3. Lead to reward hacking

### Practical Application

For Melee, good potential functions include:

```python
# Position-based potential
def position_potential(state):
    """Higher potential when opponent is at high % near edge"""
    opponent_percent = state.opponent.percent
    opponent_edge_distance = distance_to_nearest_edge(state.opponent)

    # High potential when opponent is in kill range near edge
    kill_potential = opponent_percent / 150.0  # Normalize to ~1 at kill %
    edge_factor = 1.0 - (opponent_edge_distance / 100.0)

    return kill_potential * edge_factor

# Advantage-based potential
def advantage_potential(state):
    """Stock and percent advantage"""
    stock_adv = (state.player.stocks - state.opponent.stocks) * 0.5
    percent_adv = (state.opponent.percent - state.player.percent) / 200.0
    return stock_adv + percent_adv
```

**Shaping reward**:
```python
shaping = gamma * position_potential(next_state) - position_potential(state)
total_reward = true_reward + shaping
```

### Recent Research (2024)

[Sample efficiency research](https://arxiv.org/abs/2404.07826) shows that using the optimal value function as potential produces the best results, but this requires approximation since we don't know V* in advance.

[Hierarchical PBRS](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2024.1444188/full) extends PBRS to structured tasks, potentially useful for Melee's hierarchical nature (neutral → advantage → kill).

---

## Character-Specific Reward Shaping

ExPhil targets low-tier characters with unique playstyles. Generic rewards may not capture optimal play.

### Mewtwo

**Characteristics**:
- Excellent recovery (Teleport is nearly unpunishable)
- Large hitboxes but light weight
- Unique tech: Shadow Ball charge management

**Shaped Rewards**:
```elixir
# Mewtwo-specific rewards
defmodule ExPhil.Rewards.Mewtwo do
  def compute(prev_state, curr_state, opts) do
    %{
      # Recovery is safe - reduce penalty
      recovery_risk: compute_recovery_risk(curr_state) * 0.3,

      # Reward Shadow Ball management
      shadow_ball: compute_shadow_ball_reward(prev_state, curr_state),

      # Reward using tail hitboxes (disjoint)
      tail_spacing: compute_spacing_reward(curr_state, range: 15..25)
    }
  end

  defp compute_shadow_ball_reward(prev, curr) do
    # Reward charging when safe
    if safe_to_charge?(curr) and charging_shadow_ball?(curr) do
      0.1
    # Bonus for landing fully charged Shadow Ball
    elsif shadow_ball_hit?(prev, curr) and full_charge?(prev) do
      0.5
    else
      0.0
    end
  end
end
```

### Ganondorf

**Characteristics**:
- Slowest character, poor recovery
- Devastating punish game
- High risk/reward playstyle

**Shaped Rewards**:
```elixir
defmodule ExPhil.Rewards.Ganondorf do
  def compute(prev_state, curr_state, opts) do
    %{
      # Approach reward needs to account for slow movement
      approach: compute_approach_reward(curr_state) * 0.5,

      # Heavy penalty for recovery situations
      recovery_risk: compute_recovery_risk(curr_state) * 2.0,

      # Bonus for spacing with down-B
      wizard_foot_spacing: compute_wizards_foot_reward(prev, curr),

      # Reward reading opponent's approach
      punish_quality: compute_punish_quality(prev, curr)
    }
  end

  defp compute_punish_quality(prev, curr) do
    # Ganon excels at hard reads - reward high-damage single hits
    damage = curr.opponent.percent - prev.opponent.percent
    if damage > 15.0 do
      damage / 30.0  # Normalize ~0.5-1.0 for big hits
    else
      0.0
    end
  end
end
```

### Link

**Characteristics**:
- Projectile zoning (Boomerang, Bomb, Arrow)
- Tether recovery (vulnerable)
- Item management complexity

**Shaped Rewards**:
```elixir
defmodule ExPhil.Rewards.Link do
  def compute(prev_state, curr_state, opts) do
    %{
      # Don't penalize zoning distance
      approach: 0.0,  # Disable approach reward

      # Reward projectile pressure
      projectile_control: compute_projectile_reward(prev, curr),

      # Reward bomb tricks
      bomb_tech: compute_bomb_reward(prev, curr),

      # Higher recovery penalty (tether is punishable)
      recovery_risk: compute_recovery_risk(curr_state) * 1.5
    }
  end

  defp compute_projectile_reward(prev, curr) do
    # Reward keeping projectiles active
    active_projectiles = count_player_projectiles(curr)
    projectile_hits = count_projectile_hits(prev, curr)

    active_projectiles * 0.05 + projectile_hits * 0.3
  end
end
```

### Game & Watch

**Characteristics**:
- Bucket (absorbs projectiles for huge damage)
- Unique frame data (no L-canceling needed)
- Light weight with good recovery

**Shaped Rewards**:
```elixir
defmodule ExPhil.Rewards.GameWatch do
  def compute(prev_state, curr_state, opts) do
    %{
      # Bucket management is crucial
      bucket: compute_bucket_reward(prev, curr),

      # Reward aerial approaches (good aerials)
      aerial_approach: compute_aerial_reward(prev, curr),

      # Standard recovery risk
      recovery_risk: compute_recovery_risk(curr_state)
    }
  end

  defp compute_bucket_reward(prev, curr) do
    bucket_level_prev = get_bucket_level(prev)
    bucket_level_curr = get_bucket_level(curr)

    cond do
      # Reward absorbing projectiles
      bucket_level_curr > bucket_level_prev ->
        0.3 * (bucket_level_curr - bucket_level_prev)

      # Big reward for landing full bucket
      bucket_level_prev >= 3 and bucket_emptied?(prev, curr) ->
        1.0

      true ->
        0.0
    end
  end
end
```

### Ice Climbers

**Characteristics**:
- Dual characters (Popo + Nana)
- Wobbling and desync techniques
- Nana AI is separate and exploitable

**Shaped Rewards**:
```elixir
defmodule ExPhil.Rewards.IceClimbers do
  def compute(prev_state, curr_state, opts) do
    %{
      # Nana preservation is critical
      nana_safety: compute_nana_safety(curr_state),

      # Reward desync setups
      desync: compute_desync_reward(prev, curr),

      # Reward grab conversions (wobbling)
      grab_conversion: compute_grab_reward(prev, curr),

      # Severe penalty for losing Nana
      nana_death: compute_nana_death_penalty(prev, curr)
    }
  end

  defp compute_nana_safety(state) do
    # Reward keeping Nana close and safe
    distance_to_popo = get_climber_distance(state)
    nana_percent = state.nana.percent

    safety = 1.0 - (distance_to_popo / 30.0) - (nana_percent / 200.0)
    max(0.0, safety) * 0.2
  end

  defp compute_nana_death_penalty(prev, curr) do
    if nana_died?(prev, curr) do
      -2.0  # Losing Nana is devastating
    else
      0.0
    end
  end
end
```

---

## Intrinsic Motivation & Curiosity

### The Exploration Problem

Melee's action space is enormous (~30 billion combinations). Standard RL explores randomly, but:
- Most random actions are useless
- Techniques like wavedashing require precise inputs
- Discovering combos requires specific sequences

### Curiosity-Driven Exploration

[Research on fighting games](https://www.mdpi.com/2073-431X/14/10/434) shows that **Intrinsic Curiosity Module (ICM) + A3C** outperforms baseline A3C consistently.

**How ICM works**:
```python
# Forward model predicts next state from current state + action
predicted_next = forward_model(state, action)

# Intrinsic reward = prediction error
curiosity_reward = ||predicted_next - actual_next||²

# Total reward
reward = extrinsic_reward + beta * curiosity_reward
```

**Benefits for Melee**:
- Encourages trying new action sequences
- Naturally discovers advanced techniques
- Avoids repetitive, safe-but-boring strategies

### The Noisy TV Problem

[A known challenge](https://pathak22.github.io/large-scale-curiosity/resources/largeScaleCuriosity2018.pdf) is agents being attracted to unpredictable phenomena:
- Stage elements like Randall
- Opponent's random behavior
- Environmental noise

**Mitigation**: Use inverse dynamics model to focus on agent-controllable aspects:
```python
# Inverse model: predict action from state transition
# Forces encoder to learn action-relevant features
predicted_action = inverse_model(state, next_state)
inverse_loss = cross_entropy(predicted_action, actual_action)
```

### Skill-Based Curiosity

[Skill-based curiosity](https://link.springer.com/article/10.1007/s10994-019-05845-8) extends ICM by discovering reusable skills:

1. **Skill discovery**: Find common action patterns
2. **Skill execution**: Reward successful skill completion
3. **Skill composition**: Combine skills into strategies

For Melee, this could discover:
- Wavedash patterns
- L-cancel timing
- Common combo routes

---

## Reward Hacking Failure Modes

### What is Reward Hacking?

[Reward hacking](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) occurs when agents exploit flaws in reward design to maximize score without learning the intended behavior.

### Classic Examples

From [Wikipedia](https://en.wikipedia.org/wiki/Reward_hacking):
- A robot arm rewarded for placing a red block high flipped it upside-down (reward measured bottom face height)
- A bicycle agent rode tiny circles around the goal (no penalty for moving away)
- A walking robot discovered glitches allowing horizontal movement without steps

### Melee-Specific Failure Modes

#### 1. Approach Reward Hacking

**Problem**: Agent oscillates back and forth to maximize approach reward
```
Frame 1: Move toward opponent (+reward)
Frame 2: Move away (no penalty or small penalty)
Frame 3: Move toward (+reward)
...never actually attacks
```

**Solution**: Only reward approach when in attack range, or use potential-based shaping:
```python
# Instead of reward for distance change
approach_reward = prev_distance - curr_distance  # HACKABLE

# Use potential-based (only rewards net progress)
approach_shaping = gamma * potential(curr) - potential(prev)  # SAFE
```

#### 2. Damage Farming

**Problem**: Agent repeatedly hits weak moves for damage reward
```
Jab → Jab → Jab → Jab (lots of small rewards)
Instead of: Grab → Throw → Aerial → Kill
```

**Solution**: Reward damage per opening, not raw damage:
```python
# Bad: reward each hit equally
reward = damage_dealt * 0.01

# Better: reward efficient punishes
reward = damage_dealt / openings_used
```

#### 3. Stalling

**Problem**: Agent stays offstage to avoid engagement
```
Camp under stage, wait for timeout
Or: Repeatedly grab ledge to stall
```

**Solution**: Add stalling penalties (slippi-ai does this):
```python
if player_offstage and distance_from_stage > threshold:
    reward -= stalling_penalty
```

#### 4. Ledge Camping

**Problem**: Agent repeatedly grabs ledge for invincibility frames

**Solution**: Track ledge grabs and penalize:
```python
ledge_grabs_this_stock += 1
if ledge_grabs_this_stock > threshold:
    reward -= ledge_penalty
```

#### 5. Suicide for Reset

**Problem**: When losing badly, agent SDs to reset percent
```
At 150% with opponent at 30%
SD → Respawn at 0%
Repeat trading until opponent dies first
```

**Solution**: Make stock loss much more negative than percent difference:
```python
stock_loss_penalty = -1.0  # Fixed large penalty
percent_advantage = (opponent_percent - player_percent) / 200.0  # ~0.5 max
# Stock loss always dominates
```

### Detection and Mitigation

[Modern approaches](https://arxiv.org/html/2507.05619v1) include:

1. **Anomaly detection**: Monitor for unusual behavior patterns
2. **Multi-objective rewards**: Balance competing objectives
3. **Human feedback**: RLHF to detect unnatural play
4. **Constraint learning**: Explicitly constrain undesirable behaviors

---

## Combining Reward Components

### Weighted Sum Approach

```elixir
defmodule ExPhil.Rewards.Combined do
  @default_weights %{
    stock: 1.0,
    damage: 0.01,
    approach: 0.001,
    combo: 0.005,
    edge_guard: 0.01,
    recovery_risk: -0.002
  }

  def compute(prev_state, curr_state, opts) do
    weights = Keyword.get(opts, :weights, @default_weights)

    standard = ExPhil.Rewards.Standard.compute(prev_state, curr_state, opts)
    shaped = ExPhil.Rewards.Shaped.compute(prev_state, curr_state, opts)

    # Character-specific if available
    character = get_character(curr_state, opts[:player_port])
    character_rewards = get_character_rewards(character, prev_state, curr_state, opts)

    # Combine all components
    all_rewards = Map.merge(Map.merge(standard, shaped), character_rewards)

    # Weighted sum
    Enum.reduce(all_rewards, 0.0, fn {key, value}, acc ->
      weight = Map.get(weights, key, 0.0)
      acc + weight * value
    end)
  end
end
```

### Curriculum Learning

Start with dense rewards, gradually reduce shaping:

```python
def get_shaping_weight(epoch, total_epochs):
    """Anneal shaping from 1.0 to 0.0"""
    progress = epoch / total_epochs
    return max(0.0, 1.0 - progress)

# During training
shaping_weight = get_shaping_weight(current_epoch, total_epochs)
reward = true_reward + shaping_weight * shaping_reward
```

### Adaptive Weighting

Adjust weights based on agent skill:

```python
def adaptive_weights(win_rate):
    """More shaping when struggling, less when winning"""
    if win_rate < 0.3:
        return {"damage": 0.02, "approach": 0.005}  # Dense guidance
    elif win_rate > 0.6:
        return {"damage": 0.005, "approach": 0.0}   # Sparse, true objective
    else:
        return {"damage": 0.01, "approach": 0.002}  # Moderate
```

---

## Practical Recommendations for ExPhil

### Phase 1: Behavioral Cloning

No rewards needed - just imitation loss. But track statistics:
- Action accuracy by type (movement, attacks, defense)
- Technique execution (L-cancel rate, wavedash frequency)
- Positioning patterns

### Phase 2: Early RL

Use dense shaped rewards:
```elixir
%{
  stock: 1.0,
  damage: 0.01,
  win: 1.0,
  approach: 0.002,
  combo: 0.01,
  recovery_risk: -0.003
}
```

Monitor for hacking:
- Approach without attacking
- Damage without stocks
- Excessive ledge grabs

### Phase 3: Late RL

Reduce shaping, increase sparse rewards:
```elixir
%{
  stock: 1.0,
  damage: 0.005,
  win: 2.0,  # Emphasize winning
  approach: 0.0,
  combo: 0.002,
  recovery_risk: -0.001
}
```

### Phase 4: Final Polish

Minimal shaping, maximize win rate:
```elixir
%{
  stock: 1.0,
  damage: 0.001,
  win: 3.0
}
```

---

## Metrics to Track

### Reward Quality Indicators

| Metric | Good Sign | Bad Sign |
|--------|-----------|----------|
| Stock correlation | High | Low |
| Damage/stock ratio | Decreasing over training | Increasing |
| Episode length | Stable or decreasing | Increasing (stalling) |
| Approach/attack ratio | High | Low (oscillating) |
| Win rate vs shaped reward | Correlated | Uncorrelated |

### Detecting Reward Hacking

```elixir
defmodule ExPhil.Training.RewardMonitor do
  def check_for_hacking(episode_stats) do
    warnings = []

    # Check for approach oscillation
    if episode_stats.approach_reward > episode_stats.attack_count * 2 do
      warnings = ["Possible approach farming" | warnings]
    end

    # Check for damage without progress
    if episode_stats.damage_dealt > 200 and episode_stats.stocks_taken == 0 do
      warnings = ["High damage, no stocks - possible weak hit spam" | warnings]
    end

    # Check for excessive ledge grabs
    if episode_stats.ledge_grabs > 20 do
      warnings = ["Excessive ledge camping" | warnings]
    end

    warnings
  end
end
```

---

## References

### Theory
- [Potential-Based Reward Shaping](https://medium.com/@sophiezhao_2990/potential-based-reward-shaping-in-reinforcement-learning-05da05cfb84a) - Sophie Zhao
- [Sample Efficiency of PBRS](https://arxiv.org/abs/2404.07826) - 2024 paper
- [Hierarchical PBRS](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2024.1444188/full) - Frontiers 2024
- [Mastering RL: Reward Shaping Notes](https://gibberblot.github.io/rl-notes/single-agent/reward-shaping.html)

### Reward Hacking
- [Reward Hacking in RL](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) - Lilian Weng
- [Wikipedia: Reward Hacking](https://en.wikipedia.org/wiki/Reward_hacking)
- [Detecting and Mitigating Reward Hacking](https://arxiv.org/html/2507.05619v1) - 2025 paper

### Intrinsic Motivation
- [Curiosity-Driven Exploration for Games](https://www.mdpi.com/2073-431X/14/10/434) - MDPI 2023
- [Large-Scale Curiosity Study](https://pathak22.github.io/large-scale-curiosity/resources/largeScaleCuriosity2018.pdf) - OpenAI
- [Skill-Based Curiosity](https://link.springer.com/article/10.1007/s10994-019-05845-8) - Machine Learning Journal

### Fighting Game AI
- [Pro-Level Fighting Game AI](https://www.researchgate.net/publication/348319705_Creating_Pro-Level_AI_for_a_Real_Time_Fighting_Game_Using_Deep_Reinforcement_Learning) - ResearchGate
- [Fighting Game Agent Study](https://onlinelibrary.wiley.com/doi/10.1155/2022/9984617) - Wiley 2022

### Melee-Specific
- [slippi-ai](https://github.com/vladfi1/slippi-ai) - vladfi1's reward implementation
- [Phillip](https://github.com/vladfi1/phillip) - Original pure RL rewards
