# Project Nabla Case Study

**Author**: Bryan Chen (@otter_collapse)
**Blog**: [bycn.github.io](https://bycn.github.io/2022/08/19/project-nabla-writeup.html)
**Status**: Research project (2022)
**Approach**: BC + RL self-play with population-based training

## Overview

Project Nabla used a dataset of tournament players and large-scale self-play to train AIs that play Super Smash Bros. Melee in a competent and human-like manner. The project provides critical insights into self-play stability and the importance of behavioral priors.

**Key Achievement**: Demonstrated that BC agents pass an informal "Melee Turing Test" and identified failure modes of naive self-play.

## Technical Approach

### Two-Stage Training

```
Stage 1: Behavioral Cloning
    Tournament replays → Supervised learning → BC Policy

Stage 2: Self-Play RL
    BC Policy → Self-play with population → Refined Policy
```

### Infrastructure

- **Data Source**: Slippi tournament replays
- **Software**: Fizzi's Slippi tools + fast-forward gecko codes
- **Training**: Consumer GPU (within days for BC)

### Key Innovation: Population-Based Self-Play

Rather than single-opponent self-play:

```
Population = [Agent_1, Agent_2, ..., Agent_N]

For each training step:
    Sample opponent from population
    Play games, collect trajectories
    Update agent with RL gradients
    Periodically add agent to population
```

## Critical Findings

### 1. BC Learns Modular Skills

> "Within the span of a few days on 1 consumer grade GPU, we can train agents that mimic human skills."

BC agents exhibit:
- Wavedashing
- L-canceling
- Short hop aerials
- Basic combo execution

These emerge naturally from imitation without explicit encoding.

### 2. Single-Opponent Self-Play Fails

**The Rock-Paper-Scissors Problem**:

```
Policy A beats Policy B
    ↓ (train against A)
Policy B' beats Policy A
    ↓ (train against B')
Policy A' beats Policy B'
    ↓ (cycle continues)

Result: Oscillating policies, no convergence
```

**Why it happens**:
- Each policy over-fits to current opponent
- Counter-strategies are easy to learn
- No pressure to remain generally strong

### 3. Population Training Necessary

**Solution**: Maintain diverse population of past and current agents

```python
def train_step(agent, population):
    # Sample from history, not just current self
    opponent = sample(population)

    # Play and learn
    trajectory = play(agent, opponent)
    agent.update(trajectory)

    # Periodically snapshot
    if should_snapshot():
        population.append(copy(agent))
```

**Why it works**:
- Can't over-fit to single opponent
- Must remain robust to variety
- Historical sampling smooths learning

### 4. Melee Turing Test

> "Agents pass an informal 'Melee Turing Test', simply through the use of a strong behavioral prior."

Human observers couldn't reliably distinguish BC agents from humans in short clips. This validates that:
- BC captures human style
- Tech skill emerges naturally
- Motion and timing are human-like

### 5. Emergent Strategies from RL

After self-play refinement:
- Agents develop strategies not in training data
- Spacing and timing improve
- Decision-making becomes more consistent

But only with population-based training—single self-play leads to degenerate strategies.

## Research Context

### Why Slippi Enabled This

Previous Melee AI (Phillip) required:
- Custom infrastructure for replay recording
- Real-time training only
- Limited data availability

Slippi provided:
- Standardized replay format (.slp)
- Massive community replay datasets
- Fast-forward capability for RL

### Game Complexity

From the blog:

> "From a research perspective, the game presents a rich and complex action space (two continuous inputs and 4 independent buttons), which allows players to execute a variety of difficult maneuvers known as 'tech skill.' Players also have to make decisions under uncertainty of the opponent's actions, which has led to development of detailed 'flowcharts', or set strategies that cover different options."

## Twitch Integration

Nabla integrated with Twitch for live testing:

> "Because bots are not allowed on unranked currently, we integrated with the Twitch API so any viewer can add themselves to a queue to play a direct match against the bot."

This allowed:
- Real-world testing against humans
- Community feedback on agent quality
- Data collection from varied opponents

## Lessons for Self-Play

### What Doesn't Work

1. **Naive self-play**: Rock-paper-scissors cycling
2. **Fixed opponent**: Over-fitting to single strategy
3. **No BC prior**: Pure RL too slow to converge

### What Works

1. **BC initialization**: Strong prior from human data
2. **Historical sampling**: Play against past checkpoints
3. **Population diversity**: Multiple concurrent agents
4. **Gradual refinement**: KL penalty to stay near BC

### Recommended Approach

```python
# Initialize with BC
agent = train_bc(replay_data)
population = [copy(agent)]

# Self-play with population
for step in range(max_steps):
    # Sample from population (not just current)
    opponent = sample(population, strategy='uniform')

    # Collect trajectories
    trajectories = play(agent, opponent)

    # Update with PPO + KL penalty to BC
    ppo_loss = compute_ppo_loss(trajectories)
    kl_loss = kl_divergence(agent, bc_agent)

    agent.update(ppo_loss + kl_weight * kl_loss)

    # Periodically add to population
    if step % snapshot_interval == 0:
        population.append(copy(agent))
```

## Comparison to Other Projects

| Aspect | Nabla | slippi-ai | Phillip |
|--------|-------|-----------|---------|
| BC Stage | Yes | Yes | No |
| Self-Play | Population | Historical | Single |
| Published | Blog | Code | Paper |
| Focus | Research | Production | Research |

## Relevance to ExPhil

### Must-Apply Lessons

1. **Never use single-opponent self-play**: Use population or historical sampling
2. **BC is essential**: Pure RL from scratch is impractical
3. **KL regularization**: Stay close to BC policy during RL

### Implementation Recommendations

```elixir
# ExPhil self-play configuration
defmodule ExPhil.SelfPlay.Config do
  defstruct [
    population_size: 10,
    snapshot_interval: 1000,
    opponent_sampling: :historical,  # or :population
    kl_weight: 0.003,
    kl_decay: 0.999
  ]
end
```

### BEAM Advantages

Elixir/BEAM naturally supports population-based training:
- GenServer per agent
- Supervisor for population
- Easy message passing for rollouts
- Built-in concurrency

## Limitations

### What Nabla Didn't Solve

1. **Superhuman play**: Agents are "low-level human" quality
2. **Real-time inference**: Not optimized for 60 FPS
3. **Character generalization**: Trained per-character
4. **Open source code**: Not publicly released

### Open Questions

1. How large should population be?
2. Optimal snapshot frequency?
3. Best opponent sampling strategy?
4. How to balance exploration vs exploitation?

## References

- [Technical Blog Post](https://bycn.github.io/2022/08/19/project-nabla-writeup.html)
- [Twitter Announcement](https://twitter.com/otter_collapse/status/1561445156246753280)
- [AlphaStar](https://www.nature.com/articles/s41586-019-1724-z) - League training inspiration
- [Slippi](https://slippi.gg/) - Data and infrastructure
