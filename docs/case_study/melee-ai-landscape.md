# Melee AI Landscape

This document provides a comprehensive overview of the Super Smash Bros. Melee AI research space, covering technical challenges, approaches, and the current state of the field.

## The Game

Super Smash Bros. Melee (2001) is a fast-paced platform fighting game with:
- 26 playable characters with unique movesets
- 60 FPS gameplay with frame-precise inputs
- Complex physics engine with character-specific properties
- Rich competitive scene with 20+ years of metagame development

## Why Melee is Hard for AI

### 1. Action Space Explosion

**Raw input space**: ~30 billion combinations per frame
- 8 binary buttons (2^8 = 256 combinations)
- Main stick: 256×256 positions (65,536)
- C-stick: 256×256 positions (65,536)
- Shoulder triggers: 256 positions each

**Effective space after discretization**: ~5,000-50,000 actions
- Still orders of magnitude larger than chess, Go, or Atari

### 2. Temporal Complexity

**Long-horizon planning required**:
- Combos span 30-200+ frames
- Edgeguards require predicting recovery paths
- Neutral game involves reading opponent habits over matches

**Frame-perfect timing**:
- L-cancel window: 7 frames
- Wavedash timing: 3-frame jump squat
- Powershield window: 4 frames
- Many techniques require 1-2 frame precision

### 3. Partial Observability

**Hidden information**:
- Some action states ambiguous in early frames
- RNG-dependent moves (G&W hammer, Peach turnips)
- Opponent intent not directly observable

**Delayed feedback**:
- Minimum 1 frame input → execution delay
- Online play: 18+ frames additional delay
- Must predict future states

### 4. Character Diversity

**Asymmetric gameplay**:
- 26 characters × 25 opponents = 650 matchups
- Each character has unique:
  - Movement physics (fall speed, air acceleration)
  - Attack properties (frame data, hitboxes)
  - Recovery options (distance, angles, mixups)
  - Tech skill requirements

### 5. Metagame Depth

**Human optimization over 20 years**:
- Advanced techniques unknown to original developers
- Wavedashing, L-canceling, dashdancing
- Character-specific tech (multishine, wobbling, pillaring)
- Constantly evolving metagame

## Major Approaches

### Pure Reinforcement Learning (Phillip, 2017)

**Method**: Actor-Critic with self-play
**Pros**: No human data dependency, can discover novel strategies
**Cons**: Sample inefficient, slow to converge, needs massive compute

```
Self-Play Loop:
  Agent vs Agent → Trajectories → Gradient Updates → Better Agent
```

**Key results**:
- Beat professional players at Genesis 4
- Required months of distributed training
- 33ms reaction time (unrealistic superhuman)

### Behavioral Cloning + RL (slippi-ai, 2020+)

**Method**: Two-stage training
1. BC: Learn from 100K+ human replays
2. RL: Refine with PPO self-play

**Pros**: Fast bootstrap, human-like play, practical compute requirements
**Cons**: Limited by replay data quality, may inherit human biases

```
Stage 1: Human Replays → Supervised Learning → BC Policy
Stage 2: BC Policy → Self-Play RL → Refined Policy
```

**Key results**:
- Competitive with top 20 professionals
- 18-frame delay tolerance (realistic online play)
- Days-weeks training vs months

### Transformer-Based (Eric Gu, 2024)

**Method**: Next-token prediction (GPT-style)
**Architecture**: 20M parameter decoder-only Transformer
**Data**: 3 billion frames of professional replays

**Pros**: Simple training objective, scales with data
**Cons**: Limited exploration, pure imitation ceiling

**Key results**:
- 95% vs Level 9 CPU
- $5 training cost, 5 hours on 2× 3090s
- All-character model > single-character

### Rule-Based (SmashBot)

**Method**: Hand-coded behavior trees
**Pros**: Frame-perfect execution, explainable, no training
**Cons**: No adaptation, doesn't scale to matchups

**Key insight**: Demonstrates ceiling of explicit programming

## Technical Challenges Deep Dive

### Frame Delay Problem

```
Frame N:   Observe state S_N
Frame N+1: Input from S_N takes effect
Frame N+18: Input from S_N visible in online play

Challenge: Must predict 18 frames ahead for online!
```

**Solutions**:
1. Train with explicit delay buffer
2. Recurrent networks for implicit prediction
3. Model-based state prediction (Phillip)

### Action Space Discretization

**Naive approach fails**:
```python
# Too coarse: 5 positions per axis
positions = [-1, -0.5, 0, 0.5, 1]  # Misses shield drops, SDI angles

# Too fine: 256 positions
positions = range(256)  # Too many actions, slow convergence
```

**Practical approach**:
```python
# K-means on replay data: ~21 clusters per axis
# Matches engine precision, captures common inputs
positions = cluster_centers_from_replays(k=21)
```

**ExPhil uses**: 9 classes per axis (81 total positions)

### Embedding Design

**State representation matters**:

| Approach | Dims | Notes |
|----------|------|-------|
| Raw pixels | ~100K | Too high-dimensional |
| Hand-crafted | ~100 | Misses information |
| Comprehensive | ~1200 | Sweet spot |
| With projectiles | ~1500 | Full game state |

**Key features**:
- Position (scaled)
- Velocity (5 components!)
- Action state (one-hot or learned)
- Character (one-hot)
- Shield, stocks, damage
- Stage-specific (platforms, hazards)

### Self-Play Stability

**Problem**: Rock-paper-scissors cycles in single-opponent self-play

```
Agent A beats Agent B
Agent B' beats Agent A
Agent A' beats Agent B'
→ Policy oscillates, never converges
```

**Solutions**:
1. **Historical sampling**: Play against past checkpoints
2. **Population-based**: Maintain diverse agent pool
3. **League system**: AlphaStar-style hierarchy
4. **KL regularization**: Stay close to BC teacher

## Current State of the Art

### Best Results (2024)

| Project | Characters | vs Human | vs CPU | Inference |
|---------|------------|----------|--------|-----------|
| slippi-ai | Fox, Falco | ~60% vs Top 20 | 95%+ | ~50ms |
| Eric Gu | All | N/A | 95% | ~10ms |
| SmashBot | Fox | Predictable | 90%+ | <1ms |

### Open Challenges

1. **Multi-character generalization**: One model for all characters
2. **Real-time inference**: Sub-16ms for 60 FPS
3. **Adaptation**: Learn opponent habits during match
4. **Explanation**: Why did the agent make that decision?
5. **Creativity**: Discover novel techniques beyond human data

## Infrastructure Ecosystem

### Slippi

**What it provides**:
- Rollback netcode for online play
- .slp replay format (portable, parseable)
- Massive replay datasets
- Spectator mode for data collection

**Impact**: Enabled all modern Melee AI research

### libmelee

**What it provides**:
- Python API to Dolphin emulator
- Real-time game state access
- Controller input abstraction
- Menu navigation utilities

**Impact**: Lowered barrier to entry significantly

### Dolphin

**What it provides**:
- Accurate Melee emulation
- Memory access for state reading
- Named pipes for controller input
- Slippi integration

## Research Timeline

| Year | Milestone |
|------|-----------|
| 2017 | Phillip beats professionals at Genesis 4 |
| 2018 | libmelee released, SmashBot development |
| 2020 | Slippi rollback enables massive online play |
| 2021 | slippi-ai BC achieves human-like play |
| 2022 | slippi-ai RL competitive with top players |
| 2023 | Mamba architecture shows promise |
| 2024 | Eric Gu's Transformer achieves 95% vs CPU |
| 2024+ | ExPhil targets low-tier characters |

## Key Papers and Resources

### Academic Papers

1. **Phillip**: "Beating the World's Best at SSBM with Deep RL" ([arXiv:1702.06230](https://arxiv.org/abs/1702.06230))
2. **Mamba**: "Linear-Time Sequence Modeling" ([arXiv:2312.00752](https://arxiv.org/abs/2312.00752))
3. **AlphaStar**: League-based population training ([Nature 2019](https://www.nature.com/articles/s41586-019-1724-z))

### Technical Blogs

1. [Eric Gu's Melee AI](https://ericyuegu.com/melee-pt1) - Transformer approach
2. [Project Nabla](https://bycn.github.io/2022/08/19/project-nabla-writeup.html) - Self-play lessons
3. [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) - Scale vs inductive bias

### Community Resources

- [SmashWiki](https://www.ssbwiki.com/) - Game mechanics
- [IKneeDta](https://ikneedata.com/) - Frame data
- [Slippi.gg](https://slippi.gg/) - Replays and rankings

## Future Directions

### Near-term (1-2 years)

1. **Multi-character models**: Single network, all characters
2. **Real-time inference**: ONNX/TensorRT optimization
3. **Better self-play**: Population-based without instability

### Medium-term (2-5 years)

1. **Superhuman low-tier**: Mewtwo, G&W, Link at top level
2. **In-match adaptation**: Learn opponent patterns live
3. **Human-AI collaboration**: AI-assisted training tools

### Long-term (5+ years)

1. **Novel technique discovery**: Beyond human metagame
2. **General fighting game AI**: Transfer across games
3. **Explainable agents**: Understand decision-making

## Relevance to ExPhil

ExPhil positions itself uniquely:

| Aspect | slippi-ai | ExPhil |
|--------|-----------|--------|
| Language | Python/TF | Elixir/Nx |
| Backbone | GRU | Mamba |
| Characters | Top-tier | Low-tier |
| Inference | ~50ms | ~9ms |
| Concurrency | Multi-process | BEAM native |

**Strategic advantages**:
1. Mamba enables longer context (90+ frames)
2. Elixir/BEAM natural for self-play concurrency
3. Low-tier focus = less competitive, more exploration room
4. ONNX export = deployment flexibility
