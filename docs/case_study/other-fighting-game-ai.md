# Other Fighting Game AI

This document covers AI research in fighting games and real-time strategy games beyond Melee, providing transferable lessons for ExPhil.

## Why Study Other Game AI?

Fighting games share common challenges:
- Real-time decision making (16-33ms frame budgets)
- Large action spaces
- Opponent modeling and adaptation
- Reaction time constraints

Lessons from these projects directly apply to Melee AI development.

---

## AlphaStar (StarCraft II)

**Organization**: DeepMind
**Paper**: [Dota 2 with Large Scale Deep Reinforcement Learning](https://arxiv.org/abs/1912.06680)
**Blog**: [deepmind.google/blog/alphastar](https://deepmind.google/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii/)
**Achievement**: Grandmaster rank (top 0.2%) in all three races

### Architecture

```
Observation (list of units + properties)
    │
    ▼
┌─────────────────────────────────────┐
│  Transformer Encoder (units)        │
│  Similar to relational deep RL      │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  Deep LSTM Core                     │
│  4096-unit LSTM (84% of params)     │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  Auto-regressive Policy Head        │
│  Pointer network for action args    │
└─────────────────────────────────────┘
```

### Key Technical Details

| Aspect | Details |
|--------|---------|
| **Action Space** | 10²⁶ possible actions per timestep |
| **Effective Actions** | ~75,000 distinct per step |
| **Frame Rate** | Acts every 4th frame (30 FPS engine) |
| **Reaction Time** | 217ms average (human: ~250ms) |
| **Training Compute** | 256 P100 GPUs + 128K CPU cores |
| **Training Time** | 180 years of gameplay per day |

### Training Approach

1. **Imitation Learning**: Bootstrap from human replays (84th percentile)
2. **Self-Play RL**: Refine with population-based training
3. **League Training**: Diverse opponent pool prevents cycling

### Lessons for ExPhil

1. **Transformer + LSTM**: Hybrid architecture works for complex games
2. **Auto-regressive Actions**: Essential for large action spaces
3. **Population Training**: Prevents rock-paper-scissors collapse
4. **Human-like Reactions**: 217ms is achievable and fair
5. **Imitation Bootstrap**: Critical for sample efficiency

---

## OpenAI Five (Dota 2)

**Organization**: OpenAI
**Paper**: [Dota 2 with Large Scale Deep Reinforcement Learning](https://arxiv.org/abs/1912.06680)
**Blog**: [openai.com/index/openai-five](https://openai.com/index/openai-five/)
**Achievement**: Defeated world champions (OG) at TI8

### Architecture

```
Game State (20,000 numbers from API)
    │
    ▼
┌─────────────────────────────────────┐
│  Single-layer LSTM (4096 units)     │
│  One per hero (5 total)             │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  Multiple Action Heads              │
│  8 enumeration values per action    │
└─────────────────────────────────────┘
```

### Key Technical Details

| Aspect | Details |
|--------|---------|
| **Observation** | 20,000 numbers from game API |
| **Action Output** | 8 enumeration values |
| **Reaction Time** | 217ms average |
| **Training** | PPO with 2M frames/2 seconds |
| **Scale** | 10 months continuous training |
| **LSTM Size** | 4096 units (84% of parameters) |

### What Was Hand-Scripted

Even OpenAI Five used scripted logic for:
- Item purchase order
- Ability upgrade order
- Courier control
- Item inventory management

**Lesson**: Some game mechanics are better scripted than learned.

### Lessons for ExPhil

1. **Simpler Architecture Works**: Single LSTM layer sufficient
2. **Massive Scale**: 180 years of gameplay/day
3. **Reaction Time**: 217ms is competitive with humans
4. **Hybrid Approach**: Script some mechanics, learn others
5. **Long Training**: 10 months for superhuman performance

---

## FightingICE / DareFightingICE

**Organization**: Ritsumeikan University
**Website**: [ice.ci.ritsumei.ac.jp/~ftgaic](https://www.ice.ci.ritsumei.ac.jp/~ftgaic/index-2.html)
**Paper**: [DareFightingICE Competition](https://arxiv.org/abs/2203.01556)
**Status**: Active annual competition (IEEE CIS sponsored)

### Competition Overview

Academic fighting game AI competition running since 2013. DareFightingICE (2022+) adds sound design component.

**Tracks**:
1. **AI Track**: Develop fighting game AI
2. **Sound Design Track**: Create audio for blind AI
3. **Standard League**: Visual input AI
4. **Blind League**: Sound-only input AI

### Technical Specs

| Aspect | Limit |
|--------|-------|
| **RAM** | 64GB max |
| **VRAM** | 32GB max |
| **CPU Threads** | 16 max |
| **GPU** | Allowed |

### Common Approaches

**Deep Q-Networks (DQN)**:
- Used in multiple winning entries
- Action prediction accuracy up to 98%

**Proximal Policy Optimization (PPO)**:
- Sample deep-learning blind AI uses PPO
- Provided as competition baseline

### Available Resources

- [FighterZero](https://github.com/MatejVitek/FighterZero) - Deep learning AI implementation
- [TeamFightingICE/FightingICE](https://github.com/TeamFightingICE/FightingICE) - Competition framework
- Sample deep RL blind AI provided

### Lessons for ExPhil

1. **Standardized Benchmarks**: Competition enables comparison
2. **Blind AI Challenge**: Sound-only demonstrates generalization
3. **Academic Rigor**: Published papers with reproducible results
4. **Resource Limits**: Competition specs are reasonable (64GB RAM)

---

## Street Fighter AI Research

### Street Fighter V Research (2022)

**Paper**: [A Study on the Agent in Fighting Games Based on Deep Reinforcement Learning](https://onlinelibrary.wiley.com/doi/10.1155/2022/9984617)
**Method**: Double Deep Q-Network (DDQN)
**Result**: 95% win rate after 2,590 rounds

### Street Fighter II Projects

**HadouQen** ([Paper](https://www.isujournals.ph/index.php/ject/article/view/213)):
- PPO-based agent
- 96.7% win rate vs M. Bison
- 100M timesteps training
- 16 parallel environments

**Open Source**:
- [street-fighter-ai](https://github.com/linyiLYi/street-fighter-ai) - PPO implementation
- [AIVO-StreetFighterRL](https://github.com/corbosiny/AIVO-StreetFigherReinforcementLearning) - Tournament training

### Pro-Level Fighting Game AI (Blade & Soul)

**Paper**: [Creating Pro-Level AI for Real-Time Fighting Game with Deep Reinforcement Learning](https://www.researchgate.net/publication/332301037)
**Game**: Blade & Soul (1v1 arena)
**Result**: 62% win rate vs professional players

**Key Challenges Addressed**:
- Vast action spaces
- Real-time constraints
- Opponent generalization

### Lessons for ExPhil

1. **DDQN/PPO Both Work**: Standard algorithms effective
2. **High Win Rates Achievable**: 95%+ vs AI possible
3. **Pro-Level Attainable**: 62% vs pros is reachable
4. **Parallel Training**: 16+ environments standard

---

## FightLadder Benchmark

**Paper**: [FightLadder: A Benchmark for Competitive Multi-Agent Reinforcement Learning](https://arxiv.org/html/2406.02081v2)
**Supported Games**: Street Fighter, Mortal Kombat, Fatal Fury, King of Fighters

### Why It Matters

Provides standardized evaluation across multiple fighting games:
- Consistent API for different emulators
- Reproducible benchmarks
- Multi-agent focus

### Lessons for ExPhil

1. **Standardization Value**: Comparable results across projects
2. **Multi-Game Generalization**: Same techniques work across titles
3. **Benchmark Importance**: Need clear evaluation criteria

---

## Comparison Table

| Project | Game | Architecture | Training | Reaction Time | Result |
|---------|------|--------------|----------|---------------|--------|
| AlphaStar | SC2 | Transformer+LSTM | IL→RL | 217ms | GM rank |
| OpenAI Five | Dota 2 | LSTM | RL | 217ms | Beat OG |
| FightingICE | Custom | DQN/PPO | RL | Variable | Competition |
| SF5 Research | SF5 | DDQN | RL | N/A | 95% WR |
| Blade & Soul | B&S | Deep RL | RL | N/A | 62% vs pros |
| slippi-ai | Melee | LSTM/Transformer | IL→RL | 300ms (18f) | Top 20 |

---

## Common Patterns

### What Works

1. **Imitation Learning Bootstrap**: All major projects use human data
2. **PPO/DDQN**: Standard algorithms are sufficient
3. **LSTM/Transformer Cores**: Temporal modeling essential
4. **Auto-regressive Actions**: For large action spaces
5. **Population/League Training**: Prevents self-play collapse
6. **Human-like Reactions**: 200-300ms is fair and achievable

### What Scales

| Compute | Result |
|---------|--------|
| Single GPU | Competition-level |
| Multi-GPU | Pro-level |
| Cluster (256+ GPU) | Superhuman |

### Fighting Game Specific

1. **Frame-perfect timing** less important than strategy
2. **Opponent adaptation** remains unsolved
3. **Reaction time** must be constrained for fairness
4. **Action spaces** are large but manageable with discretization

---

## Relevance to ExPhil

### Direct Applications

| Concept | Source | Application |
|---------|--------|-------------|
| Transformer+LSTM | AlphaStar | Mamba is similar linear-time alternative |
| 217ms reaction | OpenAI Five | 18-frame delay = 300ms, close |
| Population training | AlphaStar | Avoid single self-play |
| PPO | FightingICE | Already using in ExPhil |
| Auto-regressive | AlphaStar | Controller head design |

### ExPhil Advantages

1. **Simpler Game**: Melee is 1v1, smaller state than Dota/SC2
2. **Rich Replay Data**: Slippi ecosystem provides data
3. **Active Community**: Testing and feedback available
4. **Defined Scope**: Low-tier characters, not all 26

### ExPhil Challenges

1. **Frame Timing**: 60 FPS stricter than SC2's 30
2. **Technical Execution**: Melee rewards frame-perfect inputs
3. **Community Expectations**: Players know the game deeply
4. **Character Diversity**: Each low-tier plays differently

---

## References

### Papers
- [Grandmaster level in StarCraft II](https://www.nature.com/articles/s41586-019-1724-z) (Nature)
- [Dota 2 with Large Scale Deep RL](https://arxiv.org/abs/1912.06680)
- [DareFightingICE Competition](https://arxiv.org/abs/2203.01556)
- [Pro-Level Fighting Game AI](https://www.researchgate.net/publication/332301037)

### Repositories
- [FighterZero](https://github.com/MatejVitek/FighterZero)
- [street-fighter-ai](https://github.com/linyiLYi/street-fighter-ai)
- [FightLadder](https://arxiv.org/html/2406.02081v2)

### Blogs
- [DeepMind AlphaStar](https://deepmind.google/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii/)
- [OpenAI Five](https://openai.com/index/openai-five/)
