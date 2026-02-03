# Melee AI Research Papers and Resources

This document catalogs key academic papers, technical blog posts, and resources relevant to Melee AI development.

## Core Papers

### 1. Beating the World's Best at SSBM with Deep RL (Phillip)

**Citation**: Firoiu, V., Whitney, W. F., & Tenenbaum, J. B. (2017)
**Link**: [arXiv:1702.06230](https://arxiv.org/abs/1702.06230)

**Key contributions**:
- First deep RL agent to beat professionals
- Actor-Critic architecture for fighting games
- Demonstrated pure RL viability for complex games

**Architecture**:
- GRU-based recurrent policy
- Discretized action space (~30-78 actions)
- Self-play with entropy regularization

**Results**:
- Beat professionals at Genesis 4 (2017)
- 33ms reaction time (superhuman)
- Falcon, Fox mains trained

**Lessons for ExPhil**:
- Pure RL is possible but expensive
- Reaction time must be realistic
- Self-play requires careful opponent selection

### 2. Mamba: Linear-Time Sequence Modeling

**Citation**: Gu, A., & Dao, T. (2023)
**Link**: [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)

**Key contributions**:
- O(n) sequence modeling (vs O(n²) Transformers)
- Selective state space models
- Competitive with Transformers at scale

**Why it matters for Melee**:
- Enables longer context windows (90+ frames)
- 5× throughput improvement
- Sub-10ms inference for real-time play

**ExPhil implementation**:
- Mamba backbone as primary architecture
- 8.93ms inference (60 FPS ready)
- Handles temporal dependencies efficiently

### 3. Proximal Policy Optimization (PPO)

**Citation**: Schulman, J., et al. (2017)
**Link**: [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

**Key contributions**:
- Stable policy gradient algorithm
- Clipped surrogate objective
- Works well with neural network policies

**PPO objective**:
```
L_CLIP = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
```

**Usage in Melee AI**:
- slippi-ai RL stage uses PPO
- ExPhil PPO trainer implemented
- Standard algorithm for game AI

### 4. AlphaStar: Mastering StarCraft II

**Citation**: Vinyals, O., et al. (2019)
**Link**: [Nature](https://www.nature.com/articles/s41586-019-1724-z)

**Key contributions**:
- League-based self-play training
- Population-based training for diversity
- Grandmaster-level play in RTS

**League system**:
```
Main Agents: General performance
League Exploiters: Find weaknesses
Main Exploiters: Counter exploiters
```

**Relevance to Melee**:
- Self-play stability techniques
- Population-based training patterns
- Handling rock-paper-scissors dynamics

### 5. Learning to Play SSBM with Delayed Actions

**Author**: Sharma, Y.
**Link**: [PDF](https://www.yash-sharma.com/files/learning-play-super%20(3).pdf)

**Key contributions**:
- Explicit delay modeling in training
- DRQN (Deep Recurrent Q-Network) approach
- Analysis of delay impact on policy quality

**Findings**:
- Training with delay teaches prediction
- Recurrent architectures essential for delay
- Performance degrades gracefully with delay increase

**ExPhil implementation**:
- `--online-robust` flag for 18-frame delay training
- Frame delay augmentation during BC
- Mamba naturally handles delayed sequences

## Technical Blog Posts

### 1. Training AI to Play Melee (Eric Gu)

**Link**: [ericyuegu.com/melee-pt1](https://ericyuegu.com/melee-pt1)

**Key insights**:
- 20M parameter decoder-only Transformer
- Next-token prediction (GPT-style)
- 3 billion frames training data

**Results**:
- 95% win rate vs Level 9 CPU
- $5 training cost
- 5 hours on 2× 3090s

**Key finding**: All-character model outperforms single-character models

**Implications**:
- Transformers work for fighting games
- Multi-character training beneficial
- Scale matters (compute, data)

### 2. Project Nabla Writeup

**Link**: [bycn.github.io](https://bycn.github.io/2022/08/19/project-nabla-writeup.html)

**Key insights**:
- BC agents learn modular skills
- Single-opponent self-play causes collapse
- Population-based training necessary

**Self-play failure modes**:
```
Rock beats Scissors
Paper beats Rock
Scissors beats Paper
→ Cycling, no convergence
```

**Solutions explored**:
- Historical sampling
- Population diversity
- Curriculum learning

### 3. The Bitter Lesson (Sutton)

**Link**: [incompleteideas.net](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)

**Core argument**:
- General methods + compute > specialized knowledge
- Search and learning scale better than hand-engineering
- Long-term: always bet on scale

**Implications for Melee**:
- Pure RL > rules (given enough compute)
- But we don't have unlimited compute
- BC+RL is pragmatic compromise

## Relevant ML Papers

### Imitation Learning

**DAgger**: Dataset Aggregation
- Interactive imitation learning
- Expert corrects distribution mismatch
- Harder to apply (needs live expert)

**GAIL**: Generative Adversarial Imitation Learning
- Adversarial training for imitation
- More robust than BC
- Complex to train

**BCO**: Behavioral Cloning from Observation
- Learn from state-only demonstrations
- Useful when actions aren't recorded
- Less relevant for replay-based learning

### Sequence Modeling

**LSTM**: Long Short-Term Memory
- Classic recurrent architecture
- Handles temporal dependencies
- Slower than Mamba for long sequences

**GRU**: Gated Recurrent Unit
- Simplified LSTM
- Often comparable performance
- Lighter compute requirements

**Transformer**: Attention Is All You Need
- O(n²) attention mechanism
- Excellent for variable-length sequences
- Memory-bound for very long contexts

### Fighting Game AI

**Fighting Game AI Overview**
**Link**: [IEEE CoG 2020](https://ieee-cog.org/2020/papers/paper_207.pdf)

- Survey of fighting game AI approaches
- Comparison of RL, IL, and hybrid methods
- Benchmark frameworks discussion

## Melee-Specific Resources

### Frame Data

**IKneeData**: [ikneedata.com](https://ikneedata.com/)
- Frame data for all characters
- Attack properties, hitbox data
- Essential for rule-based and reward design

**SmashWiki**: [ssbwiki.com](https://www.ssbwiki.com/)
- Comprehensive game mechanics
- Character-specific techniques
- Community-maintained knowledge base

### Community Tools

**Slippi**: [slippi.gg](https://slippi.gg/)
- Replay format and infrastructure
- Online matchmaking with rollback
- Massive replay dataset

**libmelee**: [github.com/altf4/libmelee](https://github.com/altf4/libmelee)
- Python API for Dolphin
- Game state access
- Controller input

### AI Projects

| Project | Link | Status |
|---------|------|--------|
| slippi-ai | [github](https://github.com/vladfi1/slippi-ai) | Active |
| Phillip | [github](https://github.com/vladfi1/phillip) | Deprecated |
| SmashBot | [github](https://github.com/altf4/SmashBot) | Active |
| ExPhil | local | Active |

## Reading Order Recommendation

### For Understanding the Landscape

1. **Start**: Phillip paper (original breakthrough)
2. **Then**: Eric Gu's blog (modern approach)
3. **Then**: Project Nabla (self-play lessons)
4. **Reference**: AlphaStar (advanced self-play)

### For Implementation

1. **Architecture**: Mamba paper
2. **Training**: PPO paper
3. **Delay**: Sharma's delayed actions paper
4. **Practice**: slippi-ai codebase

### For Philosophy

1. **The Bitter Lesson**: Scale vs inductive bias
2. **AlphaStar**: What's possible with resources
3. **Project Nabla**: What fails and why

## Citation Template

For papers using these resources:

```bibtex
@article{firoiu2017phillip,
  title={Beating the World's Best at Super Smash Bros. Melee with Deep Reinforcement Learning},
  author={Firoiu, Vlad and Whitney, William F and Tenenbaum, Joshua B},
  journal={arXiv preprint arXiv:1702.06230},
  year={2017}
}

@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}

@article{schulman2017ppo,
  title={Proximal Policy Optimization Algorithms},
  author={Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  journal={arXiv preprint arXiv:1707.06347},
  year={2017}
}
```
