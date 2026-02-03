# Q-Learning Melee Case Study

**Repository**: https://github.com/kape142/Machine-learning-melee
**Author**: kape142
**Status**: Inactive
**Language**: Python
**License**: LGPL-3.0

## Overview

An educational project demonstrating **tabular Q-learning** applied to Melee. Represents the simplest RL approach, useful for understanding fundamentals before scaling to deep RL.

## Algorithm

### Q-Value Update

```
Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
```

### Hyperparameters

| Parameter | Initial | Decay | Minimum |
|-----------|---------|-------|---------|
| Alpha (learning rate) | 0.2 | 0.0002 | 0 |
| Epsilon (exploration) | 0.99 | 0.0007 | 0.05 |
| Gamma (discount) | 0.9 | — | — |

### Training Scale

- **Episodes**: 10,000
- **Duration**: 20 seconds (~1,200 frames)
- **Agents**: 2 simultaneous Q-learning bots

## State Representation

**4 dimensions** (reduced/discretized):

| Dimension | Range | Description |
|-----------|-------|-------------|
| Agent action | 0-29 | Quantized animation ID |
| Opponent action | 0-29 | Opponent's animation |
| Distance | 0-20 | Discretized, non-linear scaling |
| Agent X position | -10 to 10 | Horizontal position |

**State space**: ~396,900 states (manageable for tabular Q)

## Action Space

**13 discrete actions**:
- Button actions (A, R, Z)
- Stick positions (various presets)
- Lambda functions map indices to controller inputs

## Reward Function

| Event | Reward |
|-------|--------|
| Base decay | -1/60 per frame |
| Damage dealt | Positive |
| Action penalties | Negative for loops |

## Dolphin Integration

Uses libmelee with custom Dolphin fork:

```
Dolphin (memorywatcher branch)
    ↓
libmelee Library
    ↓
MeleeBot Environment (Gym API)
    ↓
Q-Learning Training Loop
```

**Menu automation**: Character/stage defaults (Falco, Final Destination)

## Code Structure

```
Machine-Learning-Melee/
├── melee/
│   ├── gamestate.py
│   ├── memorywatcher.py
│   ├── controller.py
│   └── dolphin.py
├── meleebot/
│   └── __init__.py      # MeleeBot Gym env
├── q_learning.py        # Main training
├── q_learning_2qtables.py
├── q_table_benchmarking.py
├── plotting_reward.py
└── Stored_results/
```

## Visualization

`plotting_reward.py` generates:
1. Cumulative rewards over episodes
2. Opponent damage % over episodes
3. Action-specific metrics

## Comparison: Q-Learning vs ExPhil

| Aspect | Q-Learning | ExPhil |
|--------|------------|--------|
| Algorithm | Tabular Q | Deep RL (PPO) |
| State | 4-dim discrete | 1204-dim continuous |
| Network | Q-table (lookup) | Neural network |
| Data source | Self-play only | Replays + self-play |
| Action space | 13 discrete | Autoregressive |
| Inference | <1ms (lookup) | 0.55-8ms |
| Scalability | Limited | Generalizable |

## Limitations

1. **State explosion**: Can't add more features without exponential growth
2. **No generalization**: Each state learned independently
3. **No temporal modeling**: Single-frame decisions
4. **Limited actions**: Coarse discretization

## Lessons for ExPhil

**Educational value**:
- Demonstrates RL fundamentals
- Shows why deep learning is necessary for Melee
- Baseline for comparison

**Why deep RL is needed**:
- Rich state representation (1204 dims)
- Function approximation (generalization)
- Temporal patterns (combos, mixups)
- Continuous action refinement

## References

- [Repository](https://github.com/kape142/Machine-learning-melee)
- [Q-Learning (Wikipedia)](https://en.wikipedia.org/wiki/Q-learning)
- [libmelee](https://github.com/altf4/libmelee)
