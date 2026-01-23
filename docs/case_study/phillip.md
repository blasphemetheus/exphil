# Phillip Case Study

**Repository**: https://github.com/vladfi1/phillip
**Author**: vladfi1
**Status**: Deprecated (historical reference)
**Language**: Python / TensorFlow 1.x → 2.x
**Paper**: [Beating the World's Best at SSBM with Deep RL](https://arxiv.org/abs/1702.06230)

## Overview

Phillip is the original deep reinforcement learning Melee bot, achieving landmark results by beating professional players at Genesis 4 (2017). Unlike its successor slippi-ai, Phillip uses **pure reinforcement learning** without imitation learning from human replays.

**Key Achievement**: First AI to beat professional Melee players in tournament conditions, demonstrating that deep RL can master complex fighting games.

**Deprecation Note**: From README:
> "This project is no longer active and is subject to bit-rot. There is a successor project based on imitation learning from slippi replays at https://github.com/vladfi1/slippi-ai."

## Why Study Phillip?

1. **Historical Context**: Understand what pure RL can achieve
2. **Architecture Insights**: Actor-Critic patterns still relevant
3. **Failure Modes**: Learn why imitation learning bootstrapping won out
4. **Distributed Training**: Infrastructure for multi-agent learning

## Architecture

### Actor-Critic Framework

```
┌─────────────────────────────────────────┐
│            Game State + History          │
│  (Position, velocity, action, %)        │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│           Trunk Layers (MLP)            │
│  Non-recurrent feature extraction       │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│         Core (Optional GRU)             │
│  Recurrent temporal modeling            │
└────────────────┬────────────────────────┘
                 │
          ┌──────┴──────┐
          ▼             ▼
┌─────────────┐  ┌─────────────┐
│   Actor     │  │   Critic    │
│ (Policy)    │  │  (Value)    │
│ 2×128 MLP   │  │  2×128 MLP  │
│ → Actions   │  │  → V(s)     │
└─────────────┘  └─────────────┘
```

### Core Network Options

| Component | Default | Description |
|-----------|---------|-------------|
| Trunk | Optional | Feedforward pre-processing |
| Core | GRU(256) | Recurrent backbone |
| Actor | 2×128 | Policy MLP |
| Critic | 2×128 | Value MLP |

### Predictive Model (Optional)

For handling frame delay, Phillip can train a state prediction model:

```python
class Model:
    # Predicts future state for "undoing" delay
    delta_layer   # Additive change prediction
    new_layer     # Full state replacement prediction
    forget_layer  # Interpolation weight (0=keep, 1=replace)

    # Output: (1-forget) * current + forget * new + delta
```

## State Representation

### Player Embedding

Each player encoded with ~19 base dimensions:

| Feature | Description |
|---------|-------------|
| `percent` | Damage % (scaled 0.01) |
| `facing` | Direction (-1 or +1) |
| `x, y` | Position (scaled 0.1) |
| `action_state` | One-hot over 383 actions |
| `action_frame` | Frame within action |
| `character` | One-hot (optional) |
| `invulnerable` | Boolean |
| `hitlag_frames_left` | Scaled |
| `hitstun_frames_left` | Scaled |
| `jumps_used` | Current / max |
| `charging_smash` | Boolean |
| `shield_size` | Remaining % (scaled 0.01) |
| `in_air` | Boolean |
| Velocities | 5 dims (air/ground/attack) |

### Action History

- Previous actions concatenated as context
- One-hot encoded: 383 dims each
- Configurable memory window (default 1 frame)

## Action Space

Phillip uses discretized action sets (not continuous):

### Action Set Options

| Type | Size | Description |
|------|------|-------------|
| `old` | 30 | 5 stick × 6 buttons |
| `cardinal` | ~30 | Cardinal directions + 6 buttons |
| `diagonal` | 78 | 3×3 stick grid + 6 buttons |
| `custom` | ~50 | Mixed granularity with chains |

### SimpleController

```python
@dataclass
class SimpleController:
    button: Button  # NONE, A, B, Z, Y, L
    stick: Tuple[float, float]  # (x, y) in [0,1]×[0,1]
    duration: int  # Frames to hold
```

### Action Chains

Complex techniques encoded as action sequences:
```python
short_hop_chain = [
    (Y_button, 2_frames),
    (neutral, N_frames)
]

wavedash_chain = [
    (Y_button, 2_frames),
    (L + angle, timing_frames)
]
```

## Training Methodology

### Pure RL Loop

```
1. Actor processes play against opponents
2. Generate trajectories (state, action, reward)
3. Send experience to learner
4. Learner computes gradients:
   - Actor loss: -mean(log_probs * advantages + entropy_scale * entropy)
   - Critic loss: mean(advantages²)
   - Total: critic_weight * critic + actor_weight * actor
5. Update weights, broadcast to actors
6. Repeat indefinitely
```

### Distributed Architecture

```
┌─────────────────────────────────────────┐
│              Learner Process             │
│  - Receives experience via nanomsg      │
│  - Computes gradients                   │
│  - Broadcasts updated weights           │
└────────────────┬────────────────────────┘
                 │ nanomsg
    ┌────────────┼────────────┐
    ▼            ▼            ▼
┌────────┐  ┌────────┐  ┌────────┐
│ Actor  │  │ Actor  │  │ Actor  │
│ (Env1) │  │ (Env2) │  │ (Env3) │
└────────┘  └────────┘  └────────┘
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 1e-4 | Adam LR |
| `entropy_scale` | 1e-3 | Exploration bonus |
| `experience_length` | 80 | Trajectory length |
| `batch_size` | 64 | Trajectories per batch |
| `gae_lambda` | 1.0 | GAE (full Monte-Carlo) |
| `reward_halflife` | 4s | For discount factor |
| `epsilon` | 0.02 | Epsilon-greedy |

### Reward Function

```python
def compute_reward(state, next_state):
    # Base: Stock differential
    stock_diff = opponent_stocks_lost - self_stocks_lost

    # Damage component
    damage_diff = damage_dealt - damage_taken

    # Combined
    reward = stocks + damage_ratio * damage_diff

    # Optional: distance-based shaping
    # Optional: action entropy bonuses
```

### Training Opponents

| Opponent Type | Description |
|---------------|-------------|
| `cpu` | Level 9 CPU |
| `delay0` | Self with 0 delay |
| `delay18` | Self with 18 frame delay |
| `self` | Current policy |
| `historical` | Past checkpoints |

## Why Phillip Was Deprecated

### 1. Sample Efficiency

Pure RL requires massive compute:
- Millions of frames just to learn basic movement
- Billions for competitive play
- slippi-ai bootstraps from 100K human games

### 2. Data Availability

Slippi replay infrastructure changed the landscape:
- Millions of human games freely available
- High-quality demonstrations from professionals
- Phillip had to generate everything from scratch

### 3. Generalization

Character training overhead:
- Phillip: Separate training per character
- slippi-ai: Train on any character with replays

### 4. Skill Discovery

Human demonstrations accelerate technique learning:
- Phillip must "discover" wavedashing via exploration
- slippi-ai learns it directly from human examples

### 5. Maintenance Burden

Complex distributed infrastructure:
- SLURM clusters required for training
- nanomsg communication layer
- Difficult to reproduce

## Lessons Learned

### What Worked

1. **Actor-Critic**: Stable RL architecture for fighting games
2. **GRU Core**: Temporal dependencies essential for combos
3. **Discretized Actions**: Continuous action space too hard
4. **Entropy Regularization**: Prevents policy collapse
5. **Distributed Training**: Parallelism necessary for RL scale

### What Didn't

1. **Pure RL from Scratch**: Too slow to be practical
2. **Single-Opponent Self-Play**: Leads to rock-paper-scissors cycles
3. **Frame-Perfect Reactions**: 33ms (2 frame) reaction was unrealistic
4. **Fixed Action Sets**: Not flexible enough for all situations

### Key Insights

> "Behavioral cloning provides a strong prior that RL can refine. Starting from scratch with RL is possible but impractical for complex games."

> "Population-based or historical sampling required for sustainable self-play. Single-opponent leads to policy collapse."

## Performance

### Reaction Time Issue

| Agent | Reaction | Notes |
|-------|----------|-------|
| Phillip | 33ms (~2 frames) | Superhuman |
| Human | 200ms (~12 frames) | Realistic |
| slippi-ai | Configurable | 18 frame delay |

Phillip's fast reactions gave unrealistic advantage.

### Trained Agents

Best agents available in `agents/` directory:
- `delay18/FalcoBF` - Falco on Battlefield (best)
- `FalconFalconBF` - Captain Falcon (~1.1M params)
- Various Fox, Marth, Peach, Sheik agents

## Code Structure

```
phillip/
├── phillip/              # Core package
│   ├── RL.py            # Base RL framework
│   ├── learner.py       # Gradient computation
│   ├── actor.py         # Policy sampling
│   ├── ac.py            # Actor-Critic policy
│   ├── critic.py        # Value network
│   ├── model.py         # State prediction
│   ├── core.py          # GRU backbone
│   ├── embed.py         # State embeddings
│   ├── ssbm.py          # Melee types
│   ├── reward.py        # Reward computation
│   ├── train.py         # Training loop
│   └── runner.py        # Parameter configs
├── agents/              # Trained checkpoints
├── enemies/             # Opponent configs
├── scripts/             # Helper scripts
├── docker/              # Containerization
└── server/              # Distributed training
```

## Relevance to ExPhil

### Applicable Concepts

1. **Actor-Critic**: Still valid architecture pattern
2. **Entropy Regularization**: Important for exploration
3. **GRU Backbone**: Though Mamba may be better
4. **Action Chains**: Complex technique encoding
5. **Delay Handling**: Predictive model concept

### What to Avoid

1. **Pure RL Start**: Use BC to bootstrap
2. **Single Self-Play**: Use population/historical
3. **Superhuman Reactions**: Train with realistic delay
4. **Complex Distributed Infra**: Keep it simple

## References

- [Repository](https://github.com/vladfi1/phillip)
- [Paper: Beating the World's Best at SSBM](https://arxiv.org/abs/1702.06230)
- [Genesis 4 Matches](https://www.youtube.com/watch?v=1Zj9M-n9V6Q) (YouTube)
