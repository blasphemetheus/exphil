# Athena Case Study

**Repository**: https://github.com/Sciguymjm/athena
**Author**: Sciguymjm
**Status**: Inactive (October 2016)
**Language**: Python
**Approach**: Deep Q-Network (DQN)

## Overview

Athena is an early-stage reinforcement learning project that applies DQN to Super Smash Bros. Melee via Dolphin emulator integration. It represents one of the first attempts at using deep RL for Melee AI.

## Architecture

### DQN Network

```
Input (22-dim observation)
    │
    ▼
Dense(128, softplus)
    │
    ▼
Dense(128, softplus)
    │
    ▼
Dense(num_actions, softmax)
    │
    ▼
Action Selection
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch size | 1,000 |
| Memory limit | 25,000 experiences |
| Warmup steps | 100 |
| Optimizer | RMSprop (lr=0.0005) |
| Training steps | 100,000 |

## State Representation

**22 dimensions** (11 features × 2 players):
- Position (x, y)
- Velocity (x, y)
- Action state (256 possible)
- Damage percentage
- Stock count
- Shield status
- Additional character data

## Dolphin Integration

```
Dolphin Memory
    ↓ (Unix domain socket)
MemoryWatcher
    ↓
StateManager (handlers)
    ↓
State object
    ↓
OpenAI Gym Environment
```

**Controller Output**: Named pipe (FIFO) interface to Dolphin

## Reward Function

| Event | Reward |
|-------|--------|
| Damage dealt | +0.05 per % |
| Damage taken | -0.05 per % |
| KO opponent | +1.0 |

## Code Structure

```
Athena/
├── game_env.py          # OpenAI Gym environment
├── memory_watcher.py    # Unix socket memory monitoring
├── state_manager.py     # Memory → State parsing
├── state.py             # State dataclass and enums
├── pad.py               # Controller interface
├── menu_manager.py      # Menu automation
├── fox.py               # Character-specific logic
└── p3.py                # DQN training loop
```

## Key Features

1. **Gym-compatible**: Standard RL interface
2. **Frame-perfect timing**: FIFO pipe controller
3. **Menu automation**: Character/stage selection
4. **Action chaining**: Multi-frame technique sequences

## Limitations

- Small observation space (22 dims vs ExPhil's 1204)
- Limited action discretization
- No recurrent architecture for temporal modeling
- Abandoned early in development

## Relevance to ExPhil

**Lessons**:
- Clean abstraction layers (Memory → State → Environment → Agent)
- Unix socket-based memory watching works but is complex
- Simple DQN insufficient for Melee's complexity

**What ExPhil does better**:
- Rich 1204-dim embeddings
- Mamba/Attention temporal modeling
- Imitation learning bootstrap
- Modern training infrastructure

## References

- [Repository](https://github.com/Sciguymjm/athena)
- [Keras-RL](https://github.com/keras-rl/keras-rl)
