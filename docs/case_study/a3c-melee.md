# A3C Melee AI Case Study

**Repository**: https://github.com/KeganMc/Super-Smash-Bros-Melee-AI
**Authors**: Jay Bishop, Jordan Cooper, Brant Dolling, Kegan McIlwaine
**Status**: Inactive (May 2017)
**Language**: Python (core), PLSQL (website)
**License**: GPL-3.0

## Overview

A deep learning project using **Asynchronous Advantage Actor-Critic (A3C)** to train Melee bots. Includes infrastructure for a community model-sharing website.

## Architecture

### Actor-Critic Network

```
Input (65-dim state)
    │
    ├──────────────────────────┐
    ▼                          ▼
┌─────────────────┐    ┌─────────────────┐
│  Actor (Policy) │    │ Critic (Value)  │
│  65→64→64→40    │    │ 65+40→64→64→1   │
│  ELU activation │    │ ELU activation  │
│  Softmax output │    │ Linear output   │
└─────────────────┘    └─────────────────┘
```

### Training Details

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-4 |
| Optimizer | Adam |
| Gradient clipping | 40.0 |
| Exploration | 2% baseline (softmax smoothing) |
| Workers | Multi-threaded A3C |

## State Representation

**65 dimensions** including:

**Per player (×2)**:
- Position (x, y) normalized
- Velocity (vx, vy)
- Action state (normalized)
- Body state
- Damage % (0-999)
- Shield integrity
- Stocks remaining
- Facing direction
- Jump counter

**Global**:
- Stage ID
- Platform positions
- Frame counter
- Menu state

## Action Space

**40 discrete actions**:
- 8 directional movements
- 9 tilt variations
- 20 button combinations
- 1 Z-button (A+R)

## Reward Function

| Component | Value |
|-----------|-------|
| Opponent damage increase | +0.01 × Δ% |
| Bot damage increase | -0.01 × Δ% |
| Allied damage | -0.002 × Δ% |
| Opponent KO | +1.0 |
| Bot KO | -1.0 |
| Allied KO | -0.2 |

## Dolphin Integration

**Memory Interface**: Unix domain datagram socket
- Timeout: 0.001s per read
- Real-time memory changes via `state_manager.py`

**Controller Interface**: Named pipe
- Device: `Pipe/0/pipe`
- Commands: `SET MAIN x y`, `PRESS`/`RELEASE`

## Code Structure

```
Source/
├── BigProject.py       # Main orchestration
├── actor_critic.py     # A3C network
├── state.py            # Enums (26 chars, 31 stages, 337 actions)
├── state_manager.py    # Memory address mapping
├── controller_outputs.py # 40 discrete actions
├── memory_watcher.py   # Dolphin socket interface
├── reward.py           # Reward computation
├── pyqtgui.py          # PyQt5 configuration UI
└── workerThread.py     # A3C threading
```

## Key Features

1. **PyQt5 GUI**: Player configuration, team assignment, training toggle
2. **Model persistence**: Save/load trained weights
3. **Multi-agent**: Supports 4 players with team assignment
4. **Community website**: Model sharing infrastructure (incomplete)

## Comparison to ExPhil

| Aspect | A3C Melee | ExPhil |
|--------|-----------|--------|
| State dims | 65 | 1204 |
| Network | 2-layer MLP | Mamba/Attention |
| Training | Self-play only | BC + RL |
| Framework | TensorFlow 1.x | Nx/Axon |
| Actions | 40 discrete | Autoregressive sampling |

## Lessons for ExPhil

**Applicable**:
- Multi-threaded training architecture
- Reward shaping for damage/KOs
- State normalization by stage dimensions

**What to avoid**:
- Small state representation limits learning
- Legacy TensorFlow requires maintenance
- No imitation learning bootstrap

## References

- [Repository](https://github.com/KeganMc/Super-Smash-Bros-Melee-AI)
- [A3C Paper](https://arxiv.org/abs/1602.01783)
