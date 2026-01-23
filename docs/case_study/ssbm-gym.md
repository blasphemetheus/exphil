# ssbm_gym Case Study

**Repository**: https://github.com/Gurvan/ssbm_gym
**Author**: Gurvan Priem
**Status**: Active (MIT License)
**Language**: Python
**Stars**: 27

## Overview

ssbm_gym provides an **OpenAI Gym-compatible environment** for training RL agents on Super Smash Bros. Melee. Built on vladfi1's foundational work, it enables both single and parallel environment training.

## Installation

```bash
pip install -e .
```

**Requirements**:
- Python 3
- SSBM NTSC v1.02 ISO
- Platform-specific Dolphin build

## Gym API

### Basic Usage

```python
env = SSBMEnv(
    rendering=True,
    player1="ai",
    player1_char="Fox",
    player2="cpu",
    player2_char="Falco",
    player2_level=7,
    stage="Battlefield"
)

obs = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
env.close()
```

### Vectorized Training

```python
env = EnvVec(
    num_workers=4,
    env_config={...}
)
obs = env.reset()
actions = [env.action_space.sample() for _ in range(4)]
obs_batch, rewards, dones, infos = env.step(actions)
```

## Observation Space

**Continuous Box** via `EmbedGame`:
- Player positions (x, y)
- Damage percentages
- Stock counts
- Action state IDs
- Speed vectors
- Shield data
- Stage information
- Frame counter

## Action Space

**DiagonalActionSpace** discretizes controller inputs:

| Component | Discretization |
|-----------|----------------|
| Buttons | 8 independent (A, B, X, Y, Z, L, R, Start) |
| Main stick | 16-32 positions per axis |
| C-stick | Same as main stick |
| Shoulder | 4 positions (0%, 33%, 66%, 100%) |

### Pre-configured Action Sets

- Cardinal directions (8-way)
- Short hop sequences
- Wavedash combinations
- Custom technique chains

### Smart Filtering

`banned()` method prevents exploits:
- Blocks Peach neutral-B stall
- Prevents infinite laser loops

## Reward Function

```
Total = Stock Reward + Damage Reward

Stock:
  +1.0 if opponent KO'd
  -1.0 if player KO'd

Damage:
  +0.01 × opponent damage dealt
  -0.01 × damage taken
```

## Dolphin Integration

### DolphinAPI

```
┌─────────────────────────────────┐
│     Dolphin Emulator Process    │
└──────────────┬──────────────────┘
               │ ZeroMQ (pyzmq)
               ▼
┌─────────────────────────────────┐
│         DolphinAPI              │
│  - Memory watcher               │
│  - Controller interface         │
│  - State synchronization        │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│     SSBMEnv (Gym Interface)     │
└─────────────────────────────────┘
```

### Platform Support

| Platform | Training | Notes |
|----------|----------|-------|
| Linux | Full | Custom Dolphin build required |
| Windows | Limited | Playback/visualization only |

## Configuration Options

```python
SSBMEnv(
    rendering=False,          # Headless training
    player1="ai",             # ai, human, cpu
    player1_char="Fox",
    player2="cpu",
    player2_level=7,          # CPU difficulty 1-9
    stage="Battlefield",
    frame_limit=8*60,         # 8 seconds default
    dolphin_path="./dolphin-exe/",
    iso_path="./ISOs/SSBM.iso"
)
```

## RL Library Compatibility

Works with any Gym-compatible library:
- **Stable-Baselines3**: PPO, A2C, DQN
- **Ray RLlib**: Distributed training
- **Standard Gym wrappers**

```python
from stable_baselines3 import PPO

env = SSBMEnv(...)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

## Code Structure

```
ssbm_gym/
├── ssbm_gym/
│   ├── __init__.py
│   ├── env.py           # SSBMEnv implementation
│   ├── dolphin_api.py   # Dolphin communication
│   ├── embed.py         # State embedding
│   └── action_space.py  # DiagonalActionSpace
├── example/
├── test_env.py
├── test_env_vectorized.py
└── setup.py
```

## Relevance to ExPhil

**What ssbm_gym provides**:
- Battle-tested Gym interface pattern
- Dolphin communication via ZeroMQ
- Vectorized environment for parallel training
- Discrete action space design

**ExPhil differences**:
- Elixir/OTP for native concurrency
- Richer state embedding (1204 dims)
- Mamba backbone for temporal modeling
- ONNX export for deployment

## References

- [Repository](https://github.com/Gurvan/ssbm_gym)
- [Custom Dolphin Build](https://github.com/Gurvan/dolphin)
- [OpenAI Gym](https://www.gymlibrary.dev/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
