# Self-Play Training Guide

Practical guide for running self-play reinforcement learning to improve policies beyond imitation learning.

## Prerequisites

1. **Pretrained policy** (recommended) - Train with imitation learning first:
   ```bash
   mix run scripts/train_from_replays.exs --replays /path/to/replays --epochs 10
   ```

2. **For Dolphin mode**: Slippi Dolphin + Melee ISO (see [SELF_PLAY_DOLPHIN.md](../operations/SELF_PLAY_DOLPHIN.md))

3. **For mock mode**: No additional setup needed

## Quick Start

### Mock Mode (Fast Testing)

Mock mode uses a physics simulation instead of Dolphin - ideal for testing and development:

```bash
# Quick test (1000 steps, ~1 min)
mix run scripts/train_self_play.exs \
  --game-type mock \
  --timesteps 1000 \
  --num-games 4

# With pretrained policy (recommended)
mix run scripts/train_self_play.exs \
  --game-type mock \
  --pretrained checkpoints/imitation_policy.bin \
  --timesteps 50000 \
  --num-games 8 \
  --track-elo

# Full training run
mix run scripts/train_self_play.exs \
  --game-type mock \
  --pretrained checkpoints/imitation_policy.bin \
  --timesteps 500000 \
  --num-games 16 \
  --track-elo \
  --save-interval 50
```

### Dolphin Mode (Real Games)

```bash
# Activate Python environment first
source .venv/bin/activate

# Run with Dolphin
mix run scripts/train_self_play.exs \
  --game-type dolphin \
  --pretrained checkpoints/imitation_policy.bin \
  --dolphin ~/.config/Slippi\ Launcher/netplay \
  --iso ~/melee.iso \
  --character mewtwo \
  --stage final_destination \
  --timesteps 100000 \
  --num-games 1
```

## Command Line Options

### Core Options

| Option | Default | Description |
|--------|---------|-------------|
| `--pretrained PATH` | none | Load pretrained policy (strongly recommended) |
| `--game-type TYPE` | mock | `mock` or `dolphin` |
| `--timesteps N` | 100000 | Total training steps |
| `--num-games N` | 4 | Parallel game instances |
| `--checkpoint PATH` | checkpoints/self_play.axon | Save location |
| `--save-interval N` | 10 | Save every N iterations |

### PPO Hyperparameters

| Option | Default | Description |
|--------|---------|-------------|
| `--rollout-length N` | 128 | Steps per rollout collection |
| `--batch-size N` | 2048 | Minibatch size for PPO updates |
| `--ppo-epochs N` | 4 | PPO epochs per update |
| `--ppo-clip F` | 0.2 | PPO clip epsilon |
| `--learning-rate F` | 3e-4 | Learning rate |
| `--gamma F` | 0.99 | Discount factor |
| `--gae-lambda F` | 0.95 | GAE lambda for advantage estimation |

### Dolphin Options

| Option | Default | Description |
|--------|---------|-------------|
| `--dolphin PATH` | - | Path to Slippi Dolphin folder |
| `--iso PATH` | - | Path to Melee 1.02 ISO |
| `--character NAME` | mewtwo | Character to play |
| `--stage NAME` | final_destination | Stage |

### Tracking Options

| Option | Default | Description |
|--------|---------|-------------|
| `--track-elo` | false | Enable Elo rating tracking |
| `--max-episode-frames N` | 3600 | Max frames per episode (60s default) |

## Training Workflow

### Step 1: Start with Imitation Learning

Self-play works best when starting from a policy that already knows basic moves:

```bash
# Train imitation policy first
mix run scripts/train_from_replays.exs \
  --replays /workspace/replays/mewtwo \
  --epochs 10 \
  --backbone mamba \
  --temporal

# Export the policy
# (training automatically saves to checkpoints/)
```

### Step 2: Run Self-Play Training

```bash
mix run scripts/train_self_play.exs \
  --pretrained checkpoints/imitation_latest_policy.bin \
  --game-type mock \
  --timesteps 100000 \
  --num-games 8 \
  --track-elo
```

### Step 3: Monitor Progress

The script outputs:
- **Iteration count** and total steps
- **Experiences collected** per iteration
- **PPO loss** (policy loss, value loss, entropy)
- **Elo leaderboard** (if `--track-elo` enabled)

Example output:
```
═══ Self-Play Training ═══
Configuration:
    Mode: simple_mix
    Game type: mock
    Num games: 8
    ...

[03:05:02] Iter 1: Collected 1024 experiences (1024 total steps)
[03:05:04]   PPO update: loss=0.1234, policy=0.0456, value=0.0789, entropy=0.0123
[03:05:06] Iter 2: Collected 1024 experiences (2048 total steps)
...

═══ Elo Leaderboard ═══
| Rank | Policy | Elo | W | L | D |
|------|--------|-----|---|---|---|
| 1 | gen_5 | 1520 | 12 | 3 | 5 |
| 2 | gen_3 | 1495 | 8 | 7 | 5 |
```

## Important Notes

### JIT Compilation Warmup

The first iteration takes 2-5 minutes due to XLA JIT compilation. This is normal:

```
[03:03:58] ⚠️  First batch triggers JIT compilation (may take 2-5 min)
```

Subsequent iterations are fast (seconds each).

### Memory Considerations

- Mock mode: ~100 MB per game
- Dolphin mode: ~300 MB per game
- PPO batch tensors: scales with batch_size

For large runs, monitor GPU memory:
```bash
watch -n 1 nvidia-smi
```

### Checkpoint Format

Self-play checkpoints contain:
- `params` - PPO actor-critic parameters
- `model` - Axon model structure
- `iteration` - Training iteration
- `total_steps` - Total environment steps
- `config` - Training configuration

Load a checkpoint:
```elixir
checkpoint = :erlang.binary_to_term(File.read!("checkpoints/self_play.axon"))
params = checkpoint.params
```

## Troubleshooting

### GenServer Timeout During First Iteration

**Symptom**: `** (exit) exited in: GenServer.call(..., 60000)`

**Cause**: JIT compilation takes longer than the default timeout.

**Fix**: The script includes warmup - if you still see this, increase `@collect_timeout` in `game_pool_supervisor.ex`:
```elixir
@collect_timeout 300_000  # 5 minutes
```

### Elo Ratings Don't Change

**Symptom**: All policies have Elo 1500, W/L/D all zeros or only draws.

**Cause**: Random/weak policies often draw (equal stocks at timeout).

**Fix**: Use longer episodes or pretrained policies:
```bash
--max-episode-frames 7200  # 2 minutes instead of 1
--pretrained checkpoints/trained_policy.bin
```

### OOM During PPO Update

**Symptom**: Out of memory during gradient computation.

**Cause**: Batch size too large or accumulated experiences.

**Fix**: Reduce batch size:
```bash
--batch-size 1024  # or 512
```

### Mock Environment Physics Issues

**Symptom**: Characters fall through floor or behave oddly.

**Cause**: Mock physics is simplified - some edge cases aren't handled.

**Note**: Mock mode is for testing the training loop, not realistic gameplay. Use Dolphin mode for real training.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    train_self_play.exs                       │
│  - Parses CLI args                                          │
│  - Creates PPO trainer                                      │
│  - Runs training loop                                       │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    GamePoolSupervisor                        │
│  - DynamicSupervisor for GameRunner processes               │
│  - collect_all_steps() gathers experiences from all games   │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    GameRunner (N instances)                  │
│  - GenServer per game                                       │
│  - Mock physics OR Dolphin bridge                           │
│  - Runs policy inference, collects (s,a,r,s') tuples        │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    PPO Trainer                               │
│  - Computes advantages (GAE)                                │
│  - Updates policy with clipped objective                    │
│  - Returns updated parameters                               │
└─────────────────────────────────────────────────────────────┘
```

## References

- [SELF_PLAY_ARCHITECTURE.md](../reference/SELF_PLAY_ARCHITECTURE.md) - Detailed architecture design
- [SELF_PLAY_DOLPHIN.md](../operations/SELF_PLAY_DOLPHIN.md) - Dolphin setup and testing
- [PPO Paper](https://arxiv.org/abs/1707.06347) - Proximal Policy Optimization
- [AlphaStar](https://www.nature.com/articles/s41586-019-1724-z) - League training approach
