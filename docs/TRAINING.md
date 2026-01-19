# Training Guide

Complete reference for training ExPhil models.

## Quick Start

```bash
# Single-frame imitation learning (baseline)
mix run scripts/train_from_replays.exs --epochs 10 --max-files 100

# Temporal training with Mamba (recommended)
mix run scripts/train_from_replays.exs --temporal --backbone mamba --epochs 5

# Using presets
mix run scripts/train_from_replays.exs --preset quick     # Fast iteration
mix run scripts/train_from_replays.exs --preset standard  # Balanced
mix run scripts/train_from_replays.exs --preset full      # Maximum quality
```

## Training Modes

### Single-Frame (Default)

Predicts actions from individual game states. Fast to train, good baseline.

```bash
mix run scripts/train_from_replays.exs --epochs 10 --max-files 100
```

### Temporal Training

Uses sequences of frames to learn temporal patterns (combos, reactions).

```bash
# With Mamba backbone (fastest inference, recommended)
mix run scripts/train_from_replays.exs --temporal --backbone mamba

# With LSTM
mix run scripts/train_from_replays.exs --temporal --backbone lstm

# With sliding window attention
mix run scripts/train_from_replays.exs --temporal --backbone sliding_window

# With hybrid LSTM + attention
mix run scripts/train_from_replays.exs --temporal --backbone hybrid
```

**Tradeoffs:**
- Slower per-epoch (sequences larger than single frames)
- Better at learning temporal patterns
- Recommended after establishing single-frame baseline

## Command-Line Options

### Core Options

| Option | Default | Description |
|--------|---------|-------------|
| `--replays PATH` | `./replays` | Directory containing .slp files |
| `--epochs N` | 10 | Number of training epochs |
| `--batch-size N` | 64 | Batch size |
| `--max-files N` | nil | Limit number of replay files |
| `--player-port N` | 1 | Which player to learn from (1 or 2) |
| `--hidden N,N` | 512,512 | Hidden layer sizes |
| `--lr X` | 1e-4 | Learning rate |
| `--dropout X` | 0.1 | Dropout rate |

### Temporal Options

| Option | Default | Description |
|--------|---------|-------------|
| `--temporal` | false | Enable temporal training |
| `--backbone TYPE` | lstm | lstm, gru, mamba, sliding_window, hybrid |
| `--window-size N` | 60 | Frames per sequence |
| `--stride N` | 1 | Step between sequences |
| `--truncate-bptt N` | nil | Truncated backprop (faster training) |

### Mamba-Specific Options

| Option | Default | Description |
|--------|---------|-------------|
| `--state-size N` | 16 | SSM state dimension |
| `--expand-factor N` | 2 | Expansion factor |
| `--conv-size N` | 4 | Convolution kernel size |
| `--num-layers N` | 2 | Number of Mamba layers |

### Training Features

| Option | Default | Description |
|--------|---------|-------------|
| `--val-split X` | 0.0 | Validation split (0.1 = 10%) |
| `--accumulation-steps N` | 1 | Gradient accumulation steps |
| `--lr-schedule TYPE` | constant | cosine, linear, exponential |
| `--warmup-steps N` | 0 | Learning rate warmup steps |
| `--decay-steps N` | 10000 | Steps for LR decay |
| `--early-stopping` | false | Enable early stopping |
| `--patience N` | 5 | Epochs without improvement before stopping |
| `--min-delta X` | 0.01 | Minimum improvement to count as progress |
| `--save-best` | false | Save model when val_loss improves |
| `--resume PATH` | nil | Resume from checkpoint |
| `--precision TYPE` | bf16 | bf16 or f32 |
| `--frame-delay N` | 0 | Simulated online delay (for Slippi) |

### Monitoring

| Option | Default | Description |
|--------|---------|-------------|
| `--wandb` | false | Enable Weights & Biases logging |
| `--wandb-project NAME` | exphil | W&B project name |

## Presets

```bash
--preset quick      # 1 epoch, 5 files, small MLP - fast iteration
--preset standard   # 10 epochs, 50 files - balanced
--preset full       # 50 epochs, Mamba, temporal - maximum quality
--preset mewtwo     # Character-optimized for Mewtwo
```

## Examples

### Full Training Run

```bash
mix run scripts/train_from_replays.exs \
  --replays /path/to/replays \
  --temporal --backbone mamba \
  --hidden 256 --window-size 60 \
  --epochs 20 --batch-size 64 \
  --lr 1e-4 --lr-schedule cosine --warmup-steps 1000 \
  --val-split 0.1 --early-stopping --patience 5 \
  --save-best --wandb --wandb-project exphil
```

### Memory-Constrained Training

```bash
mix run scripts/train_from_replays.exs \
  --temporal --backbone mamba \
  --hidden 128 --window-size 30 \
  --num-layers 2 --batch-size 32 \
  --max-files 5 --epochs 3
```

### Faster Training with Truncated BPTT

Limits how far gradients flow back through time:

```bash
mix run scripts/train_from_replays.exs \
  --temporal --backbone lstm \
  --window-size 60 --truncate-bptt 20
```

| Setting | Speed | Accuracy | Use Case |
|---------|-------|----------|----------|
| Full BPTT | 1x | Best | Final training |
| `--truncate-bptt 30` | ~1.5x | Good | Balanced |
| `--truncate-bptt 20` | ~2x | Moderate | Prototyping |
| `--truncate-bptt 10` | ~3x | Lower | Quick experiments |

### Training with Frame Delay (for online play)

```bash
# Simulate 18-frame Slippi online delay
mix run scripts/train_from_replays.exs \
  --temporal --backbone mamba \
  --frame-delay 18 --epochs 5
```

| Connection | Delay | Notes |
|------------|-------|-------|
| LAN/Local | 0-4 | Near-instant |
| Good online | 12-18 | Standard Slippi |
| Poor online | 24-36 | High latency |

### Gradient Accumulation

Simulate larger batch sizes on limited memory:

```bash
# Effective batch size = 32 * 4 = 128
mix run scripts/train_from_replays.exs \
  --batch-size 32 --accumulation-steps 4 --epochs 10
```

## Performance Tips

### CPU Training Optimization

**XLA Multi-threading** (auto-enabled):
```bash
XLA_FLAGS="--xla_cpu_multi_thread_eigen=true" mix run scripts/train_from_replays.exs
```

**Batch size tuning:**
- Larger batches (128, 256) reduce per-batch overhead
- Monitor RAM: `free -h` during training
- If swap increases, reduce batch size

**System tuning:**
```bash
# Lower swappiness if swap thrashing occurs
sudo sysctl vm.swappiness=10

# Increase process priority
sudo renice -n -5 -p <PID>
```

### Training Time Expectations

| Dataset Size | Epochs | Estimated Time (CPU) |
|--------------|--------|---------------------|
| 1 file (~14K frames) | 1 | ~5-6 minutes |
| 10 files (~140K frames) | 1 | ~15-20 minutes |
| 100 files (~1.4M frames) | 10 | ~3-4 hours |

*First epoch includes ~5 min JIT compilation overhead*

### Memory Usage by Config

| Config | Embedding | Training | Notes |
|--------|-----------|----------|-------|
| Full (hidden=256, window=60, files=10) | ~10GB | ~12GB | Best accuracy |
| Medium (hidden=128, window=30, files=5) | ~5GB | ~9GB | Good balance |
| Minimal (hidden=64, window=20, files=3) | ~3GB | ~5GB | Fast iteration |

## PPO Fine-tuning

After imitation learning, refine with reinforcement learning:

```bash
# Test PPO loop with mock environment
mix run scripts/train_ppo.exs --mock \
  --pretrained checkpoints/imitation_latest_policy.bin \
  --timesteps 10000

# Full PPO training with Dolphin
mix run scripts/train_ppo.exs \
  --pretrained checkpoints/imitation_latest_policy.bin \
  --dolphin /path/to/slippi \
  --iso /path/to/melee.iso \
  --character mewtwo \
  --opponent cpu3 \
  --timesteps 100000
```

## Evaluation

```bash
# Evaluate model on replay frames
mix run scripts/eval_model.exs --policy checkpoints/imitation_latest_policy.bin

# Interactive analysis with Livebook
livebook server notebooks/evaluation_dashboard.livemd --port 8080
```

The evaluation dashboard provides:
1. **Load Policy** - View config and architecture
2. **Test Inference** - Run on sample states
3. **Compare to Replays** - Accuracy vs human play
4. **Visualize Actions** - Charts of button/stick distributions

## Checkpointing

Training produces:
```
checkpoints/
├── {backbone}_{name}_{timestamp}.axon          # Full checkpoint
├── {backbone}_{name}_{timestamp}_policy.bin    # Exported policy
└── {backbone}_{name}_{timestamp}_config.json   # Training config
```

### Resume Training

```bash
mix run scripts/train_from_replays.exs --resume checkpoints/model.axon
```

Restores: model weights, optimizer state, step counter, config.

### Best Model Checkpointing

```bash
mix run scripts/train_from_replays.exs --save-best --val-split 0.1
```

Saves whenever validation loss improves.

## Tests

```bash
mix test                                      # All tests
mix test --cover                              # With coverage
mix test test/exphil/training/imitation_test.exs  # Specific file
mix test --include slow                       # Include slow tests
```
