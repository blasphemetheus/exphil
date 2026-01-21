# ExPhil Scripts

This document describes all available scripts in the `scripts/` directory.

## Training Scripts

### train_from_replays.exs
**Main training script for imitation learning from Slippi replays.**

```bash
# Basic training
mix run scripts/train_from_replays.exs --replays /path/to/replays

# Temporal training with Mamba backbone
mix run scripts/train_from_replays.exs --temporal --backbone mamba --epochs 10

# Using presets
mix run scripts/train_from_replays.exs --preset quick      # Fast iteration
mix run scripts/train_from_replays.exs --preset standard   # Default settings
mix run scripts/train_from_replays.exs --preset production # Maximum quality

# Character/stage filtering
mix run scripts/train_from_replays.exs --character mewtwo,fox --stage fd,battlefield
```

Key options:
- `--replays PATH` - Directory containing .slp files
- `--epochs N` - Number of training epochs
- `--temporal` - Enable temporal (sequence) model
- `--backbone TYPE` - mlp, lstm, gru, mamba, sliding_window
- `--preset NAME` - quick, standard, production, mewtwo, ganondorf
- `--checkpoint PATH` - Resume from checkpoint
- `--dry-run` - Validate configuration without training

### train_ppo.exs
**PPO reinforcement learning training.**

```bash
mix run scripts/train_ppo.exs --policy checkpoints/policy.bin
```

### train_self_play.exs
**Self-play training with population-based training.**

```bash
mix run scripts/train_self_play.exs --game-type mock --timesteps 10000
mix run scripts/train_self_play.exs --game-type dolphin --num-games 4
```

### train_distillation.exs
**Knowledge distillation from teacher to student model.**

```bash
mix run scripts/train_distillation.exs \
  --soft-labels soft_labels.bin \
  --hidden 64,64 \
  --epochs 10
```

### train_kmeans.exs
**Train K-means clustering for action discretization.**

```bash
mix run scripts/train_kmeans.exs --replays /path/to/replays --k 32
```

### find_lr.exs
**Learning rate finder using Leslie Smith's LR range test.**

```bash
mix run scripts/find_lr.exs --replays /path/to/replays --min-lr 1e-7 --max-lr 1.0
```

## Evaluation & Benchmarking

### eval_model.exs
**Evaluate a trained model's performance.**

```bash
mix run scripts/eval_model.exs --checkpoint checkpoints/model.axon
mix run scripts/eval_model.exs --policy checkpoints/policy.bin --replays /path/to/replays
```

### benchmark_architectures.exs
**Benchmark inference speed of different architectures.**

```bash
mix run scripts/benchmark_architectures.exs
```

Compares: MLP, LSTM, GRU, Mamba, Sliding Window, Attention

## Dolphin Integration

### play_dolphin.exs
**Play against a trained agent in Dolphin (synchronous).**

```bash
mix run scripts/play_dolphin.exs \
  --policy checkpoints/policy.bin \
  --dolphin ~/.config/Slippi\ Launcher/netplay \
  --iso ~/melee.iso \
  --character mewtwo \
  --stage final_destination
```

Key options:
- `--policy PATH` - Path to exported policy file (required)
- `--dolphin PATH` - Path to Slippi/Dolphin folder (required)
- `--iso PATH` - Path to Melee 1.02 ISO (required)
- `--port N` - Agent controller port (default: 1)
- `--opponent-port N` - Your controller port (default: 2)
- `--frame-delay N` - Simulated online delay
- `--deterministic` - Disable action sampling
- `--action-repeat N` - Only compute new action every N frames

### play_dolphin_async.exs
**Play against a trained agent in Dolphin (asynchronous).**

Recommended for slower models (LSTM). Separates frame reading from inference.

```bash
mix run scripts/play_dolphin_async.exs \
  --policy checkpoints/policy.bin \
  --dolphin ~/.config/Slippi\ Launcher/netplay \
  --iso ~/melee.iso \
  --on-game-end restart
```

Additional options:
- `--on-game-end MODE` - restart (auto-start next game) or stop (exit after one game)

### example_bot.exs
**Example rule-based bot demonstrating the bridge API.**

```bash
mix run scripts/example_bot.exs \
  --dolphin /path/to/slippi \
  --iso /path/to/melee.iso \
  --character mewtwo
```

## Model Export

### export_onnx.exs
**Export trained model to ONNX format.**

```bash
mix run scripts/export_onnx.exs --policy checkpoints/policy.bin --output model.onnx
```

Note: Requires axon_onnx compatibility. If unavailable, use export_numpy.exs.

### export_numpy.exs
**Export model weights to NumPy format for Python/ONNX conversion.**

```bash
mix run scripts/export_numpy.exs --policy checkpoints/policy.bin --output exports/
```

Then in Python:
```bash
python priv/python/build_onnx_from_numpy.py exports/
```

### test_onnx_pipeline.exs
**Test the full ONNX export + INT8 quantization pipeline.**

```bash
mix run scripts/test_onnx_pipeline.exs
```

Tests: LSTM model creation, ONNX export, INT8 quantization, benchmark comparison.

## Data Processing

### scan_replays.exs
**Scan replay directory and report character statistics.**

```bash
mix run scripts/scan_replays.exs --replays /path/to/replays
mix run scripts/scan_replays.exs --replays /path/to/replays --max-files 1000
```

Shows:
- Total replay count
- Games per character
- Low-tier character summary
- Training recommendations

### generate_soft_labels.exs
**Generate soft labels from a teacher model for knowledge distillation.**

```bash
mix run scripts/generate_soft_labels.exs \
  --teacher checkpoints/mamba_policy.bin \
  --replays /path/to/replays \
  --output soft_labels.bin \
  --temperature 2.0
```

Temperature parameter:
- T=1.0: Original (sharp) distributions
- T=2.0: Softer distributions (recommended)
- T>3.0: Very soft, may lose information

## Model Management

### registry.exs
**Model registry management CLI.**

```bash
# List all models
mix run scripts/registry.exs list

# Show model details
mix run scripts/registry.exs show MODEL_ID

# Find best model
mix run scripts/registry.exs best

# Tag models
mix run scripts/registry.exs tag MODEL_ID production validated
mix run scripts/registry.exs untag MODEL_ID experimental

# Delete model (keeps files)
mix run scripts/registry.exs delete MODEL_ID

# Delete model and files
mix run scripts/registry.exs delete MODEL_ID --files

# Show model lineage
mix run scripts/registry.exs lineage MODEL_ID
```

Filter options:
- `--tags TAG,TAG` - Filter by tags
- `--backbone TYPE` - Filter by backbone
- `--limit N` - Limit results
- `--json` - Output as JSON

## Quick Reference

| Script | Purpose | Key Command |
|--------|---------|-------------|
| train_from_replays.exs | Imitation learning | `--preset standard --temporal` |
| train_ppo.exs | RL training | `--policy policy.bin` |
| train_self_play.exs | Self-play | `--game-type mock` |
| eval_model.exs | Evaluate model | `--checkpoint model.axon` |
| play_dolphin.exs | Live play (sync) | `--policy --dolphin --iso` |
| play_dolphin_async.exs | Live play (async) | Same as above |
| export_onnx.exs | ONNX export | `--policy policy.bin` |
| scan_replays.exs | Replay stats | `--replays /path` |
| registry.exs | Model management | `list`, `show`, `best` |

## Environment Variables

Some scripts accept environment variables:
- `SLIPPI_PATH` - Default Dolphin/Slippi path
- `MELEE_ISO` - Default Melee ISO path
- `XLA_FLAGS` - EXLA/XLA configuration
