# Training Guide

Complete reference for training ExPhil models.

## Quick Start

**New to ExPhil?** Use the interactive wizard:
```bash
mix exphil.setup
```

Or use presets directly:
```bash
# Using presets (recommended)
mix run scripts/train_from_replays.exs --preset quick     # Fast iteration (~5 min)
mix run scripts/train_from_replays.exs --preset standard  # Balanced (~30 min)
mix run scripts/train_from_replays.exs --preset full      # Maximum quality (~2 hrs)
mix run scripts/train_from_replays.exs --preset mewtwo    # Character-specific

# Manual configuration
mix run scripts/train_from_replays.exs --epochs 10 --max-files 100
mix run scripts/train_from_replays.exs --temporal --backbone mamba --epochs 5
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

### Character & Port Selection

When training from replay archives, the target character may be on different ports across games.

**Option 1: Fixed port (default)**
```bash
# Always learn from player 1
mix run scripts/train_from_replays.exs --player-port 1
```

**Option 2: Auto-select by character (recommended for single-character training)**
```bash
# Learn from whichever port has Mewtwo
mix run scripts/train_from_replays.exs --train-character mewtwo
```

**Option 3: Dual-port (2x data)**
```bash
# Learn from BOTH players in every game
mix run scripts/train_from_replays.exs --dual-port
```

| Mode | Use Case | Data Volume |
|------|----------|-------------|
| `--player-port N` | Know which port to train on | 1x |
| `--train-character X` | Train specific character, unknown ports | 1x (filtered) |
| `--dual-port` | Maximum data, mixed characters | 2x |

**Note:** `--dual-port` trains on all characters in the replays, not just one. Use `--train-character` for pure single-character training.

## Command-Line Options

### Core Options

| Option | Default | Description |
|--------|---------|-------------|
| `--replays PATH` | `./replays` | Directory containing .slp files |
| `--epochs N` | 10 | Number of training epochs |
| `--batch-size N` | 64 | Batch size |
| `--max-files N` | nil | Limit number of replay files |
| `--player-port N` | 1 | Which player to learn from (1 or 2) |
| `--train-character CHAR` | nil | Auto-select port with this character |
| `--dual-port` | false | Train on BOTH players (2x data) |
| `--hidden N,N` | 512,512 | Hidden layer sizes |
| `--lr X` | 1e-4 | Learning rate |
| `--dropout X` | 0.1 | Dropout rate |
| `--name NAME` | nil | Custom checkpoint name |
| `--preset NAME` | nil | Training preset (quick, standard, full, mewtwo) |
| `--config PATH` | nil | YAML config file path |
| `--dry-run` | false | Validate config without training |

### Error Handling

| Option | Default | Description |
|--------|---------|-------------|
| `--skip-errors` | true | Continue past bad replay files |
| `--fail-fast` | false | Stop on first error |
| `--show-errors` | true | Show individual file errors |
| `--hide-errors` | false | Hide individual file errors |
| `--error-log PATH` | nil | Log errors to file |

### Data Filtering

| Option | Default | Description |
|--------|---------|-------------|
| `--characters CHAR,...` | [] | Filter replays by character |
| `--stages STAGE,...` | [] | Filter replays by stage |
| `--balance-characters` | false | Weight sampling by inverse char frequency |
| `--skip-duplicates` | true | Skip duplicate replay files by hash |
| `--no-skip-duplicates` | false | Include all files even if duplicates |
| `--min-quality N` | nil | Minimum quality score (0-100) for replays |
| `--show-quality-stats` | false | Show quality distribution after filtering |

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

### Attention/Hybrid Options

| Option | Default | Description |
|--------|---------|-------------|
| `--num-heads N` | 4 | Number of attention heads |
| `--attention-every N` | 2 | Add attention every N layers (hybrid) |

### Model Architecture

| Option | Default | Description |
|--------|---------|-------------|
| `--optimizer TYPE` | adam | adam, adamw, lamb, radam |
| `--layer-norm` | false | Enable layer normalization (MLP) |
| `--no-layer-norm` | - | Disable layer normalization |
| `--residual` | false | Enable residual connections (MLP) |
| `--no-residual` | - | Disable residual connections |

### Training Features

| Option | Default | Description |
|--------|---------|-------------|
| `--val-split X` | 0.0 | Validation split (0.1 = 10%) |
| `--accumulation-steps N` | 1 | Gradient accumulation steps |
| `--lr-schedule TYPE` | constant | cosine, linear, exponential |
| `--warmup-steps N` | 1 | Learning rate warmup steps |
| `--decay-steps N` | nil | Steps for LR decay |
| `--restart-period N` | 1000 | Cosine annealing restart period (T_0) |
| `--restart-mult N` | 2 | Restart period multiplier (T_mult) |
| `--max-grad-norm X` | 1.0 | Gradient clipping norm (0 = disabled) |
| `--early-stopping` | false | Enable early stopping |
| `--patience N` | 5 | Epochs without improvement before stopping |
| `--min-delta X` | 0.01 | Minimum improvement to count as progress |
| `--save-best` | true | Save model when val_loss improves |
| `--save-every N` | nil | Save checkpoint every N epochs |
| `--resume PATH` | nil | Resume from checkpoint |
| `--precision TYPE` | bf16 | bf16 or f32 |
| `--frame-delay N` | 0 | Simulated online delay (for Slippi) |
| `--stream-chunk-size N` | nil | Load N files at a time (memory-bounded) |

### Data Augmentation

| Option | Default | Description |
|--------|---------|-------------|
| `--augment` | false | Enable data augmentation |
| `--mirror-prob X` | 0.5 | Mirror augmentation probability |
| `--noise-prob X` | 0.3 | Noise augmentation probability |
| `--noise-scale X` | 0.01 | Noise magnitude |
| `--label-smoothing X` | 0.0 | Label smoothing (0.1 = typical) |
| `--focal-loss` | false | Enable focal loss for rare actions |
| `--focal-gamma X` | 2.0 | Focal loss gamma (higher = focus on hard) |

### Online Play Training

| Option | Default | Description |
|--------|---------|-------------|
| `--online-robust` | false | Enable online play training mode |
| `--frame-delay-augment` | false | Enable frame delay augmentation |
| `--frame-delay-min N` | 0 | Minimum delay frames (local play) |
| `--frame-delay-max N` | 18 | Maximum delay frames (online play) |

### Monitoring

| Option | Default | Description |
|--------|---------|-------------|
| `--wandb` | false | Enable Weights & Biases logging |
| `--wandb-project NAME` | exphil | W&B project name |
| `--wandb-name NAME` | nil | W&B run name (auto-generated if nil) |

### Verbosity & Reproducibility

| Option | Default | Description |
|--------|---------|-------------|
| `--quiet` | false | Minimal output (errors only) |
| `--verbose` | false | Debug output (timing, memory) |
| `--seed N` | random | Random seed for reproducibility |

### Checkpoint Safety

| Option | Default | Description |
|--------|---------|-------------|
| `--overwrite` | false | Allow overwriting existing checkpoints |
| `--no-overwrite` | false | Fail if checkpoint exists |
| `--backup` | true | Create .bak before overwrite |
| `--no-backup` | false | Skip backup creation |
| `--backup-count N` | 3 | Number of backup versions to keep |

### Performance Options

| Option | Default | Description |
|--------|---------|-------------|
| `--precompute` | true | Precompute embeddings (2-3x speedup) |
| `--no-precompute` | - | Disable embedding precomputation |
| `--prefetch` | true | Prefetch batches while GPU trains |
| `--no-prefetch` | - | Disable batch prefetching |
| `--prefetch-buffer N` | 2 | Number of batches to prefetch |
| `--gradient-checkpoint` | false | Trade memory for compute |
| `--checkpoint-every N` | 1 | Checkpoint every N layers |

### Advanced Options

| Option | Default | Description |
|--------|---------|-------------|
| `--ema` | false | Enable model EMA |
| `--ema-decay X` | 0.999 | EMA decay rate |
| `--no-register` | false | Skip model registry |
| `--keep-best N` | nil | Keep best N checkpoints (prune others) |
| `--kmeans-centers PATH` | nil | K-means cluster centers for sticks |

### Embedding Options

| Option | Default | Description |
|--------|---------|-------------|
| `--stage-mode MODE` | full | Stage embedding: full, compact, learned |
| `--num-player-names N` | 112 | Player name dims (0 to disable) |

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

Reduce model/batch size for limited GPU memory:

```bash
mix run scripts/train_from_replays.exs \
  --temporal --backbone mamba \
  --hidden 128 --window-size 30 \
  --num-layers 2 --batch-size 32 \
  --max-files 5 --epochs 3
```

Or use streaming for large datasets that don't fit in RAM (see [Streaming Data Loading](#streaming-data-loading)):

```bash
mix run scripts/train_from_replays.exs \
  --stream-chunk-size 30  # Process 30 files at a time
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

### Streaming Data Loading

Train on large datasets without running out of memory by loading files in chunks:

```bash
# Process 30 files at a time (good for 56GB RAM)
mix run scripts/train_from_replays.exs \
  --temporal --backbone jamba \
  --train-character mewtwo \
  --stream-chunk-size 30 \
  --epochs 20
```

**How it works:**
1. Files are split into chunks of N files each
2. For each epoch, iterate through all chunks
3. Each chunk: parse → embed → train → free memory → next chunk

| Chunk Size | RAM Usage | Speed | Use Case |
|------------|-----------|-------|----------|
| `30` | ~20GB | 1x | Standard GPU pods (56GB RAM) |
| `50` | ~35GB | ~1.1x | High-memory machines |
| `100` | ~70GB | ~1.15x | Large RAM servers |
| nil (default) | All data | Fastest | When data fits in RAM |

**Trade-offs:**
- ~10-20% slower due to repeated file I/O each epoch
- Validation uses training loss as proxy (no separate val set in streaming mode)
- Memory bounded by chunk size, not total dataset size

**Prefetching works with streaming mode** (fixed in commit 72e4c8f):
```bash
mix run scripts/train_from_replays.exs \
  --temporal --backbone jamba \
  --stream-chunk-size 30
  # Prefetching enabled by default, works correctly
```

The prefetcher eagerly buffers batches from lazy chunk streams.

**Automatic optimizations in streaming mode:**
- Precompute is auto-disabled (embeddings computed on-the-fly)
- This is intentional: precomputing per-chunk then discarding is wasteful
- On-the-fly embedding is faster for streaming since chunks aren't reused

**Recommended chunk sizes by RAM:**
- 32GB RAM: `--stream-chunk-size 15`
- 56GB RAM: `--stream-chunk-size 30`
- 128GB RAM: `--stream-chunk-size 60` or no streaming

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

## Interactive Setup Wizard

New to ExPhil? Use the interactive wizard to build your training command:

```bash
mix exphil.setup
```

The wizard walks you through:

1. **Goal Selection** - Quick experiment, character training, production model, or fine-tuning
2. **Character Selection** - Choose from Mewtwo, Ganondorf, Link, G&W, Zelda, Ice Climbers, or general
3. **Hardware Configuration** - Auto-detects GPU and recommends batch size
4. **Data Configuration** - Replay directory, file limits
5. **Advanced Options** - Backbone, augmentation, W&B logging

At the end, it generates a ready-to-run command and optionally executes it.

**Example session:**
```
╔════════════════════════════════════════════════════════════════╗
║                ExPhil Training Setup Wizard                    ║
╚════════════════════════════════════════════════════════════════╝

Detected Hardware:
  GPU: NVIDIA GeForce RTX 4090 (24 GB)

What would you like to do?
  [1] Quick experiment (test setup, ~5 minutes)
  [2] Train a character-specific model
  [3] Train a general-purpose model
  [4] Fine-tune an existing model

Choice [1]: 2

Which character?
  [1] Mewtwo - Floaty, teleport recovery, tail hitboxes
  [2] Ganondorf - Heavy, powerful, spacing-focused
  ...

Choice [1]: 1

GPU Memory Tier: 24GB+
  Recommended batch size: 256
Use recommended settings? [Y/n]: y

...

════════════════════════════════════════════════════════════════
                              Command
════════════════════════════════════════════════════════════════

  mix run scripts/train_from_replays.exs --train-character mewtwo --epochs 20 --batch-size 256 --temporal --backbone mamba --augment

Run this command now? [y/N]:
```

## Environment Variables

Configure defaults via environment variables (CLI args still override):

| Variable | Default | Description |
|----------|---------|-------------|
| `EXPHIL_REPLAYS_DIR` | `./replays` | Default replay directory |
| `EXPHIL_WANDB_PROJECT` | `exphil` | Default W&B project name |
| `EXPHIL_DEFAULT_PRESET` | none | Default preset to use |

**Usage:**
```bash
# Set in shell profile (~/.bashrc or ~/.zshrc)
export EXPHIL_REPLAYS_DIR="/data/melee/replays"
export EXPHIL_WANDB_PROJECT="my-melee-ai"

# Now training uses these defaults
mix run scripts/train_from_replays.exs --epochs 10

# CLI args still override
mix run scripts/train_from_replays.exs --replays /other/path
```

## Verbosity Control

Control output verbosity with `--quiet` or `--verbose`:

| Flag | Level | Output |
|------|-------|--------|
| `--quiet` | 0 | Errors and warnings only, no progress bars |
| (default) | 1 | Normal output with progress bars |
| `--verbose` | 2 | Debug info: timing, memory, gradients |

**Examples:**
```bash
# Quiet mode for CI/scripted runs
mix run scripts/train_from_replays.exs --quiet --preset quick

# Verbose mode for debugging
mix run scripts/train_from_replays.exs --verbose --preset quick
```

**Verbose output includes:**
- Per-batch timing breakdown
- GPU memory usage after each epoch
- Gradient norm statistics
- Data loading vs training time
- Cache hit rates
- Debug messages marked with `[DEBUG]`

## Reproducibility

Training runs can be exactly reproduced using random seeds.

### Seed Display

Every training run shows its seed in the startup banner:
```
  Model Name:  mamba_mewtwo_20260123_143052
  Seed:        1234567890 (use --seed 1234567890 to reproduce)
```

### Explicit Seed

```bash
# Reproduce a previous run exactly
mix run scripts/train_from_replays.exs --seed 1234567890 --preset quick
```

### What the seed controls

- Parameter initialization (Nx/EXLA random operations)
- Data shuffling order
- Augmentation random choices (mirror, noise)
- Train/validation split randomness

**Note:** For exact reproduction, you also need the same:
- Replay files (same files in same order)
- Hardware (GPU vs CPU may differ slightly)
- ExPhil version

## Checkpoint Safety

Protect valuable checkpoints from accidental overwrites.

### Collision Warnings

If a checkpoint already exists, you'll see a warning:
```
⚠️  Checkpoint 'checkpoints/mewtwo_v1.axon' already exists
       Size: 45.2 MB, Modified: 2026-01-23 14:30:00
       Use --overwrite to replace, or choose a different --name
```

### Overwrite Control

| Flag | Behavior |
|------|----------|
| (default) | Error if checkpoint exists |
| `--overwrite` | Allow overwriting (with backup) |
| `--no-overwrite` | Explicitly fail if exists (for CI) |

### Automatic Backups

When overwriting, the existing checkpoint is automatically backed up:

```bash
# This creates a backup before overwriting
mix run scripts/train_from_replays.exs --overwrite --name mewtwo_v1
```

Backup files:
```
checkpoints/
├── mewtwo_v1.axon        # Current (new)
├── mewtwo_v1.axon.bak    # Previous version
├── mewtwo_v1.axon.bak.1  # Two versions ago
└── mewtwo_v1.axon.bak.2  # Three versions ago
```

### Backup Options

| Flag | Default | Description |
|------|---------|-------------|
| `--backup` | true | Create .bak before overwrite |
| `--no-backup` | - | Skip backup (faster) |
| `--backup-count N` | 3 | Number of backup versions to keep |

**Example:**
```bash
# Keep 5 backup versions
mix run scripts/train_from_replays.exs --overwrite --backup-count 5

# Skip backups (for ephemeral training)
mix run scripts/train_from_replays.exs --overwrite --no-backup
```

## Embedding Options

### Stage Embedding Mode

Control how stages are embedded:

| Mode | Dims | Description |
|------|------|-------------|
| `--stage-mode full` | 64 | One-hot for all 64 stages (default) |
| `--stage-mode compact` | 7 | One-hot for 6 competitive + "other" |
| `--stage-mode learned` | 1 | Stage ID with trainable embedding |

```bash
# Save 57 dimensions with compact mode
mix run scripts/train_from_replays.exs --stage-mode compact

# Learned embedding (most compact)
mix run scripts/train_from_replays.exs --stage-mode learned
```

### Player Name Embedding

Control player name embedding dimensions:

| Option | Dims | Description |
|--------|------|-------------|
| (default) | 112 | slippi-ai compatible |
| `--num-player-names 0` | 0 | Disable (saves 112 dims) |
| `--num-player-names N` | N | Custom size |

```bash
# Disable player names to save dimensions
mix run scripts/train_from_replays.exs --num-player-names 0
```

**Note:** Existing models trained with 112 dims require `--num-player-names 112` for inference compatibility.
