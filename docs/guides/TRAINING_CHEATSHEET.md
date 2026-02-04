# Training Cheatsheet

Quick reference for common training commands. See [TRAINING.md](TRAINING.md) for full documentation.

## Presets (Start Here)

```bash
mix run scripts/train_from_replays.exs --preset quick      # Testing (~5 min)
mix run scripts/train_from_replays.exs --preset standard   # Balanced (~30 min)
mix run scripts/train_from_replays.exs --preset full       # Maximum quality (~2 hrs)
mix run scripts/train_from_replays.exs --preset mewtwo     # Mewtwo-specific
mix run scripts/train_from_replays.exs --preset ganondorf  # Ganondorf-specific
```

## Common Patterns

### Basic Training
```bash
# Simple MLP training
mix run scripts/train_from_replays.exs --epochs 10 --max-files 100

# With validation split
mix run scripts/train_from_replays.exs --epochs 10 --val-split 0.1 --save-best
```

### Temporal Training (Recommended)
```bash
# Mamba backbone (recommended - best speed/accuracy)
mix run scripts/train_from_replays.exs --temporal --backbone mamba

# Liquid Neural Networks (adaptive dynamics)
mix run scripts/train_from_replays.exs --temporal --backbone liquid

# RWKV-7 (O(1) memory inference)
mix run scripts/train_from_replays.exs --temporal --backbone rwkv

# LSTM (best accuracy, but slow)
mix run scripts/train_from_replays.exs --temporal --backbone lstm

# With K-means stick discretization (better precision)
mix run scripts/train_kmeans.exs --replays ./replays --k 21 --output priv/kmeans_centers.nx
mix run scripts/train_from_replays.exs --temporal --backbone mamba --kmeans-centers priv/kmeans_centers.nx
```

### Character-Specific
```bash
# Auto-select port with Mewtwo
mix run scripts/train_from_replays.exs --train-character mewtwo

# Learn from BOTH players (2x data)
mix run scripts/train_from_replays.exs --dual-port

# Filter by character
mix run scripts/train_from_replays.exs --characters mewtwo,ganondorf
```

### Production Training
```bash
mix run scripts/train_from_replays.exs \
  --preset production \
  --train-character mewtwo \
  --augment --cache-augmented \
  --early-stopping --patience 10 \
  --name mewtwo_v1
```

## Key Flags Quick Reference

| Flag | What it does |
|------|--------------|
| `--preset NAME` | Load preset (quick, standard, full, mewtwo, ganondorf) |
| `--temporal` | Enable temporal training (sequences) |
| `--backbone TYPE` | mamba, lstm, liquid, rwkv, gla, zamba, + 9 more (see TRAINING.md) |
| `--train-character CHAR` | Auto-select port with character |
| `--dual-port` | Train on both players (2x data) |
| `--augment` | Enable data augmentation |
| `--cache-augmented` | Precompute augmented data (~100x speedup) |
| `--val-split X` | Validation split (e.g., 0.1 = 10%) |
| `--early-stopping` | Stop when validation stops improving |
| `--save-best` | Save checkpoint on improvement |
| `--resume PATH` | Resume from checkpoint |
| `--dry-run` | Validate config without training |
| `--verbose` | Show debug output |
| `--quiet` | Suppress non-error output |

## Performance Tips

```bash
# Faster training with cached augmentation
mix run scripts/train_from_replays.exs --augment --cache-augmented

# Memory-bounded for large datasets
mix run scripts/train_from_replays.exs --stream-chunk-size 50 --gc-every 100

# Gradient accumulation for larger effective batch
mix run scripts/train_from_replays.exs --batch-size 64 --accumulation-steps 4  # Effective: 256
```

## Learning Rate Schedules

```bash
# Cosine annealing with warm restarts (recommended)
--lr-schedule cosine_restarts --warmup-steps 500 --restart-period 1000

# Linear warmup then decay
--lr-schedule cosine --warmup-steps 1000 --decay-steps 10000

# Constant (default)
--lr-schedule constant
```

## Checkpointing

```bash
# Auto-generated name
mix run scripts/train_from_replays.exs --save-best
# => checkpoints/mewtwo_mamba_fancy_name_20260131_123456.axon

# Custom name
mix run scripts/train_from_replays.exs --name mymodel --save-best
# => checkpoints/mymodel_20260131_123456.axon

# Resume training
mix run scripts/train_from_replays.exs --resume checkpoints/model.axon

# Save every N epochs
mix run scripts/train_from_replays.exs --save-every 5

# Streaming mode: save every N batches
mix run scripts/train_from_replays.exs --save-every-batches 1000
```

## After Training

```bash
# Evaluate model
mix run scripts/eval_model.exs --checkpoint checkpoints/model.axon

# Play against Dolphin
mix run scripts/play_dolphin_async.exs \
  --policy checkpoints/model_policy.bin \
  --dolphin ~/.config/Slippi\ Launcher/netplay \
  --iso ~/melee.iso
```

## Environment Variables

```bash
export EXPHIL_REPLAYS_DIR=/path/to/replays  # Default replays directory
export EXPHIL_WANDB_PROJECT=my-project      # W&B project name
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | `--batch-size 32 --gc-every 50` |
| Training too slow | `--cache-augmented` or `--max-files 50` |
| Bad checkpoint | `--dry-run` to validate config first |
| Can't find replays | `--replays /absolute/path` |
| JIT too slow | First batch JIT takes 2-5 min, then fast |

## See Also

- [TRAINING.md](TRAINING.md) - Full documentation
- [TRAINING_FEATURES.md](TRAINING_FEATURES.md) - Feature details
- [Architecture Guide](../reference/architectures/ARCHITECTURE_GUIDE.md) - Backbone explanations
- [GOTCHAS.md](../reference/GOTCHAS.md) - Common pitfalls
