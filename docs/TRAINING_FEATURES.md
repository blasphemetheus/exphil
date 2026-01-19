# Training Features Design

This document outlines planned and speculative features for the ExPhil training system.

## Current State

### Model Naming (Implemented)
```
checkpoints/{backbone}_{YYYYMMDD_HHMMSS}.axon
checkpoints/{backbone}_{YYYYMMDD_HHMMSS}_policy.bin
checkpoints/{backbone}_{YYYYMMDD_HHMMSS}_config.json
```

Example: `mamba_20260119_123456.axon`

### Config JSON (Implemented)
Saves all training parameters alongside the model for reproducibility.

---

## Proposed Features

### 1. Training Presets

Add `--preset` flag to quickly configure common training scenarios.

```bash
# Quick iteration (testing changes)
mix run scripts/train_from_replays.exs --preset quick

# Standard training (balanced)
mix run scripts/train_from_replays.exs --preset standard

# Full training (maximum quality)
mix run scripts/train_from_replays.exs --preset full

# Character-optimized
mix run scripts/train_from_replays.exs --preset mewtwo
```

#### Preset Definitions

| Preset | Epochs | Max Files | Hidden | Temporal | Backbone | Window |
|--------|--------|-----------|--------|----------|----------|--------|
| `quick` | 1 | 5 | 32,32 | no | - | - |
| `standard` | 10 | 50 | 64,64 | no | - | - |
| `full` | 50 | all | 256,256 | yes | mamba | 60 |
| `full-cpu` | 20 | 100 | 128,128 | no | - | - |

#### Character Presets

| Character | Notes | Window | Special |
|-----------|-------|--------|---------|
| `mewtwo` | Complex recovery | 90 | teleport tracking |
| `ganondorf` | Spacing-focused | 60 | - |
| `link` | Projectiles | 75 | item tracking |
| `gameandwatch` | No L-cancel | 45 | hammer RNG |
| `zelda` | Transform | 60 | dual-character |

#### Implementation

```elixir
defmodule ExPhil.Training.Config do
  def preset(:quick) do
    [epochs: 1, max_files: 5, hidden_sizes: [32, 32], temporal: false]
  end

  def preset(:standard) do
    [epochs: 10, max_files: 50, hidden_sizes: [64, 64], temporal: false]
  end

  def preset(:full) do
    [epochs: 50, max_files: nil, hidden_sizes: [256, 256],
     temporal: true, backbone: :mamba, window_size: 60]
  end

  def preset(:mewtwo) do
    preset(:full) |> Keyword.merge([character: :mewtwo, window_size: 90])
  end
end
```

---

### 2. Enhanced Model Naming

#### Option A: Include Character
```
checkpoints/{character}_{backbone}_{timestamp}.axon
```
Example: `mewtwo_mamba_20260119_123456.axon`

#### Option B: Include Key Hyperparameters
```
checkpoints/{backbone}_h{hidden}_w{window}_{timestamp}.axon
```
Example: `mamba_h256_w60_20260119_123456.axon`

#### Option C: Include Performance
```
checkpoints/{backbone}_{timestamp}_loss{val_loss}.axon
```
Example: `mamba_20260119_123456_loss4.05.axon`

#### Option D: Semantic Versioning (Manual)
```
checkpoints/{character}_{backbone}_v{major}.{minor}.axon
```
Example: `mewtwo_mamba_v1.2.axon`

#### Recommended: Hybrid Approach
```
checkpoints/{character}/{backbone}_{timestamp}.axon
```

With directory structure:
```
checkpoints/
├── mewtwo/
│   ├── mamba_20260119_123456.axon
│   ├── mamba_20260119_123456_config.json
│   └── mamba_20260119_123456_policy.bin
├── ganondorf/
│   └── mlp_20260118_234567.axon
└── registry.json
```

---

### 3. Model Registry

Central JSON file tracking all trained models.

#### Registry Structure

```json
{
  "models": {
    "mamba_20260119_123456": {
      "id": "mamba_20260119_123456",
      "path": "checkpoints/mamba_20260119_123456.axon",
      "policy_path": "checkpoints/mamba_20260119_123456_policy.bin",
      "config_path": "checkpoints/mamba_20260119_123456_config.json",
      "created_at": "2026-01-19T12:34:56Z",
      "character": null,
      "backbone": "mamba",
      "temporal": true,
      "hidden_sizes": [256, 256],
      "window_size": 60,
      "epochs": 10,
      "training_frames": 100000,
      "val_loss": 4.05,
      "parent_model": null,
      "tags": ["production"]
    }
  },
  "leaderboard": {
    "best_overall": "mamba_20260119_123456",
    "best_by_backbone": {
      "mamba": "mamba_20260119_123456",
      "lstm": "lstm_20260118_111111",
      "mlp": "mlp_20260117_222222"
    },
    "best_by_character": {
      "mewtwo": "mewtwo_mamba_20260119_123456"
    }
  }
}
```

#### Registry Commands

```bash
# List all models
mix run scripts/model_registry.exs list

# List by backbone
mix run scripts/model_registry.exs list --backbone mamba

# Show leaderboard
mix run scripts/model_registry.exs leaderboard

# Tag a model
mix run scripts/model_registry.exs tag mamba_20260119_123456 production

# Show model details
mix run scripts/model_registry.exs show mamba_20260119_123456

# Show lineage
mix run scripts/model_registry.exs lineage mamba_20260119_123456
```

---

### 4. Model Lineage Tracking

Track fine-tuning chains to understand model evolution.

```
base_mlp (10 epochs on 1000 replays)
    └── fine_tune_1 (5 epochs on 500 replays, PPO)
        └── fine_tune_2 (3 epochs self-play)
            └── production_v1 (tagged)
```

#### Implementation

Add to config JSON:
```json
{
  "parent_model": "base_mlp_20260118_000000",
  "training_type": "fine_tune",  // "imitation", "ppo", "self_play"
  "inherited_epochs": 10,
  "total_epochs": 15
}
```

#### Commands

```bash
# Continue training from checkpoint
mix run scripts/train_from_replays.exs \
  --from-checkpoint checkpoints/base_mlp.axon \
  --epochs 5

# Track in registry
# Automatically sets parent_model in config
```

---

### 5. Checkpoint Pruning

Automatically manage disk space by keeping only best/recent models.

#### Pruning Strategies

| Strategy | Description |
|----------|-------------|
| `keep-best-n` | Keep top N by validation loss |
| `keep-recent-n` | Keep N most recent |
| `keep-tagged` | Only keep tagged models |
| `archive` | Move old models to archive folder |

#### Implementation

```bash
# Keep only 10 best models
mix run scripts/prune_checkpoints.exs --keep-best 10

# Archive models older than 7 days
mix run scripts/prune_checkpoints.exs --archive-older-than 7d

# Dry run (show what would be deleted)
mix run scripts/prune_checkpoints.exs --keep-best 5 --dry-run
```

---

### 6. Validation & Sanity Checks

Add validation to Config module.

```elixir
def validate!(opts) do
  cond do
    opts[:epochs] <= 0 ->
      raise "epochs must be positive"
    opts[:batch_size] <= 0 ->
      raise "batch_size must be positive"
    opts[:window_size] > 120 ->
      IO.warn("window_size > 120 may cause memory issues")
    opts[:temporal] and opts[:backbone] not in [:lstm, :gru, :mamba, :sliding_window, :hybrid] ->
      raise "invalid backbone for temporal training"
    true ->
      :ok
  end
  opts
end
```

---

### 7. Training Resumption

Save optimizer state for exact training resumption.

```elixir
# Current: saves model weights only
Imitation.save_checkpoint(trainer, path)

# Proposed: save full training state
Imitation.save_training_state(trainer, path)
# Saves:
# - model weights
# - optimizer state (momentum, Adam moments)
# - epoch number
# - batch index
# - learning rate schedule position
# - RNG state for reproducibility
```

#### Usage

```bash
# Training gets interrupted at epoch 7/10
mix run scripts/train_from_replays.exs --epochs 10 ...

# Resume exactly where we left off
mix run scripts/train_from_replays.exs --resume checkpoints/mamba_incomplete.state
```

---

### 8. Early Stopping

Stop training when validation loss stops improving.

```bash
mix run scripts/train_from_replays.exs \
  --epochs 100 \
  --early-stopping \
  --patience 5 \
  --min-delta 0.01
```

| Option | Description |
|--------|-------------|
| `--early-stopping` | Enable early stopping |
| `--patience N` | Stop after N epochs without improvement |
| `--min-delta X` | Minimum improvement to count as progress |

---

## Priority Order

1. **Training Presets** - Quick win, high value for iteration speed
2. **Validation** - Catch errors early
3. **Model Registry** - Better organization as model count grows
4. **Enhanced Naming** - Easier to identify models
5. **Lineage Tracking** - Important for fine-tuning workflows
6. **Early Stopping** - Save time on long training runs
7. **Training Resumption** - Important for GPU jobs that might be interrupted
8. **Checkpoint Pruning** - Disk management

---

## Implementation Roadmap

### Phase 1: Core Improvements
- [ ] Training presets (`--preset quick|standard|full`)
- [ ] Validation in Config module
- [ ] Character flag (`--character mewtwo`)

### Phase 2: Organization
- [ ] Model registry JSON
- [ ] Registry CLI commands
- [ ] Directory structure by character

### Phase 3: Advanced
- [ ] Lineage tracking
- [ ] Early stopping
- [ ] Training resumption
- [ ] Checkpoint pruning
