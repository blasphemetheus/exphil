# Model Provenance & Training History

This document outlines the current state of model metadata tracking and planned improvements for training provenance.

**Last Updated:** 2026-01-24

---

## Current State

### What's Already Tracked

| Location | Contents |
|----------|----------|
| `checkpoints/*.axon` | Model weights, optimizer state, step counter |
| `checkpoints/*_config.json` | Training params, timestamps, metrics |
| `checkpoints/registry.json` | All models with tags, lineage, paths |

### Model Registry (`lib/exphil/training/registry.ex`)

The registry tracks per-model:

```json
{
  "id": "a1b2c3d4",
  "name": "mamba_mewtwo_20260124",
  "checkpoint_path": "checkpoints/model.axon",
  "policy_path": "checkpoints/model_policy.bin",
  "config_path": "checkpoints/model_config.json",
  "created_at": "2026-01-24T10:30:00Z",
  "training_config": { ... },
  "metrics": {
    "final_loss": 3.68,
    "epochs_completed": 10,
    "training_frames": 125000,
    "validation_frames": 13000,
    "stopped_early": false,
    "total_time_seconds": 1847
  },
  "tags": ["mewtwo", "temporal", "mamba"],
  "parent_id": null
}
```

### Config JSON Files

Alongside checkpoints, `*_config.json` files store:

```json
{
  "timestamp": "2026-01-24T10:30:00Z",
  "backbone": "mamba",
  "epochs": 10,
  "batch_size": 64,
  "temporal": true,
  "window_size": 30,
  "hidden_sizes": [256, 256],
  "training_frames": 125000,
  "validation_frames": 13000,
  "final_training_loss": 3.68,
  "total_time_seconds": 1847
}
```

---

## Gaps & Improvement Ideas

### 1. Data Provenance (HIGH PRIORITY)

**Problem:** No way to know which replay files trained a model.

| Improvement | Effort | Description | Status |
|-------------|--------|-------------|--------|
| **Replay manifest** | Medium | Save list of replay file paths used | **Done** |
| **Replay file hash** | Low | SHA256 of sorted file list for dedup detection | **Done** |
| Replay source URL | Low | If downloaded from cloud, store source | Future |
| **Per-character frame counts** | Low | Store `{mewtwo: 50000, fox: 30000}` | **Done** |

**Implementation (Complete):**
- `replay_files`: List of relative paths (if ≤500 files) for reproducibility
- `replay_manifest_hash`: SHA256 hash of sorted file list (always computed)
- `replay_count`: Number of replay files used
- `character_distribution`: Frame counts per character from training data

### 2. Character Metadata (HIGH PRIORITY)

**Problem:** Can't tell which character(s) a model was trained on.

| Improvement | Effort | Description | Status |
|-------------|--------|-------------|--------|
| **Character filter in config** | Low | Store `--characters mewtwo` explicitly | **Done** |
| **Character distribution** | Low | Store actual frame counts per character | **Done** |
| **Stage filter in config** | Low | Store `--stages battlefield` explicitly | **Done** |
| Character-specific metrics | Medium | Loss per character if multi-char | Future |

**Implementation (Complete):**
- `characters`: List of character filter atoms (e.g., `["mewtwo", "fox"]`), null if unfiltered
- `stages`: List of stage filter atoms, null if unfiltered
- `character_distribution`: Actual frame counts from training data

### 3. Training Trajectory (MEDIUM PRIORITY)

**Problem:** Only final loss saved, no history of convergence.

| Improvement | Effort | Description | Status |
|-------------|--------|-------------|--------|
| Loss history | Low | Store `[7.5, 6.2, 5.8, 5.2, ...]` per epoch | Future |
| Validation loss history | Low | Store alongside training loss | Future |
| Learning rate history | Low | Store if using scheduler | Future |
| Best epoch tracking | Low | Store which epoch had best val loss | Future |
| Gradient norm history | Medium | Track gradient health | Future |

**Implementation:**
```json
{
  "loss_history": {
    "train": [7.5, 6.2, 5.8, 5.2, 4.9],
    "val": [7.8, 6.5, 6.0, 5.5, 5.3]
  },
  "best_epoch": 5,
  "best_val_loss": 5.3
}
```

### 4. Git Provenance (MEDIUM PRIORITY)

**Problem:** Can't reproduce training without knowing code version.

| Improvement | Effort | Description | Status |
|-------------|--------|-------------|--------|
| Git commit SHA | Low | Store HEAD commit at training start | Future |
| Git branch | Low | Store current branch name | Future |
| Git dirty flag | Low | Warn if uncommitted changes | Future |
| Git remote URL | Low | Store origin for reproducibility | Future |

**Implementation:**
```json
{
  "git": {
    "commit": "abc123def456...",
    "branch": "main",
    "dirty": false,
    "remote": "git@github.com:user/exphil.git"
  }
}
```

### 5. System Information (LOW PRIORITY)

**Problem:** Can't debug hardware-specific issues.

| Improvement | Effort | Description | Status |
|-------------|--------|-------------|--------|
| GPU model | Low | Store "RTX 4090" or "M1 Max" | Future |
| CUDA/ROCm version | Low | Store driver version | Future |
| EXLA version | Low | Store Elixir ML stack versions | Future |
| Peak memory usage | Medium | Track max GPU memory during training | Future |
| CPU cores used | Low | Store `System.schedulers_online()` | Future |

### 6. Reproducibility Metadata (LOW PRIORITY)

**Problem:** Random seed alone doesn't guarantee reproducibility.

| Improvement | Effort | Description | Status |
|-------------|--------|-------------|--------|
| Random seed | Low | Already stored, ensure always captured | Partial |
| Shuffle order | Medium | Store or deterministically derive from seed | Future |
| Data split seed | Low | Store train/val split seed separately | Future |
| Augmentation seed | Low | Store if different from main seed | Future |

### 7. Model Description & Summary (MEDIUM PRIORITY)

**Problem:** No quick way to understand what a model is.

| Improvement | Effort | Description | Status |
|-------------|--------|-------------|--------|
| Human description | Low | `"Mewtwo BC, 10 epochs on ranked replays"` | Future |
| Auto-generated summary | Low | `Registry.describe/1` function | Future |
| Model comparison CLI | Medium | `mix exphil.compare model1 model2` | Done |
| Training notes field | Low | Free-form notes in registry | Future |

**Implementation:**
```elixir
Registry.describe("mamba_mewtwo_20260124")
# => "Mamba backbone, trained on 150 Mewtwo replays (125K frames),
#     10 epochs, final loss 3.68, 31 minutes training time"
```

### 8. Artifact Verification (LOW PRIORITY)

**Problem:** No way to verify checkpoint integrity.

| Improvement | Effort | Description | Status |
|-------------|--------|-------------|--------|
| Checkpoint hash | Low | SHA256 of .axon file | Future |
| Policy hash | Low | SHA256 of _policy.bin | Future |
| File size tracking | Low | Store for corruption detection | Future |
| ONNX export info | Low | Store quantization settings | Future |

---

## Implementation Priority

### Phase 1: Essential Provenance (COMPLETE)
1. ~~**Replay manifest**~~ - Know what data trained the model ✓
2. ~~**Character filter**~~ - Know what character(s) the model targets ✓
3. ~~**Character distribution**~~ - Actual frame counts per character ✓
4. ~~**Replay hash**~~ - Verify data identity ✓

### Phase 2: Training Visibility
3. Loss history - See convergence trajectory
4. Best epoch tracking - Know when to stop
5. Git commit - Reproducible code version

### Phase 3: Advanced Reproducibility
6. System info - Debug hardware issues
7. Full reproducibility metadata - Exact training reproduction
8. Artifact verification - Integrity checking

---

## Usage Examples

### Current: Identifying a Model

```bash
# Check registry
cat checkpoints/registry.json | jq '.models[] | select(.name == "mamba_mewtwo")'

# Check config
cat checkpoints/mamba_mewtwo_config.json
```

### Future: Quick Model Summary

```bash
# One-liner description
mix exphil.describe checkpoints/mamba_mewtwo.axon
# => "Mamba backbone trained on 150 Mewtwo replays (125K frames),
#     10 epochs, final loss 3.68"

# Compare two models
mix exphil.compare model_a.axon model_b.axon
# => Shows diff in config, metrics, data sources
```

### Future: Reproducibility

```bash
# Reproduce exact training
mix run scripts/train_from_replays.exs \
  --reproduce checkpoints/mamba_mewtwo_config.json

# Verify checkpoint
mix exphil.verify checkpoints/mamba_mewtwo.axon
# => "Checkpoint valid, hash matches registry"
```

---

## Related Documentation

- [TRAINING.md](TRAINING.md) - Training options and flags
- [ARCHITECTURE.md](ARCHITECTURE.md) - System design
- [GOALS.md](GOALS.md) - Project roadmap
