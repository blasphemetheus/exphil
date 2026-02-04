# Mewtwo Architecture Exploration Plan

Training plan to explore ExPhil's capabilities using 129 Mewtwo replays.

**Hardware:**
- Training: RunPod RTX 4090 (24GB VRAM)
- Inference/Playtesting: Local machine (Intel Iris Xe, 8 cores, 15GB RAM)

---

## Quick Start: Automated Comparison

Run all architectures with one command:

```bash
# On RunPod
./scripts/train_all_architectures.sh --character mewtwo --replays /workspace/replays/mewtwo
```

This runs MLP, LSTM, GRU, Mamba, Attention, and Jamba with identical hyperparameters for fair comparison.

---

## Key Training Features

### Learned Embeddings (Default)
New default embeddings (~287 dims) vs old (1204 dims). Enables 2x larger networks at same speed.

```bash
# Explicit (already default):
--action-mode learned --character-mode learned --stage-mode compact
```

### Focal Loss for Rare Actions
Mewtwo relies on rare buttons: Z (teleport cancel), L/R (wavedash, teching).
Focal loss upweights hard-to-predict actions.

```bash
--focal-loss --focal-gamma 2.0
```

### Label Smoothing
Prevents overconfidence, improves generalization.

```bash
--label-smoothing 0.1
```

### Online Robustness
Frame delay augmentation for netplay (handles 2-4 frame input delay).

```bash
--online-robust
```

---

## Data Expectations

| Replay Count | Expected Outcome |
|--------------|------------------|
| 129 (current) | Basic movement, some combos, inconsistent neutral |
| 500+ | Solid neutral game, better recovery decisions |
| 2000+ | Tournament-viable, matchup-aware, consistent |

**TODO:** Sync more Mewtwo replays from slippi.gg archives when available.

---

## Phase 1: Baseline Models (Quick Validation)

Verify the training pipeline works with fast experiments.

### 1.1 MLP Baseline (Single-Frame)
**Purpose:** Establish baseline accuracy without temporal context.

```bash
# On RunPod
mix run scripts/train_from_replays.exs \
  --replays /workspace/replays/mewtwo \
  --train-character mewtwo \
  --epochs 10 \
  --batch-size 128 \
  --hidden-sizes 128,128 \
  --lr-schedule cosine \
  --early-stopping \
  --name mlp_mewtwo_baseline
```

**Expected:** ~2.5-3.0 loss, 70-75% accuracy, ~5-10 min training

### 1.2 Quick Temporal Test (Mamba)
**Purpose:** Verify temporal training works.

```bash
mix run scripts/train_from_replays.exs \
  --replays /workspace/replays/mewtwo \
  --train-character mewtwo \
  --temporal \
  --backbone mamba \
  --epochs 5 \
  --batch-size 64 \
  --hidden-sizes 128,128 \
  --window-size 30 \
  --name mamba_mewtwo_quick
```

**Expected:** Lower loss than MLP due to temporal context, ~15-20 min

---

## Phase 2: Architecture Comparison

Train identical configs across all backbones to compare architectures.

**Shared Config:**
- Epochs: 20
- Batch size: 128
- Hidden: [256, 256]
- Window: 60 frames
- LR: cosine schedule with warmup
- Early stopping: patience=7
- **Focal loss: gamma=2.0** (better rare action learning)
- **Learned embeddings** (default, ~287 dims)

### 2.1 MLP (No Temporal)
```bash
mix run scripts/train_from_replays.exs \
  --replays /workspace/replays/mewtwo \
  --train-character mewtwo \
  --epochs 20 \
  --batch-size 128 \
  --hidden-sizes 256,256 \
  --lr-schedule cosine \
  --warmup-steps 500 \
  --early-stopping --patience 7 \
  --focal-loss --focal-gamma 2.0 \
  --name mlp_mewtwo_full
```

### 2.2 LSTM
```bash
mix run scripts/train_from_replays.exs \
  --replays /workspace/replays/mewtwo \
  --train-character mewtwo \
  --temporal \
  --backbone lstm \
  --epochs 20 \
  --batch-size 128 \
  --hidden-sizes 256,256 \
  --window-size 60 \
  --lr-schedule cosine \
  --warmup-steps 500 \
  --early-stopping --patience 7 \
  --focal-loss --focal-gamma 2.0 \
  --name lstm_mewtwo_full
```

### 2.3 GRU
```bash
mix run scripts/train_from_replays.exs \
  --replays /workspace/replays/mewtwo \
  --train-character mewtwo \
  --temporal \
  --backbone gru \
  --epochs 20 \
  --batch-size 128 \
  --hidden-sizes 256,256 \
  --window-size 60 \
  --lr-schedule cosine \
  --warmup-steps 500 \
  --early-stopping --patience 7 \
  --focal-loss --focal-gamma 2.0 \
  --name gru_mewtwo_full
```

### 2.4 Mamba (Recommended)
```bash
mix run scripts/train_from_replays.exs \
  --replays /workspace/replays/mewtwo \
  --train-character mewtwo \
  --temporal \
  --backbone mamba \
  --epochs 20 \
  --batch-size 128 \
  --hidden-sizes 256,256 \
  --window-size 60 \
  --num-layers 4 \
  --lr-schedule cosine \
  --warmup-steps 500 \
  --early-stopping --patience 7 \
  --focal-loss --focal-gamma 2.0 \
  --name mamba_mewtwo_full
```

### 2.5 Sliding Window Attention
```bash
mix run scripts/train_from_replays.exs \
  --replays /workspace/replays/mewtwo \
  --train-character mewtwo \
  --temporal \
  --backbone sliding_window \
  --epochs 20 \
  --batch-size 128 \
  --hidden-sizes 256,256 \
  --window-size 60 \
  --num-heads 4 \
  --attention-every 2 \
  --lr-schedule cosine \
  --warmup-steps 500 \
  --early-stopping --patience 7 \
  --focal-loss --focal-gamma 2.0 \
  --name attention_mewtwo_full
```

### 2.6 Jamba (Mamba + Attention Hybrid)
```bash
mix run scripts/train_from_replays.exs \
  --replays /workspace/replays/mewtwo \
  --train-character mewtwo \
  --temporal \
  --backbone jamba \
  --epochs 20 \
  --batch-size 128 \
  --hidden-sizes 256,256 \
  --window-size 60 \
  --num-layers 6 \
  --lr-schedule cosine \
  --warmup-steps 500 \
  --early-stopping --patience 7 \
  --focal-loss --focal-gamma 2.0 \
  --name jamba_mewtwo_full
```

---

## Phase 3: Hyperparameter Exploration

Using the best architecture from Phase 2, explore key hyperparameters.

### 3.1 Model Size Comparison
```bash
# Small (fast inference, limited capacity)
mix run scripts/train_from_replays.exs \
  --replays /workspace/replays/mewtwo \
  --train-character mewtwo \
  --temporal --backbone mamba \
  --epochs 20 --batch-size 128 \
  --hidden-sizes 64,64 \
  --window-size 60 \
  --name mamba_mewtwo_small

# Medium
mix run scripts/train_from_replays.exs \
  --replays /workspace/replays/mewtwo \
  --train-character mewtwo \
  --temporal --backbone mamba \
  --epochs 20 --batch-size 128 \
  --hidden-sizes 256,256 \
  --window-size 60 \
  --name mamba_mewtwo_medium

# Large (best quality, slower inference)
mix run scripts/train_from_replays.exs \
  --replays /workspace/replays/mewtwo \
  --train-character mewtwo \
  --temporal --backbone mamba \
  --epochs 20 --batch-size 256 \
  --hidden-sizes 512,512 \
  --window-size 60 \
  --num-layers 6 \
  --name mamba_mewtwo_large
```

### 3.2 Window Size Comparison
```bash
# Short window (faster, less context)
mix run scripts/train_from_replays.exs \
  --replays /workspace/replays/mewtwo \
  --train-character mewtwo \
  --temporal --backbone mamba \
  --epochs 20 --batch-size 128 \
  --hidden-sizes 256,256 \
  --window-size 30 \
  --name mamba_mewtwo_win30

# Standard window
mix run scripts/train_from_replays.exs \
  --replays /workspace/replays/mewtwo \
  --train-character mewtwo \
  --temporal --backbone mamba \
  --epochs 20 --batch-size 128 \
  --hidden-sizes 256,256 \
  --window-size 60 \
  --name mamba_mewtwo_win60

# Long window (more context, higher memory)
mix run scripts/train_from_replays.exs \
  --replays /workspace/replays/mewtwo \
  --train-character mewtwo \
  --temporal --backbone mamba \
  --epochs 20 --batch-size 64 \
  --hidden-sizes 256,256 \
  --window-size 120 \
  --name mamba_mewtwo_win120
```

### 3.3 Training Features
```bash
# With data augmentation (mirror states)
mix run scripts/train_from_replays.exs \
  --replays /workspace/replays/mewtwo \
  --train-character mewtwo \
  --temporal --backbone mamba \
  --epochs 20 --batch-size 128 \
  --hidden-sizes 256,256 \
  --augment \
  --name mamba_mewtwo_augmented

# With label smoothing (better generalization)
mix run scripts/train_from_replays.exs \
  --replays /workspace/replays/mewtwo \
  --train-character mewtwo \
  --temporal --backbone mamba \
  --epochs 20 --batch-size 128 \
  --hidden-sizes 256,256 \
  --label-smoothing 0.1 \
  --name mamba_mewtwo_smoothed

# Online-robust (frame delay augmentation for netplay)
mix run scripts/train_from_replays.exs \
  --replays /workspace/replays/mewtwo \
  --train-character mewtwo \
  --temporal --backbone mamba \
  --epochs 20 --batch-size 128 \
  --hidden-sizes 256,256 \
  --online-robust \
  --name mamba_mewtwo_online
```

---

## Phase 4: Best Model (Production Training)

Combine best settings from previous phases.

```bash
mix run scripts/train_from_replays.exs \
  --replays /workspace/replays/mewtwo \
  --train-character mewtwo \
  --temporal \
  --backbone mamba \
  --epochs 50 \
  --batch-size 256 \
  --hidden-sizes 512,512 \
  --num-layers 6 \
  --window-size 60 \
  --lr-schedule cosine \
  --warmup-steps 1000 \
  --early-stopping --patience 10 \
  --augment \
  --label-smoothing 0.1 \
  --ema \
  --wandb \
  --name mamba_mewtwo_production
```

---

## Workflow: RunPod → Local

### 1. Sync Checkpoints from RunPod
```bash
# On RunPod, after training
sync-all-up  # Or: rclone copy checkpoints/ b2:exphil-artifacts/checkpoints/ --progress

# On local machine
rclone copy b2:exphil-artifacts/checkpoints/ ~/git/melee/exphil/checkpoints/ --progress
```

### 2. Evaluate Models Locally
```bash
# Quick comparison of all models
mix run scripts/eval_model.exs --compare \
  checkpoints/mlp_mewtwo_full_best_policy.bin \
  checkpoints/lstm_mewtwo_full_best_policy.bin \
  checkpoints/mamba_mewtwo_full_best_policy.bin

# Detailed evaluation of best model
mix run scripts/eval_model.exs \
  --policy checkpoints/mamba_mewtwo_production_best_policy.bin \
  --replays replays/mewtwo \
  --detailed
```

### 3. Playtest Against Dolphin
```bash
# Start Dolphin with netplay ISO
# Then run agent

# Async runner (recommended)
source .venv/bin/activate
mix run scripts/play_dolphin_async.exs \
  --policy checkpoints/mamba_mewtwo_production_best_policy.bin \
  --dolphin ~/.config/Slippi\ Launcher/netplay \
  --iso ~/path/to/melee.iso \
  --character mewtwo

# Or sync runner for debugging
mix run scripts/play_dolphin.exs \
  --policy checkpoints/mamba_mewtwo_production_best_policy.bin \
  --dolphin ~/.config/Slippi\ Launcher/netplay \
  --iso ~/path/to/melee.iso
```

---

## Expected Results Matrix

| Model | Est. Loss | Est. Accuracy | Training Time | Inference Speed |
|-------|-----------|---------------|---------------|-----------------|
| MLP baseline | 2.5-3.0 | 70-75% | 5-10 min | <1ms |
| LSTM | 2.0-2.5 | 75-80% | 30-45 min | ~220ms (too slow) |
| GRU | 2.0-2.5 | 75-80% | 25-40 min | ~150ms (too slow) |
| Mamba | 1.8-2.2 | 78-83% | 20-35 min | ~9ms (60fps ready) |
| Attention | 1.9-2.3 | 77-82% | 35-50 min | ~15ms (60fps ready) |
| Jamba | 1.7-2.1 | 80-85% | 40-60 min | ~12ms (60fps ready) |

**Key insight:** LSTM/GRU have good accuracy but inference is too slow for real-time play. Mamba and Jamba are the practical choices for playable bots.

---

## Quick Reference: Presets

Instead of manual flags, use presets:

```bash
# Quick iteration (2 epochs, small model)
mix run scripts/train_from_replays.exs --preset quick --train-character mewtwo

# Standard training (10 epochs, balanced)
mix run scripts/train_from_replays.exs --preset standard --train-character mewtwo --temporal --backbone mamba

# Full quality (20 epochs, all features)
mix run scripts/train_from_replays.exs --preset full --train-character mewtwo --temporal --backbone mamba
```

---

## Tracking Results

Add `--wandb` to any training command to log metrics to Weights & Biases:

```bash
# Set up wandb (one time)
wandb login

# Train with logging
mix run scripts/train_from_replays.exs \
  --replays /workspace/replays/mewtwo \
  --train-character mewtwo \
  --temporal --backbone mamba \
  --epochs 20 \
  --wandb \
  --name mamba_experiment_1
```

This enables:
- Loss curves comparison across runs
- Hyperparameter tracking
- Model artifact storage
