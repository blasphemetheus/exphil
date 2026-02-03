# ExPhil Scripts

This document describes all available scripts in the `scripts/` directory.

## Quick Reference

| Category | Script | Key Command |
|----------|--------|-------------|
| **Training** | train_from_replays.exs | `--preset standard --temporal` |
| | train_ppo.exs | `--policy policy.bin` |
| | train_self_play.exs | `--game-type mock` |
| | train_distillation.exs | `--soft-labels labels.bin` |
| | train_kmeans.exs | `--replays ./replays --k 21` |
| | find_lr.exs | `--replays ./replays` |
| **Evaluation** | eval_model.exs | `--checkpoint model.axon` |
| | analyze_replays.exs | `--replays ./replays` |
| | scan_replays.exs | `--replays ./replays` |
| **League** | run_league.exs | `--replays ./replays` |
| | league_report.exs | `--checkpoint-dir ./checkpoints` |
| **Benchmarks** | benchmark_architectures.exs | `--replays ./replays` |
| | benchmark_attention.exs | `--seq-lens 64,128,256` |
| | benchmark_embeddings.exs | `--iterations 1000` |
| **Play** | play_dolphin.exs | `--policy --dolphin --iso` |
| | play_dolphin_async.exs | Same (recommended for slow models) |
| | example_bot.exs | `--dolphin --iso` |
| **Export** | export_onnx.exs | `--policy policy.bin` |
| | export_numpy.exs | `--policy policy.bin` |
| **Utilities** | registry.exs | `list`, `show`, `best` |
| | inspect_cache.exs | (no args) |

## Training Scripts

### train_from_replays.exs
**Main training script for imitation learning from Slippi replays.**

```bash
# Basic training
mix run scripts/train_from_replays.exs --replays /path/to/replays

# Temporal training with Mamba backbone
mix run scripts/train_from_replays.exs --temporal --backbone mamba --epochs 10

# Using presets
mix run scripts/train_from_replays.exs --preset quick      # Fast iteration (~5 min)
mix run scripts/train_from_replays.exs --preset standard   # Default settings (~30 min)
mix run scripts/train_from_replays.exs --preset production # Maximum quality
mix run scripts/train_from_replays.exs --preset mewtwo     # Character-specific

# Character/stage filtering
mix run scripts/train_from_replays.exs --character mewtwo,fox --stage fd,battlefield

# With augmentation (recommended with caching)
mix run scripts/train_from_replays.exs --cache-augmented --augment --temporal --backbone mamba
```

Key options:
- `--replays PATH` - Directory containing .slp files
- `--epochs N` - Number of training epochs
- `--temporal` - Enable temporal (sequence) model
- `--backbone TYPE` - mlp, lstm, gru, mamba, mamba_nif, sliding_window, attention, jamba
- `--preset NAME` - quick, standard, full, production, mewtwo, ganondorf, link, gameandwatch, zelda
- `--checkpoint PATH` - Resume from checkpoint
- `--dry-run` - Validate configuration without training
- `--cache-embeddings` / `--cache-augmented` - Enable disk caching (2-100x speedup)

See [TRAINING.md](TRAINING.md) for complete CLI reference.

### train_ppo.exs
**PPO reinforcement learning training.**

```bash
# Test PPO loop with mock environment
mix run scripts/train_ppo.exs --mock --pretrained checkpoints/policy.bin --timesteps 10000

# Full PPO training with Dolphin
mix run scripts/train_ppo.exs \
  --pretrained checkpoints/policy.bin \
  --dolphin /path/to/slippi \
  --iso /path/to/melee.iso \
  --character mewtwo \
  --opponent cpu3 \
  --timesteps 100000
```

Key options:
- `--pretrained PATH` - Starting policy (recommended)
- `--timesteps N` - Total training timesteps
- `--rollout-length N` - Steps per rollout (default: 2048)
- `--mock` - Use mock environment for testing

### train_self_play.exs
**Self-play training with population-based training and Elo tracking.**

```bash
# Quick test with mock environment
mix run scripts/train_self_play.exs --game-type mock --timesteps 1000 --max-episode-frames 600

# Full training with pretrained policy
mix run scripts/train_self_play.exs \
  --pretrained checkpoints/imitation_policy.bin \
  --num-games 8 \
  --timesteps 100000 \
  --track-elo

# League mode (AlphaStar-style)
mix run scripts/train_self_play.exs --mode league --num-games 4
```

Key options:
- `--mode TYPE` - simple_mix (default) or league
- `--game-type TYPE` - mock or dolphin
- `--num-games N` - Parallel games to run
- `--track-elo` - Enable Elo rating tracking
- `--max-episode-frames N` - Max frames per episode (default: 28800 = ~8 min)

### train_distillation.exs
**Knowledge distillation from teacher to student model.**

```bash
# First generate soft labels from teacher
mix run scripts/generate_soft_labels.exs \
  --teacher checkpoints/mamba_policy.bin \
  --replays /path/to/replays \
  --output soft_labels.bin

# Then train student
mix run scripts/train_distillation.exs \
  --soft-labels soft_labels.bin \
  --hidden 64,64 \
  --epochs 10 \
  --alpha 0.7
```

Key options:
- `--soft-labels PATH` - Soft label file from generate_soft_labels
- `--hidden N,N` - Student hidden sizes (default: 64,64)
- `--alpha F` - Soft vs hard label weight (default: 0.7)

### train_kmeans.exs
**Train K-means clustering for stick discretization (~5% better precision).**

```bash
mix run scripts/train_kmeans.exs \
  --replays /path/to/replays \
  --k 21 \
  --output priv/kmeans_centers.nx

# Then use in training
mix run scripts/train_from_replays.exs --kmeans-centers priv/kmeans_centers.nx
```

Key options:
- `--replays DIR` - Replay directory
- `--k NUM` - Number of clusters (default: 21)
- `--max-files NUM` - Limit files to process
- `--max-iters NUM` - K-means iterations (default: 100)

### find_lr.exs
**Learning rate finder using Leslie Smith's LR range test.**

```bash
mix run scripts/find_lr.exs \
  --replays /path/to/replays \
  --min-lr 1e-7 \
  --max-lr 1.0 \
  --num-steps 100
```

Outputs a loss vs LR curve and suggests optimal learning rate.

## Evaluation & Analysis Scripts

### eval_model.exs
**Evaluate a trained model's performance on replay data.**

```bash
# Evaluate single model
mix run scripts/eval_model.exs --checkpoint checkpoints/model.axon

# Evaluate exported policy
mix run scripts/eval_model.exs --policy checkpoints/policy.bin --replays /path/to/replays

# Compare multiple models
mix run scripts/eval_model.exs --compare model1.axon model2.axon model3.axon

# Detailed per-component metrics
mix run scripts/eval_model.exs --checkpoint model.axon --detailed
```

Key options:
- `--checkpoint PATH` or `--policy PATH` - Model to evaluate
- `--compare P1 P2 ...` - Compare multiple models
- `--detailed` - Show per-button, per-stick metrics
- `--output PATH` - Save results to JSON

### analyze_replays.exs
**Analyze replay data for action distributions and training insights.**

```bash
mix run scripts/analyze_replays.exs \
  --replays /path/to/replays \
  --player-port 1 \
  --top-actions 15 \
  --show-stages \
  --show-positions
```

Shows:
- Action state distribution (top N)
- Button press frequencies
- Stick position heatmaps (if `--show-positions`)
- Stage distribution (if `--show-stages`)
- Potential training issues (rare actions, imbalances)

### scan_replays.exs
**Fast replay scanner for character/matchup statistics without full parsing.**

```bash
mix run scripts/scan_replays.exs --replays /path/to/replays --max-files 1000
```

Shows:
- Total replay count
- Games per character
- Low-tier character summary
- Training recommendations

## League System Scripts

### run_league.exs
**Run architecture league competition (compares different backbones).**

```bash
mix run scripts/run_league.exs \
  --replays /path/to/replays \
  --architectures mlp,mamba,lstm,attention,jamba \
  --target-loss 1.0 \
  --max-epochs 50 \
  --generations 10 \
  --matches-per-pair 20 \
  --checkpoint-dir checkpoints/league
```

Key options:
- `--architectures LIST` - Backbones to compare (default: all)
- `--target-loss FLOAT` - Pretraining stopping point
- `--generations N` - Evolution generations
- `--matches-per-pair N` - Games per matchup

### league_report.exs
**Generate reports from architecture league competitions.**

```bash
# HTML report (default)
mix run scripts/league_report.exs \
  --checkpoint-dir checkpoints/league \
  --output report.html \
  --include-charts

# JSON export
mix run scripts/league_report.exs \
  --league-state league_state.json \
  --format json

# Terminal output
mix run scripts/league_report.exs --format terminal
```

Key options:
- `--format TYPE` - html, json, or terminal
- `--include-charts` - Add visualization charts
- `--include-history` - Include training history
- `--theme TYPE` - light or dark

## Benchmarking Scripts

### benchmark_architectures.exs
**Comprehensive benchmark comparing all backbone architectures.**

```bash
mix run scripts/benchmark_architectures.exs \
  --replays /path/to/replays \
  --max-files 30 \
  --epochs 3 \
  --cache-embeddings

# Skip specific architectures
mix run scripts/benchmark_architectures.exs --skip lstm,gru

# Run only specific architectures
mix run scripts/benchmark_architectures.exs --only mlp,mamba
```

**Output:**
- `checkpoints/benchmark_results.json` - Raw results
- `checkpoints/benchmark_report.html` - Visual report with charts

Compares: MLP, LSTM, GRU, Mamba, Attention, Jamba

Key options:
- `--only ARCH,...` - Run only these architectures
- `--skip ARCH,...` - Skip these architectures
- `--continue-on-error` - Don't stop on failures
- `--cache-embeddings` - Cache embeddings for speed
- `--gpu-prealloc FRAC` - Pre-allocate GPU memory (e.g., 0.7)
- `--lazy-sequences` - Memory-efficient (150MB) vs eager (13GB)

### benchmark_attention.exs
**Benchmark attention implementation variants.**

```bash
mix run scripts/benchmark_attention.exs \
  --seq-lens 32,64,128,256,512 \
  --iterations 100 \
  --quiet
```

Compares:
- **Standard**: O(n²) memory, fastest for short sequences
- **Chunked**: Lower peak memory, same algorithm
- **Memory-Efficient**: O(n) memory using online softmax
- **NIF FlashAttention**: Native Rust/CUDA (forward-only)

### benchmark_embeddings.exs
**Benchmark different embedding configurations.**

```bash
mix run scripts/benchmark_embeddings.exs \
  --iterations 1000 \
  --batch-size 32
```

Compares learned vs one-hot modes for actions, characters, stages, and Nana.

### benchmark_mixed_precision.exs
**Compare BF16 vs F32 precision for training speed/accuracy.**

```bash
mix run scripts/benchmark_mixed_precision.exs
```

### benchmark_mamba_training.exs
**Profile Mamba architecture training performance.**

```bash
mix run scripts/benchmark_mamba_training.exs
```

### benchmark_mamba_vs_gated.exs
**Compare Mamba vs GatedSSM architectures.**

```bash
mix run scripts/benchmark_mamba_vs_gated.exs
```

### benchmark_fused_ops.exs
**Benchmark fused kernel operations.**

```bash
mix run scripts/benchmark_fused_ops.exs
```

### benchmark_nif_scan.exs
**Benchmark NIF-based operations (Rust/C backend).**

```bash
mix run scripts/benchmark_nif_scan.exs
```

### test_gpu_speed.exs
**Quick smoke test for GPU training performance.**

```bash
mix run scripts/test_gpu_speed.exs
```

## Dolphin Integration Scripts

### play_dolphin.exs
**Play against a trained agent in Dolphin (synchronous).**

Best for fast models (<16ms inference).

```bash
mix run scripts/play_dolphin.exs \
  --policy checkpoints/policy.bin \
  --dolphin ~/.config/Slippi\ Launcher/netplay \
  --iso ~/melee.iso \
  --character mewtwo \
  --stage final_destination \
  --port 1 \
  --opponent-port 2
```

Key options:
- `--policy PATH` - Exported policy file (required)
- `--dolphin PATH` - Slippi/Dolphin folder (required)
- `--iso PATH` - Melee 1.02 ISO (required)
- `--frame-delay N` - Simulate online delay
- `--deterministic` - Disable action sampling (greedy)
- `--action-repeat N` - Compute action every N frames

### play_dolphin_async.exs
**Play against a trained agent in Dolphin (asynchronous).**

Recommended for slower models (LSTM, Mamba). Separates frame reading from inference.

```bash
mix run scripts/play_dolphin_async.exs \
  --policy checkpoints/policy.bin \
  --dolphin ~/.config/Slippi\ Launcher/netplay \
  --iso ~/melee.iso \
  --on-game-end restart
```

Additional options:
- `--on-game-end MODE` - restart (auto-start next game) or stop (exit)

### example_bot.exs
**Example rule-based bot demonstrating the bridge API.**

```bash
mix run scripts/example_bot.exs \
  --dolphin /path/to/slippi \
  --iso /path/to/melee.iso \
  --character mewtwo
```

## Model Export Scripts

### export_onnx.exs
**Export trained model to ONNX format.**

```bash
mix run scripts/export_onnx.exs \
  --policy checkpoints/policy.bin \
  --output model.onnx
```

Note: May have compatibility issues with Nx 0.10+. Use export_numpy.exs as workaround.

### export_numpy.exs
**Export model weights to NumPy format for Python/ONNX conversion.**

```bash
mix run scripts/export_numpy.exs \
  --policy checkpoints/policy.bin \
  --output exports/
```

Then convert in Python:
```bash
python priv/python/build_onnx_from_numpy.py exports/
```

### generate_soft_labels.exs
**Generate soft labels from teacher model for knowledge distillation.**

```bash
mix run scripts/generate_soft_labels.exs \
  --teacher checkpoints/mamba_policy.bin \
  --replays /path/to/replays \
  --output soft_labels.bin \
  --temperature 2.0 \
  --max-files 100
```

Temperature:
- T=1.0: Original (sharp) distributions
- T=2.0: Softer distributions (recommended)
- T>3.0: Very soft, may lose information

### test_onnx_pipeline.exs
**Test full ONNX export + INT8 quantization pipeline.**

```bash
mix run scripts/test_onnx_pipeline.exs
```

Tests: LSTM model creation, ONNX export, INT8 quantization, benchmark comparison.

### test_flash_attention.exs
**Test Python/PyTorch FlashAttention bridge.**

```bash
mix run scripts/test_flash_attention.exs
```

## Model Management Scripts

### registry.exs
**Model registry management CLI.**

```bash
# List all models
mix run scripts/registry.exs list
mix run scripts/registry.exs list --tags production --backbone mamba --limit 10 --json

# Show model details
mix run scripts/registry.exs show MODEL_ID

# Find best model
mix run scripts/registry.exs best

# Tag models
mix run scripts/registry.exs tag MODEL_ID production validated
mix run scripts/registry.exs untag MODEL_ID experimental

# Show model lineage
mix run scripts/registry.exs lineage MODEL_ID

# Delete model
mix run scripts/registry.exs delete MODEL_ID          # Keeps files
mix run scripts/registry.exs delete MODEL_ID --files  # Deletes files too
```

### inspect_cache.exs
**Inspect structure and metadata of cached embeddings.**

```bash
mix run scripts/inspect_cache.exs
```

Uses `EXPHIL_CACHE_DIR` environment variable (default: `cache/embeddings`).

### regenerate_benchmark_html.exs
**Regenerate HTML reports from benchmark results.**

```bash
mix run scripts/regenerate_benchmark_html.exs
```

## Mix Tasks (CLI Integration)

### mix exphil.setup
**Interactive setup wizard for configuring training.**

```bash
mix exphil.setup
```

Walks through:
1. Goal selection (quick experiment, character training, production, fine-tuning)
2. Character selection
3. Hardware detection and batch size recommendation
4. Replay directory configuration
5. Advanced options (backbone, augmentation, W&B logging)

### mix exphil.info [PATH]
**Show detailed information about a checkpoint or policy file.**

```bash
mix exphil.info checkpoints/model.axon
```

Displays:
- File type, size, modification date
- Architecture: mode, backbone, window size
- Training config: LR, batch size, optimization
- Parameter counts by layer

### mix exphil.list [OPTIONS]
**List all checkpoints with metadata.**

```bash
mix exphil.list
mix exphil.list --dir ./checkpoints --sort date --reverse
```

Options:
- `--dir PATH` - Checkpoint directory (default: ./checkpoints)
- `--sort TYPE` - date, size, or name
- `--reverse` - Reverse sort order

### mix exphil.compare [PATH_A] [PATH_B]
**Side-by-side comparison of two checkpoints.**

```bash
mix exphil.compare model_v1.axon model_v2.axon
mix exphil.compare model_v1.axon model_v2.axon --all --metrics
```

Options:
- `--all` - Show all config fields (not just differences)
- `--metrics` - Focus on training metrics

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EXPHIL_REPLAYS_DIR` | `./replays` | Default replay directory |
| `EXPHIL_WANDB_PROJECT` | `exphil` | Default W&B project name |
| `EXPHIL_CACHE_DIR` | `cache/embeddings` | Embedding cache location |
| `SLIPPI_PATH` | - | Default Dolphin/Slippi path |
| `MELEE_ISO` | - | Default Melee ISO path |
| `XLA_FLAGS` | - | EXLA/XLA configuration |
