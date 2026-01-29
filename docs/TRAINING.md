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
# With Mamba NIF (CUDA-accelerated, fastest inference, recommended)
mix run scripts/train_from_replays.exs --temporal --backbone mamba_nif

# With Mamba (pure Nx/XLA, no NIF needed)
mix run scripts/train_from_replays.exs --temporal --backbone mamba

# With LSTM
mix run scripts/train_from_replays.exs --temporal --backbone lstm

# With sliding window attention
mix run scripts/train_from_replays.exs --temporal --backbone sliding_window

# With hybrid LSTM + attention
mix run scripts/train_from_replays.exs --temporal --backbone hybrid
```

**Note:** `mamba_nif` requires the Rust NIF to be compiled (see `native/selective_scan_nif/`).
It's 5x faster than pure Mamba (~11ms vs ~55ms inference).

**Recommended workflow:** Train with `mamba`, infer with `mamba_nif`:
```bash
# Training (mamba has correct gradients)
mix run scripts/train_from_replays.exs --temporal --backbone mamba --checkpoint model.axon

# Inference/playing (mamba_nif is 5x faster, uses same checkpoint!)
mix run scripts/play_dolphin_async.exs --policy model.axon --backbone mamba_nif
```

The NIF breaks the computation graph (can't backprop through `Nx.to_binary`), so use pure Mamba for training. Both use identical layer names, so checkpoints are interchangeable.

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
| `--backbone TYPE` | lstm | lstm, gru, mamba, mamba_nif, sliding_window, hybrid |
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
| `--qk-layernorm` | on | Normalize Q/K before attention (stabilizes training) |
| `--no-qk-layernorm` | - | Disable QK LayerNorm |
| `--chunked-attention` | off | Use chunked attention for 20-30% memory reduction |
| `--no-chunked-attention` | - | Disable chunked attention |
| `--memory-efficient-attention` | off | Use memory-efficient attention (true O(n) memory via online softmax) |
| `--no-memory-efficient-attention` | - | Disable memory-efficient attention |
| `--chunk-size N` | 32 | Chunk size for chunked/memory-efficient attention |
| `--flash-attention-nif` | off | Use FlashAttention NIF for inference (forward-only, Ampere+ GPU) |
| `--no-flash-attention-nif` | - | Disable FlashAttention NIF |

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
| `--save-every-batches N` | nil | Save checkpoint every N batches (for streaming) |
| `--resume PATH` | nil | Resume from checkpoint |
| `--precision TYPE` | f32 | f32 or bf16 (FP32 is 2x faster due to XLA issues) |
| `--mixed-precision` | false | FP32 master weights + BF16 compute (not recommended) |
| `--frame-delay N` | 0 | Simulated online delay (for Slippi) |
| `--stream-chunk-size N` | nil | Load N files at a time (memory-bounded) |
| `--gc-every N` | 100 | Run garbage collection every N batches (0=disabled) |

### Data Augmentation

| Option | Default | Description |
|--------|---------|-------------|
| `--augment` | false | Enable data augmentation |
| `--mirror-prob X` | 0.5 | Mirror augmentation probability |
| `--noise-prob X` | 0.3 | Noise augmentation probability |
| `--noise-scale X` | 0.01 | Noise magnitude |
| `--cache-augmented` | false | Precompute augmented variants (~100x speedup) |
| `--num-noisy-variants N` | 2 | Number of noisy variants to precompute |
| `--label-smoothing X` | 0.0 | Label smoothing (0.1 = typical) |
| `--focal-loss` | false | Enable focal loss for rare actions |
| `--focal-gamma X` | 2.0 | Focal loss gamma (higher = focus on hard) |

**Augmented Embedding Cache (Recommended)**

Use `--cache-augmented` to precompute augmented embedding variants for ~100x speedup:

```bash
# Fast augmented training (recommended)
mix run scripts/train_from_replays.exs \
  --augment --cache-augmented \
  --num-noisy-variants 2

# More variety with additional noisy variants
mix run scripts/train_from_replays.exs \
  --augment --cache-augmented \
  --num-noisy-variants 4
```

This precomputes multiple versions of each frame (original, mirrored, noisy variants) and randomly selects among them during training, providing similar regularization to on-the-fly augmentation.

> **Note:** Without `--cache-augmented`, `--augment` applies augmentation on-the-fly which is ~100x slower.
> See [Gotcha #40](GOTCHAS.md#40---augment-flag-bypasses-precomputed-embeddings-100x-slower) for details.

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
| `--quiet` | false | Minimal output (errors only), suppresses XLA/ptxas logs |
| `--verbose` | false | Debug output (timing, memory) |
| `--log-interval N` | 100 | Progress bar update frequency (every N batches) |
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
| `--val-concurrency N` | 4 | Parallel validation batches (1=sequential) |
| `--profile` | false | Enable timing profiler (report at end) |
| `--mmap-embeddings` | false | Use memory-mapped embeddings (for datasets > RAM) |
| `--mmap-path PATH` | nil | Custom path for mmap file (auto-generated if not set) |
| `--auto-batch-size` | false | Auto-tune batch size for optimal GPU utilization |
| `--auto-batch-min N` | 32 | Minimum batch size to test |
| `--auto-batch-max N` | 4096 | Maximum batch size to test |
| `--auto-batch-backoff X` | 0.8 | Safety factor (0.8 = 20% headroom) |

> **Note:** `--prefetch` only has effect when used with `--stream-chunk-size`. In standard (non-streaming) mode, prefetching is disabled due to EXLA tensor process limitations. A warning is shown if you use `--prefetch` without streaming mode.
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
| `--stage-mode MODE` | compact | Stage embedding: full, compact, learned |
| `--action-mode MODE` | learned | Action embedding: one_hot (399 dims) or learned (64-dim trainable) |
| `--character-mode MODE` | learned | Character embedding: one_hot (33 dims) or learned (64-dim trainable) |
| `--nana-mode MODE` | compact | Ice Climbers Nana: compact (39), enhanced (14+ID), full (449) |
| `--jumps-normalized` | true | Jumps as 1 normalized dim (false = 7-dim one-hot) |
| `--num-player-names N` | 112 | Player name dims (0 to disable) |
| `--learn-player-styles` | false | Enable style-conditional training |
| `--player-registry PATH` | nil | Save/load player registry JSON |
| `--min-player-games N` | 1 | Min games for player to be in registry |

**Embedding dimension reduction:**
Using learned embeddings dramatically reduces input dimensions:
- `--action-mode learned` saves ~670 dims (399×2 players → 2 action IDs)
- `--character-mode learned` saves ~64 dims (33×2 players → 2 char IDs)
- `--stage-mode compact` saves 57 dims (64 → 7)
- `--nana-mode enhanced` optimizes IC handling with action ID

**Note:** These optimized settings are now the **default**. Total embedding is 287 dims (vs 1204 with all one-hot). You only need to specify these flags if you want to change back to one-hot mode:

```bash
# Use one-hot mode (larger embeddings, slower training)
mix run scripts/train_from_replays.exs \
  --action-mode one_hot \
  --character-mode one_hot \
  --stage-mode full
```

### Player Style Learning

Style-conditional training learns player-specific playstyles by embedding player tags from replays. This allows the model to learn "how Plup plays Sheik" vs "how Jmook plays Sheik" - same character, different tendencies.

**Usage:**
```bash
# Enable style learning (builds registry from replay tags)
mix run scripts/train_from_replays.exs --learn-player-styles --temporal --backbone mamba

# Save registry for later use or inspection
mix run scripts/train_from_replays.exs --learn-player-styles --player-registry players.json

# Reuse saved registry (ensures consistent player IDs across runs)
mix run scripts/train_from_replays.exs --learn-player-styles --player-registry players.json
```

**How it works:**
1. Scans replay metadata for player tags (e.g., "Plup", "Jmook", "Mango")
2. Assigns each unique player a numeric ID (0 to num_player_names-1)
3. Embeds player ID as one-hot vector during training
4. Model learns player-specific tendencies through this conditioning

**Options:**
- `--min-player-games N` - Only include players with N+ games (filter rare tags)
- `--num-player-names N` - Max unique players (overflow uses hash bucketing)

**Use cases:**
- Train a model that can mimic specific player styles
- Analyze playstyle differences between players
- Condition generation: "play aggressively like Mango"

### K-means Stick Discretization

By default, stick positions are discretized into 17 uniform buckets (0-16). K-means clustering learns the actual distribution from replay data, placing more cluster centers where human inputs concentrate (cardinal/diagonal positions, specific angles for techniques).

**Benefits:**
- ~5% accuracy improvement on rare but important inputs (wavedash angles, shield drops)
- Better coverage of deadzone boundaries
- Character-specific patterns captured

**Usage:**
```bash
# Step 1: Train K-means centers from your replays
mix run scripts/train_kmeans.exs --replays ./replays --k 21 --output priv/kmeans_centers.nx

# Step 2: Use centers during training
mix run scripts/train_from_replays.exs --kmeans-centers priv/kmeans_centers.nx --temporal --backbone mamba
```

The default 21 clusters matches slippi-ai's research findings. You can also save to JSON for inspection:
```bash
cat priv/kmeans_centers.json
```

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

> **Why prefetch only works in streaming mode:** EXLA tensors cannot safely cross Erlang process boundaries. The streaming prefetcher uses a producer process that buffers Elixir data structures (not tensors), allowing true async batch preparation. In non-streaming mode, batches contain pre-computed EXLA tensors, so prefetching would require unsafe tensor transfers between processes.

**Automatic optimizations in streaming mode:**
- Precompute is auto-disabled (embeddings computed on-the-fly)
- This is intentional: precomputing per-chunk then discarding is wasteful
- On-the-fly embedding is faster for streaming since chunks aren't reused

**Recommended chunk sizes by RAM:**
- 32GB RAM: `--stream-chunk-size 15`
- 56GB RAM: `--stream-chunk-size 30`
- 128GB RAM: `--stream-chunk-size 60` or no streaming

## Memory Management

Training can consume significant RAM and GPU memory. This section covers how to prevent OOM errors and optimize memory usage.

### Quick Reference: When to Use Streaming

| Total Frames | RAM Needed (approx) | Recommendation |
|--------------|---------------------|----------------|
| < 200K | ~2 GB | Standard mode OK |
| 200K - 500K | 2-8 GB | Standard usually OK, streaming safer |
| 500K - 1M | 8-20 GB | Use `--stream-chunk-size 20000` |
| > 1M | 20+ GB | **Always** use `--stream-chunk-size 10000-20000` |

**Rule of thumb:** If the script shows "Total training frames: X" where X > 500K, add `--stream-chunk-size 20000` to avoid OOM during embedding.

The embedding step happens before training starts. If your terminal becomes unresponsive at "Embedding: N%" with memory at 100%, the dataset is too large for available RAM.

### Memory Usage Patterns

**Where memory goes during training:**
1. **Replay data** - Parsed frames loaded into RAM
2. **Embeddings** - Pre-computed state embeddings (if `--precompute`)
3. **Model parameters** - Network weights and optimizer state
4. **Batch tensors** - Current batch on GPU
5. **Gradient tensors** - Backprop intermediate values

### Symptoms of Memory Issues

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| SSH freezes, can't connect | System RAM exhausted | Use `--stream-chunk-size`, reduce `--max-files` |
| Training crashes with OOM | GPU VRAM exhausted | Reduce `--batch-size` (BF16 saves VRAM but is slower) |
| Training gets progressively slower | Memory leak / fragmentation | Use `--gc-every 50` |
| Swap usage increasing | RAM pressure | Use streaming mode, reduce batch size |

### Memory Optimization Flags

| Flag | Default | Effect |
|------|---------|--------|
| `--stream-chunk-size N` | nil | Load N files at a time instead of all |
| `--gc-every N` | 100 | Run garbage collection every N batches |
| `--batch-size N` | 64 | Smaller = less GPU memory |
| `--precision bf16` | f32 | Half precision uses ~50% less VRAM (but 2x slower) |
| `--no-precompute` | false | Compute embeddings on-the-fly (saves RAM) |

### Embedding Disk Cache

Embedding precomputation can take 1+ hours for large datasets. Enable disk caching to reuse embeddings across runs.

**Cache flags (consistent across all scripts):**

| Flag | Default | Effect |
|------|---------|--------|
| `--cache-embeddings` | false | Enable disk caching of embeddings |
| `--cache-augmented` | false | Cache augmented variants (original + mirrored + noisy) |
| `--num-noisy-variants N` | 2 | Number of noisy variants to cache |
| `--cache-dir PATH` | `cache/embeddings` | Cache directory |
| `--no-cache` | false | Ignore existing cache and recompute |

#### Training Script Examples

```bash
# Enable embedding cache (saves ~1 hour on re-runs)
mix run scripts/train_from_replays.exs \
  --replays /workspace/replays/mewtwo \
  --cache-embeddings \
  --temporal --backbone mlp

# Enable AUGMENTED embedding cache (~100x speedup for --augment)
mix run scripts/train_from_replays.exs \
  --replays /workspace/replays/mewtwo \
  --cache-augmented --augment \
  --num-noisy-variants 2

# Custom cache directory
mix run scripts/train_from_replays.exs \
  --replays /workspace/replays/mewtwo \
  --cache-embeddings \
  --cache-dir /workspace/cache/embeddings

# Force recompute even if cache exists
mix run scripts/train_from_replays.exs \
  --replays /workspace/replays/mewtwo \
  --cache-embeddings \
  --no-cache
```

**Cache Types:**
- `--cache-embeddings`: Caches single embedding per frame. Incompatible with `--augment`.
- `--cache-augmented`: Caches multiple variants per frame (original, mirrored, noisy). **Compatible with `--augment`** - this is the recommended way to use augmentation.

**Incompatible flags:**
- `--cache-embeddings` + `--augment`: Use `--cache-augmented` instead for fast augmentation
- `--cache-embeddings` + `--no-precompute`: Nothing to cache without precomputation
- `--cache-embeddings` + `--stream-chunk-size`: Streaming mode processes chunks on-the-fly, cache not used

#### Benchmark Script Examples

```bash
# Enable embedding cache
mix run scripts/benchmark_architectures.exs --replays /workspace/replays --cache-embeddings

# Custom cache directory
mix run scripts/benchmark_architectures.exs --replays /workspace/replays --cache-embeddings --cache-dir /workspace/my_cache

# Force recompute even if cache exists
mix run scripts/benchmark_architectures.exs --replays /workspace/replays --cache-embeddings --no-cache
```

**Cache key is based on:**
- Replay file list (sorted paths)
- Embedding config (action_mode, character_mode, stage_mode, etc.)
- Window size and stride (for temporal embeddings)

**Managing cache:**
```elixir
# List cached embeddings
ExPhil.Training.EmbeddingCache.list()

# Clear all cached embeddings
ExPhil.Training.EmbeddingCache.clear()

# Invalidate specific cache
ExPhil.Training.EmbeddingCache.invalidate("cache_key_here")
```

### Recommended Configurations by Hardware

**RunPod RTX 4090 (24GB VRAM, ~50GB RAM):**
```bash
mix run scripts/train_from_replays.exs \
  --temporal --backbone mamba \
  --hidden-sizes 512,256 \
  --stream-chunk-size 20000 \
  --gc-every 50 \
  --batch-size 512 \
  --seq-len 64 \
  --save-best \
  --name mamba_training \
  2>&1 | tee /workspace/logs/training_$(date +%Y%m%d_%H%M%S).log
```

**Local GPU (8-12GB VRAM):**
```bash
mix run scripts/train_from_replays.exs \
  --temporal --backbone mamba \
  --stream-chunk-size 15 \
  --gc-every 50 \
  --batch-size 32
# Add --precision bf16 if VRAM constrained (slower but uses 50% less memory)
```

**CPU-only (32GB RAM):**
```bash
mix run scripts/train_from_replays.exs \
  --backbone mlp \
  --stream-chunk-size 10 \
  --gc-every 100 \
  --batch-size 128 \
  --no-precompute
```

### Monitoring Memory During Training

**System RAM:**
```bash
# Watch memory usage
watch -n 1 'free -h'

# Or use htop
htop
```

**GPU Memory (NVIDIA):**
```bash
# Watch GPU memory
watch -n 1 'nvidia-smi'

# Or continuous monitoring
nvidia-smi -l 1
```

**From Elixir (in IEx):**
```elixir
# System memory
:erlang.memory(:total) / 1_000_000  # MB

# Force garbage collection
:erlang.garbage_collect()
```

### Troubleshooting Memory Issues

**If SSH freezes during training:**
1. The pod is likely out of system RAM
2. Use RunPod web terminal (Connect → Web Terminal) instead
3. For future runs, use `--stream-chunk-size` to bound memory

**If training crashes with CUDA OOM:**
1. Reduce `--batch-size` (try halving it)
2. Try `--precision bf16` (uses 50% less VRAM, but 2x slower due to XLA issues)
3. For temporal models, reduce `--window-size`

**If memory grows over time:**
1. Enable `--gc-every 50` for more frequent garbage collection
2. This is normal for long runs - Erlang's GC is generational
3. The `--gc-every` flag forces full collection periodically

### Technical Details

**Why streaming helps:**
- Without streaming: All replay files are parsed and loaded into RAM before training starts
- With streaming: Files are loaded in chunks, processed, then freed

**Why `--gc-every` helps:**
- BEAM's garbage collector is per-process and generational
- Long-running training can accumulate garbage in older generations
- Periodic `:erlang.garbage_collect()` forces full collection
- Default of 100 batches balances memory vs overhead (~1ms per GC)

**Prefetcher memory fix (v6911c16):**
- Previous versions materialized ALL batches into memory before training
- Now uses lazy streaming - only current batch is in memory
- This alone can reduce RAM usage by 50%+ for large datasets

## Performance Tips

### GPU Training Optimization

**See [GPU_OPTIMIZATIONS.md](GPU_OPTIMIZATIONS.md) for comprehensive GPU guide.**

Quick wins for GPU training:

```bash
# Maximum speed - larger batch with prefetch
mix run scripts/train_from_replays.exs \
  --batch-size 512 \
  --prefetch

# If GPU memory limited, use gradient accumulation
mix run scripts/train_from_replays.exs \
  --batch-size 256 \
  --accumulation-steps 2 \
  --prefetch
```

| Optimization | Flag | Impact |
|--------------|------|--------|
| Larger batch | `--batch-size 512` | Better GPU utilization |
| Prefetching | `--prefetch` | Overlap data/compute |
| Grad accumulation | `--accumulation-steps 4` | Effective larger batch |

> **Note on BF16**: Benchmarks show BF16 is actually 2x SLOWER than FP32 on RTX 4090 due to XLA issues
> (dimension misalignment, type casting overhead). FP32 is the default and recommended.

**Training time (GPU):**

| Dataset Size | Epochs | Time (RTX 4090) |
|--------------|--------|-----------------|
| 100 files | 10 | ~30 min |
| 1000 files | 10 | ~3-4 hours |
| Full dataset | 10 | ~8-10 hours |

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

## Self-Play Training

Train multiple agents against each other with Elo tracking:

```bash
# Basic self-play (4 parallel games)
mix run scripts/train_self_play.exs \
  --num-games 4 \
  --track-elo

# With custom episode length (default: 28800 frames = ~8 min)
mix run scripts/train_self_play.exs \
  --num-games 4 \
  --max-episode-frames 18000 \
  --track-elo
```

### Self-Play Options

| Option | Default | Description |
|--------|---------|-------------|
| `--num-games N` | 4 | Number of parallel games to run |
| `--game-type TYPE` | mock | Game type: mock, dolphin, libmelee |
| `--max-episode-frames N` | 28800 | Max frames per episode (~8 min at 60fps) |
| `--track-elo` | false | Enable Elo rating tracking |
| `--ppo-epochs N` | 4 | PPO update epochs per batch |
| `--clip-epsilon F` | 0.2 | PPO clipping parameter |
| `--gae-lambda F` | 0.95 | GAE lambda for advantage estimation |

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

### Batch-Interval Checkpointing

For streaming mode or long epochs, save checkpoints every N batches:

```bash
# Save every 500 batches (recommended for streaming mode)
mix run scripts/train_from_replays.exs --stream-chunk-size 30 --save-every-batches 500

# More frequent saves for large datasets
mix run scripts/train_from_replays.exs --save-every-batches 200
```

This creates `checkpoints/{name}_batch.axon` that gets overwritten at each interval.
Protects against losing progress if training crashes or is interrupted.

**When to use:**
- Streaming mode (epochs can be very long)
- Training on large datasets (>1000 batches per epoch)
- Unreliable GPU/cloud instances (preemptible VMs)

### Graceful Shutdown (Ctrl+C)

Training automatically handles SIGTERM and SIGINT (Ctrl+C) signals:

- Pressing Ctrl+C during training saves a checkpoint before exiting
- Checkpoint saved to `checkpoints/{name}_interrupt.axon`
- Resume interrupted training: `--resume checkpoints/{name}_interrupt.axon`

```bash
# Example: interrupt during training
# Press Ctrl+C, see:
#   ⚠ Received sigint - saving checkpoint before exit...
#   Saving trainer state (epoch 3, batch 450)...
#   ✓ Interrupt checkpoint saved to checkpoints/mewtwo_interrupt.axon
#   Resume with: --resume checkpoints/mewtwo_interrupt.axon

# Then resume:
mix run scripts/train_from_replays.exs --resume checkpoints/mewtwo_interrupt.axon
```

The trainer state is updated every 10 batches, so at most ~10 batches of work may be lost on interrupt.

### Config JSON Contents

Each checkpoint produces a `*_config.json` file with complete training provenance:

```json
{
  "timestamp": "2026-01-24T17:00:00Z",

  "// Training parameters": "",
  "backbone": "mamba",
  "epochs": 10,
  "batch_size": 64,
  "hidden_sizes": [256, 256],
  "temporal": true,
  "window_size": 30,

  "// Data filtering (what was trained on)": "",
  "characters": ["mewtwo"],
  "stages": null,
  "replays_dir": "/path/to/replays",
  "max_files": 100,

  "// Replay manifest (provenance)": "",
  "replay_count": 100,
  "replay_files": ["game1.slp", "game2.slp", "..."],
  "replay_manifest_hash": "sha256:abc123...",
  "character_distribution": {"mewtwo": 80000, "fox": 20000},

  "// Results": "",
  "training_frames": 125000,
  "validation_frames": 13000,
  "final_training_loss": 3.68,
  "total_time_seconds": 1847
}
```

**Key provenance fields:**

| Field | Description |
|-------|-------------|
| `characters` | Character filter used (`--characters mewtwo,fox`), null if unfiltered |
| `stages` | Stage filter used (`--stages battlefield,fd`), null if unfiltered |
| `replay_count` | Number of replay files used |
| `replay_files` | Actual file paths (if ≤500 files), for reproducibility |
| `replay_manifest_hash` | SHA256 of sorted file list, for deduplication |
| `character_distribution` | Frame counts per character in training data |

**Use cases:**
- Know exactly which replays trained a model
- Verify two models used the same data (compare hashes)
- Understand character composition of training data
- Reproduce training with same data

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
| `--quiet` | 0 | Errors only, suppresses warnings and XLA/ptxas logs |
| (default) | 1 | Normal output with progress bars |
| `--verbose` | 2 | Debug info: timing, memory, gradients |

**Progress Bar Frequency:**
```bash
# Default is 100 batches between updates (keeps logs readable)
# For more frequent updates during debugging:
mix run scripts/train_from_replays.exs --log-interval 10

# For minimal log output (updates ~5 times per epoch on large datasets):
mix run scripts/train_from_replays.exs --log-interval 1000 --preset standard
```

**Examples:**
```bash
# Quiet mode for CI/scripted runs (suppresses XLA warnings like ptxas register spills)
mix run scripts/train_from_replays.exs --quiet --preset quick

# Verbose mode for debugging
mix run scripts/train_from_replays.exs --verbose --preset quick
```

**What `--quiet` suppresses:**
- Progress bar output (use `--log-interval N` for reduced updates)
- Warnings from Output module
- XLA/EXLA info logs (ptxas register spills, JIT compilation notices)
- Logger `:info` level messages

**Verbose output includes:**
- Per-batch timing breakdown
- GPU memory usage after each epoch
- Gradient norm statistics
- Data loading vs training time
- Cache hit rates
- Debug messages marked with `[DEBUG]`

### Progress Intervals (Programmatic)

Data processing functions accept a `:progress_interval` option to control update frequency:

```elixir
# Default: update every 10 batches
Data.precompute_frame_embeddings(dataset, show_progress: true)

# Custom: update every 50 batches (less log spam)
Data.precompute_frame_embeddings(dataset,
  show_progress: true,
  progress_interval: 50
)
```

| Function | Default Interval | Unit |
|----------|-----------------|------|
| `precompute_frame_embeddings` | 10 | batches (1000 frames each) |
| `precompute_embeddings` | 10 | chunks |
| `precompute_augmented_frame_embeddings` | 10 | batches |
| `sequences_from_frame_embeddings` | 50,000 | sequences |

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
