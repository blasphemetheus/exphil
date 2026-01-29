# Training Improvements Roadmap

Ideas for improving training speed, effectiveness, and developer experience. Prioritized by impact and effort.

## Quick Wins (Easy, High Impact)

### Larger Batch Sizes on GPU
- **Current:** 32-64 (CPU-oriented defaults)
- **Proposed:** 256-512 on GPU
- **Benefit:** ~1.5x speedup, better gradient estimates
- **Implementation:** Already supported via `--batch-size`, just increase in GPU presets

### Optimizer Selection ✅
- **Status:** IMPLEMENTED (2025-01-20)
- **Usage:** `--optimizer adamw` (or adam, lamb, radam, sgd, rmsprop)
- **Note:** XLA automatically fuses operations, so there's no separate "fused" variant
- **Recommendations:**
  - `adamw` (default) - Decoupled weight decay, good general choice
  - `lamb` - Good for large batch training
  - `radam` - More stable early training

### GPU Memory Tracking
- **Current:** No visibility into VRAM usage
- **Proposed:** Log VRAM at epoch start/end
- **Benefit:** Know when to increase batch size or if OOM is imminent
- **Implementation:**
  ```elixir
  # Add to training loop
  {:ok, mem} = EXLA.Client.get_memory_info(:cuda, 0)
  Output.puts("  GPU Memory: #{mem.used / 1_000_000_000:.1f}GB / #{mem.total / 1_000_000_000:.1f}GB")
  ```

## Medium Effort, High Impact

### Precompute Embeddings
- **Current:** Embeddings computed every batch, every epoch
- **Proposed:** Embed all frames once, train on dense tensors
- **Benefit:** 2-3x speedup (embedding is ~40% of forward pass)
- **Trade-off:** More memory usage (store embedded dataset)
- **Implementation:**
  ```elixir
  # Before training loop
  embedded_dataset = dataset
  |> Enum.map(fn {state, action} ->
    {Embeddings.embed_game_state(state), action}
  end)
  |> Nx.stack()  # Single tensor for fast batching

  # Training uses pre-embedded data
  # Model starts from dense layer, not embedding
  ```

### Data Prefetching
- **Current:** Synchronous batch loading
- **Proposed:** Load next batch while GPU trains on current
- **Benefit:** ~1.2x speedup (hide data loading latency)
- **Implementation:** Use `Task.async` or `Flow` for pipeline parallelism

### Training Loss Plot
- **Current:** Loss only in terminal output
- **Proposed:** Save loss curve PNG at training end
- **Benefit:** Quick visual check for convergence issues
- **Implementation:**
  ```elixir
  # At end of training
  VegaLite.new()
  |> VegaLite.data_from_values(loss_history)
  |> VegaLite.mark(:line)
  |> VegaLite.encode_field(:x, "epoch", type: :quantitative)
  |> VegaLite.encode_field(:y, "loss", type: :quantitative)
  |> VegaLite.Export.save!("#{checkpoint_dir}/loss_curve.png")
  ```
  Requires adding `{:vega_lite, "~> 0.1"}` to deps.

### Auto-Resume from Checkpoint
- **Current:** Manual `--resume` flag required
- **Proposed:** Detect incomplete training and offer to resume
- **Benefit:** Don't lose progress from crashes/disconnects
- **Implementation:**
  ```elixir
  # At startup, check for .incomplete marker
  incomplete_marker = "#{opts[:checkpoint]}.incomplete"
  if File.exists?(incomplete_marker) do
    Output.puts("⚠️  Found incomplete training. Resume? [Y/n]")
    # Load last epoch checkpoint and continue
  end
  # Create marker at start, delete on successful completion
  ```

## Higher Effort, High Impact

### Residual MLP
- **Current:** Plain feedforward MLP
- **Proposed:** Add skip connections between layers
- **Benefit:** Better gradient flow, can go deeper without degradation
- **Implementation:**
  ```elixir
  def residual_block(input, hidden_size) do
    input
    |> Axon.dense(hidden_size, activation: :relu)
    |> Axon.dense(hidden_size)  # No activation before residual
    |> Axon.add(input)  # Skip connection
    |> Axon.activation(:relu)
  end
  ```

### Focal Loss for Rare Actions
- **Current:** Cross-entropy treats all actions equally
- **Proposed:** Focal loss down-weights easy examples
- **Benefit:** Better learning of rare actions (Z, D-pad)
- **Implementation:**
  ```elixir
  def focal_loss(logits, targets, gamma \\ 2.0) do
    probs = Nx.sigmoid(logits)
    ce_loss = Nx.binary_cross_entropy(probs, targets)
    focal_weight = Nx.pow(1 - probs, gamma)
    Nx.mean(focal_weight * ce_loss)
  end
  ```

### Layer Normalization ✅
- **Status:** IMPLEMENTED (2025-01-20)
- **Usage:** `--layer-norm` flag (or `layer_norm: true` in config)
- **Benefit:** More stable training, especially with larger LR
- **Implementation:** Adds `Axon.layer_norm()` after each dense layer in backbone

### Dropout Scheduling
- **Current:** Fixed dropout rate
- **Proposed:** Start at 0.3, anneal to 0.0 over training
- **Benefit:** Strong regularization early, full capacity late
- **Implementation:**
  ```elixir
  dropout_rate = 0.3 * (1 - epoch / total_epochs)
  # Requires dynamic dropout in model (tricky with Axon's static graph)
  ```

## For GPU Testing & Experimentation

### Architecture Benchmark Script (NEW)
- **Script:** `scripts/benchmark_architectures.exs`
- **Usage:** `mix run scripts/benchmark_architectures.exs --replay-dir /path --epochs 3`
- **Output:** Comparison table + HTML report with loss curves
- **Benefit:** Quickly test which architecture works best on your data

### Per-Action Accuracy Metrics (NEW)
- **Module:** `ExPhil.Training.Metrics`
- **What it shows:**
  - Per-button accuracy (A, B, X, Y, Z, L, R, Start)
  - Per-stick MSE (main_x, main_y, c_x, c_y)
  - Rare action recall (Z, L, R - important for tech skill)
- **Why:** Loss alone hides whether rare actions are learned

### Hyperparameter Sweep (TODO)
- Run multiple configs automatically
- Track results in JSON/CSV
- Generate comparison report
- **Implementation:** Similar to benchmark script but varies hyperparams

### Action Distribution Analysis (TODO)
- Visualize predicted vs actual action distributions
- Confusion matrix for buttons
- Stick position heatmaps
- **Why:** Understand systematic prediction errors

## Nice to Have (Lower Priority)

### Batch Progress Bar ✅
- **Status:** IMPLEMENTED (2025-01-20)
- Shows live updating progress within each epoch
- Format: `Epoch 1: ████████████░░░░░░░░  60% | 642/1606 | loss: 0.1234 | 0.5s/it | ETA: 8m 12s`
- Updates in-place using carriage return for cleaner output

### Checkpoint Compression
- Compress checkpoint files with `:zlib`
- Reduces storage for many checkpoints

### Training Metrics Dashboard
- Real-time web UI showing loss, LR, GPU stats
- Could use Phoenix LiveView or Livebook integration

### Gradient Noise Injection
- Add small noise to gradients during training
- Can help escape local minima

### Stochastic Weight Averaging (SWA)
- Average weights over last N epochs
- Often improves generalization

---

## Implementation Priority

Based on impact/effort ratio:

1. **Larger batch sizes** - Just change preset defaults
2. **GPU memory tracking** - Few lines of code
3. **Precompute embeddings** - Biggest speedup, medium effort
4. **Training loss plot** - Nice for debugging
5. **Auto-resume** - Important for long runs on spot instances
6. **Residual MLP** - Try if hitting accuracy ceiling
7. **Focal loss** - Try if rare actions aren't learned

## Performance Optimizations (2026-01)

Optimizations identified for reducing per-epoch overhead on large datasets (1.6M+ frames).

### High Impact

#### Cache JIT-compiled loss function in validation ✅
- **Status:** COMPLETED (2026-01-29)
- **Current:** `build_loss_fn` called every `evaluate()` call, may re-JIT
- **Solution:** Added `eval_loss_fn` field to trainer struct, built once in `new/1`
- **Benefit:** Avoid re-JIT overhead during validation (~5-10s savings)
- **Implementation:** `build_eval_loss_fn/2` creates JIT-compiled loss function, reused in `evaluate/3`

#### Lazy index shuffling ✅
- **Status:** COMPLETED (2026-01-29)
- **Current:** `Enum.shuffle(0..1_600_000)` takes ~2s every epoch
- **Solution:** Chunked lazy shuffle for datasets >100K - shuffle within 10K chunks, shuffle chunk order
- **Benefit:** ~2s savings per epoch, O(10K) memory instead of O(n)
- **Implementation:** `lazy_shuffled_batches/5` and `lazy_shuffled_frame_batches/7` in Data module

#### Parallel validation batches ✅
- **Status:** COMPLETED (2026-01-29)
- **Usage:** `--val-concurrency N` (default: 4, set to 1 for sequential)
- **Implementation:** `Task.async_stream` in `Imitation.evaluate/3` for parallel batch processing
- **Benefit:** ~2-3x faster validation if GPU memory allows concurrent batches
- **Caveat:** May increase GPU memory pressure; reduce `--val-concurrency` if OOM

#### Pre-embed validation data ✅
- **Status:** COMPLETED (2026-01-29)
- **Implementation:** Validation batches materialized once before training loop, reused every epoch
- **Benefit:** ~2-5s savings per epoch by avoiding batch recreation
- **Note:** Works automatically when not in streaming mode

### Medium Impact

#### Reduce Enum.to_list calls
- **Status:** TODO
- **Current:** Several places convert ranges to lists unnecessarily
- **Proposed:** Use Streams where possible
- **Benefit:** Lower memory allocation overhead

#### Batch tensor stacking optimization ✅
- **Status:** COMPLETED (2026-01-29)
- **Current:** In `evaluate/3`, losses collected as list then stacked
- **Solution:** Use running sum tensor `Nx.add(acc_loss, loss)` instead of list accumulation
- **Benefit:** Avoid final Nx.stack allocation, single GPU→CPU transfer

#### Skip validation on non-improvement epochs
- **Status:** TODO
- **Current:** Full validation every epoch
- **Proposed:** After N epochs without improvement, run validation less frequently
- **Benefit:** Save validation time when model is plateaued

### Lower Priority

#### ONNX inference for validation
- Use quantized INT8 model for approximate validation loss
- Much faster but introduces approximation error

#### Custom CUDA kernels
- Already have flash attention NIF
- Could add custom kernels for specific bottlenecks

#### Gradient checkpointing tuning
- Trade compute for memory more aggressively
- Already supported via `--gradient-checkpoint`

### Profiler for Performance Analysis ✅
- **Status:** COMPLETED (2026-01-29)
- **Module:** `ExPhil.Training.Profiler`
- **Usage:** `mix run scripts/train_from_replays.exs --profile --epochs 1`
- **Features:**
  - Times key operations: batch_prep, forward, backward, optimizer, validation, checkpoint
  - Tracks min/max/avg/total for each phase
  - Generates formatted report at training end
  - Stores last 1000 samples for percentile calculations
- **Implementation:**
  ```elixir
  # Wrap operations to time them
  Profiler.time(:batch_prep, fn -> prepare_batch() end)
  Profiler.time(:forward, fn -> forward_pass() end)

  # Get report at end
  Profiler.print_report()
  ```
- **Report format:**
  ```
  ╔══════════════════════════════════════════════════════════════════════════════╗
  ║                           TRAINING PROFILE REPORT                            ║
  ╠══════════════════════════════════════════════════════════════════════════════╣
  ║ Phase              │ Count   │ Total (s) │ Avg (ms) │ Min (ms) │ Max (ms) │ % ║
  ╟────────────────────┼─────────┼───────────┼──────────┼──────────┼──────────┼───╢
  ║ forward            │    1606 │     45.23 │    28.16 │    25.12 │    42.31 │ 45║
  ║ backward           │    1606 │     38.92 │    24.23 │    22.01 │    35.18 │ 39║
  ╚══════════════════════════════════════════════════════════════════════════════╝
  ```

### Future Optimizations (TODO)

#### Double-buffered batch prefetching ✅
- **Status:** COMPLETED (2026-01-29)
- **Usage:** Enabled by default. Disable with `--no-prefetch`
- **How it works:** Uses `Prefetcher.reduce_stream_indexed` with a producer process that buffers batches while GPU trains
- **Benefit:** Hide batch prep latency (~50-100ms per batch), ~10-20% speedup
- **Options:**
  - `--prefetch-buffer N` - Number of batches to prefetch (default: 2)
  - `--no-prefetch` - Disable prefetching
- **Note:** Previously only worked in streaming mode, now enabled for standard mode too

#### Memory-mapped embedding cache ✅
- **Status:** COMPLETED (2026-01-29)
- **Module:** `ExPhil.Training.MmapEmbeddings`
- **Usage:**
  ```elixir
  # Save embeddings to disk
  MmapEmbeddings.save(embeddings_tensor, "embeddings.bin")

  # Open for reading (doesn't load into RAM)
  {:ok, handle} = MmapEmbeddings.open("embeddings.bin")

  # Read batches on-demand
  batch = MmapEmbeddings.read_batch(handle, [0, 5, 10, 15])

  MmapEmbeddings.close(handle)
  ```
- **File format:** Custom binary with 32-byte header (magic, version, dims, dtype)
- **Benefit:** Train on datasets larger than RAM
- **Performance:** ~1-2ms per batch read (SSD), ~10-20ms (HDD)

#### Batch size auto-tuning ✅
- **Status:** COMPLETED (2026-01-29)
- **Module:** `ExPhil.Training.BatchTuner`
- **Usage:**
  ```elixir
  # Find optimal batch size by testing
  {:ok, optimal} = BatchTuner.find_optimal(trainer, sample_states, sample_actions)

  # Quick heuristic based on VRAM
  suggested = BatchTuner.suggest(embed_size, hidden_sizes, vram_gb: 24)
  ```
- **How it works:**
  1. Start with small batch (32)
  2. Double until OOM
  3. Back off 20% for safety
  4. Round to power of 2
- **Benefit:** Automatic optimal GPU utilization

#### Async checkpoint saving ✅
- **Status:** COMPLETED (2026-01-29)
- **Usage:** Batch and epoch checkpoints now use `save_checkpoint_async/3`
- **Benefit:** ~1-5s savings per checkpoint (doesn't block training)

#### Reduce Enum.to_list calls ✅
- **Status:** COMPLETED (2026-01-29)
- **Changes:**
  - Use Range directly with Stream.chunk_every (no list allocation)
  - Added `maybe_drop_last_stream/3` for lazy pipelines
- **Benefit:** Lower memory allocation for large datasets

#### Adaptive time display ✅
- **Status:** COMPLETED (2026-01-29)
- **Change:** Progress bar now shows ms/it for fast training, s/it for slow
- **Example:** "5.2ms/it" instead of "0.01s/it"

### Future Optimizations (TODO)

#### Validation sampling
- **Status:** TODO
- **Current:** Full validation every epoch
- **Proposed:** Randomly sample 10-20% of validation set per epoch, full validation every N epochs
- **Benefit:** ~5-10x faster validation feedback
- **Trade-off:** Noisier loss estimates (but converges to same result)
- **Implementation:**
  ```elixir
  # Quick validation: sample 10%
  sample_indices = Enum.take_random(0..val_size, div(val_size, 10))
  quick_val_loss = evaluate_subset(trainer, val_batches, sample_indices)

  # Full validation every 5 epochs
  if rem(epoch, 5) == 0 do
    full_val_loss = evaluate(trainer, val_batches)
  end
  ```

#### Warm restart JIT cache
- **Status:** TODO
- **Current:** JIT compilation happens fresh each run (~2-5 minutes)
- **Proposed:** Save compiled XLA functions to disk, reload on resume
- **Benefit:** Skip JIT on resume, instant training start
- **Blocker:** EXLA doesn't expose compiled function serialization
- **Workaround:** Could save a "warmed up" checkpoint after first batch

#### CPU/GPU pipelining for embeddings
- **Status:** TODO
- **Current:** All embedding lookups happen on GPU
- **Proposed:** Embedding lookup on CPU, transfer to GPU while previous batch trains
- **Benefit:** Better CPU utilization, overlap CPU/GPU work
- **Trade-off:** More complex pipeline, PCIe bandwidth becomes bottleneck
- **Note:** May not be beneficial with pre-computed embeddings

#### Fused optimizer kernels
- **Status:** TODO
- **Current:** Optimizer update is multiple separate operations
- **Proposed:** Single CUDA kernel for AdamW update (momentum + weight decay + LR in one pass)
- **Benefit:** Fewer kernel launches, better memory locality
- **Implementation:** Custom NIF using CUDA, similar to flash_attention_nif

#### Gradient accumulation micro-optimization
- **Status:** TODO
- **Current:** `--gradient-accumulation N` accumulates over N full batches
- **Proposed:** Auto-compute micro-batch size based on VRAM, accumulate transparently
- **Benefit:** Simpler UX - just specify effective batch size, let system figure out micro-batches

#### Mixed precision training
- **Status:** TODO (partially supported)
- **Current:** Training uses FP32 (BF16 has XLA issues)
- **Proposed:** Forward pass in FP16/BF16, gradients in FP32
- **Benefit:** 2x faster forward pass, half memory usage
- **Blocker:** EXLA/XLA BF16 matmul performance issues on some GPUs

#### Skip validation on plateau
- **Status:** TODO
- **Current:** Full validation every epoch
- **Proposed:** After N epochs without improvement, run validation less frequently
- **Benefit:** Save validation time when model is plateaued
- **Implementation:**
  ```elixir
  val_interval = if epochs_without_improvement > 5, do: 3, else: 1
  if rem(epoch, val_interval) == 0, do: run_validation()
  ```

### Completed (2026-01-29)

- [x] Cache array conversions with `prepare_for_batching/1`
- [x] Cache character weights to avoid recomputing every epoch
- [x] Add validation progress indicator
- [x] Add batch preparation timing visibility
- [x] Central time timestamps with tz library
- [x] Cache JIT-compiled eval loss function in trainer struct
- [x] Lazy chunked shuffle for large datasets (>100K samples)
- [x] Running sum for validation loss (avoid Nx.stack allocation)
- [x] Fix Data.split to maintain correspondence with embedded_sequences/embedded_frames
- [x] Profile-guided optimization infrastructure (Profiler module)
- [x] Parallel validation batches with `Task.async_stream` (--val-concurrency flag)
- [x] Pre-compute validation batches once before training loop
- [x] Enable prefetching for standard (non-streaming) mode
- [x] Memory-mapped embedding cache (MmapEmbeddings module)
- [x] Batch size auto-tuning (BatchTuner module)
- [x] Async checkpoint saving for batch/epoch checkpoints
- [x] Reduce Enum.to_list calls with lazy streams
- [x] Adaptive time display (ms/it for fast training)

## Completed

- [x] Timestamps in training output (2025-01-20)
- [x] Early stopping
- [x] Learning rate scheduling (cosine, warmup, restarts)
- [x] Gradient accumulation
- [x] Label smoothing
- [x] Data augmentation (mirror, noise)
- [x] Model EMA
- [x] Checkpoint pruning
- [x] Frame delay augmentation
- [x] Precompute embeddings for MLP training (2025-01-20)
- [x] GPU memory tracking via nvidia-smi (2025-01-20)
- [x] Config diff display at training start (2025-01-20)
- [x] Training loss plot with VegaLite (2025-01-20)
- [x] Auto-resume from incomplete training (2025-01-20)
- [x] Architecture test presets (gpu_mlp_quick, gpu_lstm_quick, etc.) (2025-01-20)
- [x] Data prefetching for GPU training (2025-01-20)
- [x] Layer normalization option for MLP backbone (2025-01-20)
- [x] Optimizer selection (adam, adamw, lamb, radam, sgd, rmsprop) (2025-01-20)
- [x] Live batch progress bar within epochs (2025-01-20)
