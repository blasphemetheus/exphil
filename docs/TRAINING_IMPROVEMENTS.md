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
