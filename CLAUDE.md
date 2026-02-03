# ExPhil - Elixir Phil

ExPhil is an Elixir-based successor to slippi-ai, creating high-ELO playable bots for lower-tier Melee characters (Mewtwo, G&W, Link, Ganondorf, Zelda, Ice Climbers).

## Quick Reference

**Embedding dimensions:**
- **Default (learned)**: 288 dims (287 raw + 1 padding for tensor core alignment)
- Player: 488 dims (base 440 + speeds 5 + frame_info 2 + stock 1 + ledge_dist 1 + compact Nana 39)
- Player (learned actions): 91 dims (base 41 + speeds 5 + frame_info 2 + stock 1 + ledge_dist 1 + compact Nana 39 + action IDs in network)
- Player (learned actions + chars): 58 dims (excludes 33-dim character one-hot)
- Game (one_hot, legacy): 1208 dims (1204 raw + 4 padding for alignment)
- Game (learned actions): 408 dims + 2 action IDs
- Game (learned actions + chars): 344 dims + 4 IDs (saves 64 dims net: 2×33 one-hot → 2 char IDs)
- **Note:** All embedding sizes are auto-aligned to multiples of 8 for GPU tensor core efficiency
- **512-dim target**: 254 continuous + 4 action IDs × 64 embedding (use `action_mode: :learned, nana_mode: :enhanced`)
- Nana modes:
  - `:compact` (default): 39 dims (preserves IC tech: handoffs, regrabs, desyncs)
  - `:enhanced`: 14 dims + Nana action ID (most efficient for learned embeddings, precise IC tech)
  - `:full`: 449 dims (complete action state, use `nana_mode: :full`)
- Jumps: Normalized (1 dim) by default; use `jumps_normalized: false` for 7-dim one-hot
- Controller: 13 dims (8 buttons + 4 sticks + 1 shoulder)
- Stick discretization: 17 uniform buckets (default) or 21 K-means clusters (`--kmeans-centers`)
- Action modes: `:one_hot` (399 dims) or `:learned` (64-dim trainable embedding, **default**, saves ~670 dims)
- Character modes: `:one_hot` (33 dims) or `:learned` (64-dim trainable embedding, **default**, saves 64 dims)
- Stage modes:
  - `:one_hot_full`: 64 dims (full stage one-hot)
  - `:one_hot_compact` (**default**): 7 dims (6 competitive stages + "other", saves 57 dims)
  - `:learned`: 1 ID + trainable embedding (saves 63 dims)
- Competitive stages for compact mode: FoD (2), PS (3), YS (8), DL (28), BF (31), FD (32)
- Action IDs: 2 (players only) or 4 (players + Nana with `nana_mode: :enhanced`)
- Character IDs: 2 when `character_mode: :learned` (own + opponent)
- Stage IDs: 1 when `stage_mode: :learned`

**Policy network:** 6-head autoregressive (buttons, main_x/y, c_x/y, shoulder)

**Inference benchmarks:**
| Architecture | Inference | 60 FPS Ready |
|--------------|-----------|--------------|
| LSTM (Axon) | 220.95 ms | No |
| Mamba (Axon) | 8.93 ms | **Yes** |
| ONNX INT8 | 0.55 ms | **Yes** |

## Current Status

**Test coverage:** 1933 tests passing

**Completed:**
- Stage embedding modes (full 64-dim, compact 7-dim, or learned embedding)
- Learned character embeddings (33 chars → 64-dim trainable, saves 64 dims)
- Imitation learning (single-frame + temporal)
- All backbones: MLP, LSTM, GRU, Mamba, attention
- PPO trainer with clipped objective
- Dolphin integration (sync + async runners)
- Training features: early stopping, LR scheduling, gradient accumulation, validation split, checkpointing
- Model EMA, cosine annealing with warm restarts, gradient clipping
- Learning rate finder (`scripts/find_lr.exs`)
- Data augmentation (mirror states, noise injection)
- Label smoothing for better generalization
- Model registry with JSON tracking and lineage
- Checkpoint pruning (keep best N)
- Frame delay augmentation for online play (`--online-robust`)
- Training presets (quick, standard, production, character-specific)
- K-means stick discretization (`scripts/train_kmeans.exs`, `--kmeans-centers`)
- Model evaluation script (`scripts/eval_model.exs`)
- Interactive setup wizard (`mix exphil.setup`)
- Environment variable support (`EXPHIL_REPLAYS_DIR`, `EXPHIL_WANDB_PROJECT`)
- Verbosity control (`--verbose` / `--quiet`)
- Reproducibility seed logging (`--seed N`)
- Checkpoint collision warnings and auto-backup (`--overwrite`, `--backup`)

## Immediate Priorities

See [GOALS.md](docs/GOALS.md) for comprehensive roadmap. Current focus:

**Completed:**
- Projectile parsing (1204-dim embedding includes projectiles)
- Focal loss (`--focal-loss --focal-gamma 2.0`)
- Embedding caching (2-3x speedup, default)
- Augmented embedding cache (`--cache-augmented`, ~100x speedup for `--augment`)
- Self-play GenServer infrastructure
- Precision benchmarking (FP32 now default - 2x faster than BF16 due to XLA issues)

**Next Phase:**
1. **Self-play Dolphin testing** - Run with real MeleePort games
2. **Large-scale self-play training** - GPU cluster deployment
3. **Character-specific rewards** - Mewtwo recovery, Ganon spacing, etc.

## Documentation

Documentation is organized into categories in `docs/`:

```
docs/
├── guides/       # How-to (training, inference, scripts)
├── reference/    # Technical specs (architecture, embeddings)
├── operations/   # DevOps (docker, runpod, cloud)
├── research/     # Research notes, case studies
├── planning/     # Project tracking (goals, fixes, roadmaps)
└── internals/    # Deep implementation (GPU, kernels)
```

**Key docs by category:**

| Category | Doc | Contents |
|----------|-----|----------|
| **planning** | [GOALS.md](docs/planning/GOALS.md) | Big picture goals, priorities |
| **planning** | [FIXES.md](docs/planning/FIXES.md) | Technical debt tracking |
| **guides** | [TRAINING.md](docs/guides/TRAINING.md) | Training commands, CLI options |
| **guides** | [SCRIPTS.md](docs/guides/SCRIPTS.md) | All scripts reference |
| **guides** | [INFERENCE.md](docs/guides/INFERENCE.md) | ONNX export, quantization |
| **guides** | [DOLPHIN.md](docs/guides/DOLPHIN.md) | Dolphin setup, running agents |
| **guides** | [TESTING.md](docs/guides/TESTING.md) | Test harness, coverage |
| **reference** | [ARCHITECTURE.md](docs/reference/ARCHITECTURE.md) | System design, network structure |
| **reference** | [architectures/](docs/reference/architectures/ARCHITECTURES.md) | Backbone docs (MLP, LSTM, Mamba) |
| **reference** | [EMBEDDING_DIMENSIONS.md](docs/reference/EMBEDDING_DIMENSIONS.md) | Tensor shape calculations |
| **reference** | [GOTCHAS.md](docs/reference/GOTCHAS.md) | Technical pitfalls and fixes |
| **operations** | [docker-workflow.md](docs/operations/docker-workflow.md) | Docker, cloud GPU deployment |
| **operations** | [REPLAY_STORAGE.md](docs/operations/REPLAY_STORAGE.md) | Cloud storage (B2/R2) |
| **research** | [RESEARCH.md](docs/research/RESEARCH.md) | Prior art, papers, lessons |
| **internals** | [GPU_OPTIMIZATIONS.md](docs/internals/GPU_OPTIMIZATIONS.md) | GPU training optimizations |
| **internals** | [MAMBA_OPTIMIZATIONS.md](docs/internals/MAMBA_OPTIMIZATIONS.md) | Mamba-specific optimizations |

**Generated API docs:** Run `mix docs` to generate ExDoc HTML in `doc/` (gitignored).

## Quick Start

**New defaults (Jan 2026):** Training now uses learned embeddings by default (~287 dims vs 1204). This allows 6x larger networks at the same training speed. Use `--action-mode one_hot --character-mode one_hot --stage-mode full` for legacy 1204-dim behavior.

```bash
# Install dependencies
mix deps.get

# NEW: Interactive setup wizard (recommended for beginners)
mix exphil.setup

# Using presets
mix run scripts/train_from_replays.exs --preset quick     # Fast iteration (~5 min)
mix run scripts/train_from_replays.exs --preset mewtwo    # Character-specific
mix run scripts/train_from_replays.exs --preset full      # Maximum quality

# Manual configuration (uses learned embeddings by default)
mix run scripts/train_from_replays.exs --temporal --backbone mamba --epochs 5

# Optional: Train K-means stick discretization for ~5% better precision inputs
mix run scripts/train_kmeans.exs --replays ./replays --k 21 --output priv/kmeans_centers.nx
mix run scripts/train_from_replays.exs --kmeans-centers priv/kmeans_centers.nx --temporal --backbone mamba

# Evaluate a trained model
mix run scripts/eval_model.exs --checkpoint checkpoints/model.axon

# Play against Dolphin
source .venv/bin/activate
mix run scripts/play_dolphin_async.exs \
  --policy checkpoints/policy.bin \
  --dolphin ~/.config/Slippi\ Launcher/netplay \
  --iso ~/melee.iso
```

**Environment variables** (optional):
```bash
export EXPHIL_REPLAYS_DIR=/path/to/replays
export EXPHIL_WANDB_PROJECT=my-project
```

**Useful flags:**
- `--verbose` / `--quiet` - Control output verbosity
- `--seed N` - Set random seed for reproducibility
- `--overwrite` - Allow checkpoint overwrite (creates .bak backup)

## RunPod Quick Reference

**On pod startup:**
```bash
cd /app
git pull origin main                    # Get latest code
source /app/scripts/runpod_entrypoint.sh  # Load helper commands
sync-checkpoints-down --latest          # Optional: pull previous checkpoints
```

**Start training (in tmux):**
```bash
tmux new -s train
mix run scripts/train_from_replays.exs \
  --backbone mlp \
  --hidden-sizes 512,512,256 \
  --epochs 10 \
  --batch-size 512 \
  --train-character mewtwo \
  --save-best \
  --replays /workspace/replays/mewtwo \
  --name mlp_mewtwo \
  2>&1 | tee /workspace/logs/training.log
# Ctrl+B, D to detach
```

**Before pod shutdown:**
```bash
sync-checkpoints-up   # Upload to B2 (organized by date)
```

**Checkpoint sync commands** (see [REPLAY_STORAGE.md](docs/REPLAY_STORAGE.md)):
- `sync-checkpoints-up` - Upload to today's date folder
- `sync-checkpoints-down --latest` - Download most recent
- `list-checkpoints` - List dates on B2

## Project Structure

```
exphil/
├── lib/exphil/
│   ├── networks/       # Policy, Value, Mamba, Attention, Recurrent
│   ├── embeddings/     # Player, Game, Controller, Character
│   ├── training/       # Imitation, PPO, Data
│   ├── agents/         # Agent GenServer, Supervisor
│   ├── bridge/         # MeleePort, AsyncRunner
│   └── rewards/        # Standard, Shaped
├── priv/python/        # libmelee bridge, ONNX tools
├── scripts/            # Training/evaluation scripts
├── docs/               # Detailed documentation
└── notebooks/          # Livebook analysis
```

## Key Technical Notes

See [docs/GOTCHAS.md](docs/GOTCHAS.md) for detailed fixes. Most common issues:

1. **EXLA tensor serialization** - Convert to BinaryBackend before saving checkpoints
2. **Dynamic shapes cause infinite JIT** - Use concrete seq_len for attention
3. **Closure tensor mismatch** - `Nx.backend_copy` captured tensors in `value_and_grad`
4. **Mix stale builds** - Run `mix compile --force` before training

## Development Practices

**File operations:**
- Use the Write tool for creating new files, not Bash with cat/heredoc
- Use the Edit tool for modifying existing files
- Use Read before Write to ensure you have current content

**Debugging tests:**
- CRITICAL: Never pipe `mix test` output (`| head`, `| tail`, etc.) - this kills streaming output and forces a full ~100s test suite rerun to see failures. Run `mix test` without piping, then use `grep -A` on visible output if needed.
- Never use `| tail` when investigating failures - it hides error details
- Use `mix test --failed` to rerun only failing tests
- For targeted debugging: `mix test path/to/test.exs:LINE_NUMBER`

**When something unexpected happens:**
- Document the finding in GOTCHAS.md or relevant docs
- If it's a pattern that could recur, add it to this file
- Update test coverage to prevent regression

**Code organization:**
- Struct field names: check `lib/exphil_bridge/types.ex` for canonical names
- JSON serialization: atoms become strings after round-trip
- File.stat mtime returns tuple `{{y,m,d},{h,m,s}}`, not NaiveDateTime

**Adding new CLI flags:**
When adding a new training flag, update these locations in order:
1. `lib/exphil/training/config.ex` - Add to `@valid_flags` list
2. `lib/exphil/training/config.ex` - Add default in `defaults/0`
3. `lib/exphil/training/config.ex` - Add parsing logic (e.g., `parse_int_arg`)
4. `docs/TRAINING.md` - Add to appropriate table: `| --flag-name | default | Description |`

The table format `| Option | Default | Description |` in TRAINING.md ensures:
- Discoverability: users can scan defaults without reading code
- Consistency: every flag follows the same documentation pattern
- Verification: easy to cross-reference against `@valid_flags` to catch drift

## Script Logging & UX Standards

All scripts use `ExPhil.Training.Output` for consistent, informative output. Key conventions:

**Always use the Output module** - `alias ExPhil.Training.Output`
```elixir
# Timestamped messages (preferred for all output)
Output.puts("Processing files...")           # [12:34:56] Processing files...
Output.success("Training complete")          # [12:34:56] ✓ Training complete (green)
Output.warning("Memory usage high")          # [12:34:56] ⚠️ Memory usage high (yellow)
Output.error("Failed to load checkpoint")    # [12:34:56] ❌ Failed to load checkpoint (red)

# Step indicators for multi-phase scripts
Output.step(1, 5, "Loading replays")         # [12:34:56] Step 1/5: Loading replays (cyan)
Output.step(2, 5, "Parsing frames")

# Script startup banner
Output.banner("ExPhil Architecture Benchmark")

# Configuration display
Output.config([
  {"Replay dir", replay_dir},
  {"Epochs", epochs},
  {"Batch size", batch_size}
])
```

**Progress bars for long operations:**
```elixir
# Live updating progress bar (overwrites line)
for {item, idx} <- Enum.with_index(items) do
  Output.progress_bar(idx + 1, length(items), label: "Processing")
  process(item)
end
Output.progress_done()  # Print newline after

# Timed blocks (shows "..." while running, then duration)
require Output
Output.timed "Loading dataset" do
  load_data()
end
# Output: [12:34:56] Loading dataset... done! (2.3s)
```

**JIT compilation indicator** - Always warn users before first batch:
```elixir
Output.puts("⏳ JIT compiling model (first batch)... this may take 2-5 minutes")
Output.puts("   (subsequent batches will be fast)")
```

**Training epoch progress:**
```elixir
# Inline progress: Epoch 1: ████████░░ 40% | 642/1606 | loss: 0.1234 | 0.5s/it | ETA: 8m 12s
# ALWAYS use \e[K to clear line (prevents artifacts in narrow terminals)
IO.write(:stderr, "\r#{progress_line}\e[K")
```

**Terminal width awareness** - For long progress lines that may wrap:
```elixir
# Truncate to terminal width to prevent line wrapping in narrow terminals (e.g., split-screen)
terminal_width = case :io.columns() do
  {:ok, cols} -> cols
  _ -> 120  # Fallback for non-TTY
end
truncated_line = if String.length(progress_line) > terminal_width - 1 do
  String.slice(progress_line, 0, terminal_width - 4) <> "..."
else
  progress_line
end
IO.write(:stderr, "\r#{truncated_line}\e[K")
```

**Progress interval standards** - Avoid log spam with intervals:
```elixir
# BAD: Updates every iteration (spams logs, no clear line escape)
Enum.each(items, fn {item, idx} ->
  IO.write(:stderr, "\r  Progress: #{idx}/#{total}")
  process(item)
end)

# GOOD: Updates at intervals with clear line escape
progress_interval = Keyword.get(opts, :progress_interval, 10)
Enum.each(items, fn {item, idx} ->
  if rem(idx, progress_interval) == 0 do
    IO.write(:stderr, "\r  Progress: #{idx}/#{total}\e[K")
  end
  process(item)
end)
```

Standard intervals by operation type:
| Operation | Default Interval | Rationale |
|-----------|-----------------|-----------|
| Training batches | 100 | ~500 updates/epoch is plenty |
| Embedding batches | 10 | ~20-30 updates total |
| Sequence building | 50,000 | Large counts, few updates |
| File hashing | 100 | Moderate update frequency |

All functions accepting `:show_progress` should also accept `:progress_interval`.

**GPU memory status** - Show at startup and each epoch:
```elixir
Output.puts(GPUUtils.memory_status_string())  # "GPU: 4.2/8.0 GB (52%)"
```

**Summary at end:**
```elixir
Output.training_summary(%{
  total_time_ms: total_time * 1000,
  epochs_completed: epochs,
  final_loss: loss,
  checkpoint_path: path
})
```

## Verbosity & Quiet Mode Standards

Scripts should support three verbosity levels via `--quiet` and `--verbose` flags:

| Flag | Level | Behavior |
|------|-------|----------|
| `--quiet` | 0 | Errors only, suppress warnings, XLA logs, progress bars |
| (default) | 1 | Normal output with progress bars at intervals |
| `--verbose` | 2 | Debug output: timing, memory, gradients |

**Implementing quiet mode:**
```elixir
# At script start (BEFORE any EXLA operations)
if "--quiet" in System.argv() do
  Logger.configure(level: :warning)  # Suppresses XLA [info] logs
end

# After parsing opts
Output.set_verbosity(opts[:verbosity])

# Progress bars respect verbosity
if opts[:verbosity] > 0 do
  IO.write(:stderr, "\r#{progress_line}")
end
```

**Key flags for log cleanliness:**
- `--quiet` - Suppresses warnings, XLA/ptxas logs, most output
- `--log-interval N` - Training progress updates every N batches (default: 100)
- `:progress_interval` option - Embedding/processing updates (default: 10)

**When adding new CLI flags:**
1. Add to `@valid_flags` in `lib/exphil/training/config.ex`
2. Add default in `defaults/0`
3. Add parsing logic
4. Document in `docs/TRAINING.md` table: `| --flag-name | default | Description |`

## References

**Melee AI Projects:**
- [slippi-ai](https://github.com/vladfi1/slippi-ai) - Primary reference (BC + RL)
- [Phillip](https://github.com/vladfi1/phillip) - Original pure RL approach
- [libmelee](https://github.com/altf4/libmelee) - Python game state API
- [SmashBot](https://github.com/altf4/SmashBot) - Rule-based baseline

**Key Papers:**
- [Beating the World's Best at SSBM](https://arxiv.org/abs/1702.06230) - Phillip paper
- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752) - Our backbone

**ML Framework:**
- [Nx](https://github.com/elixir-nx/nx) / [Axon](https://github.com/elixir-nx/axon) - Elixir ML
