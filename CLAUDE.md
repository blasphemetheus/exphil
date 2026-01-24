# ExPhil - Elixir Phil

ExPhil is an Elixir-based successor to slippi-ai, creating high-ELO playable bots for lower-tier Melee characters (Mewtwo, G&W, Link, Ganondorf, Zelda).

## Quick Reference

**Embedding dimensions:**
- Player: 488 dims (base 440 + speeds 5 + frame_info 2 + stock 1 + ledge_dist 1 + compact Nana 39)
- Player (learned actions): 91 dims (base 41 + speeds 5 + frame_info 2 + stock 1 + ledge_dist 1 + compact Nana 39 + action IDs in network)
- Player (learned actions + chars): 58 dims (excludes 33-dim character one-hot)
- Game: 1204 dims (2 players + stage + spatial features + projectiles)
- Game (learned actions): 408 dims + 2 action IDs (use `action_mode: :learned` for trainable action embedding)
- Game (learned actions + chars): 344 dims + 4 IDs (saves 64 dims net: 2×33 one-hot → 2 char IDs)
- **512-dim target**: 254 continuous + 4 action IDs × 64 embedding (use `action_mode: :learned, nana_mode: :enhanced`)
- Nana modes:
  - `:compact` (default): 39 dims (preserves IC tech: handoffs, regrabs, desyncs)
  - `:enhanced`: 14 dims + Nana action ID (most efficient for learned embeddings, precise IC tech)
  - `:full`: 449 dims (complete action state, use `nana_mode: :full`)
- Jumps: Normalized (1 dim) by default; use `jumps_normalized: false` for 7-dim one-hot
- Controller: 13 dims (8 buttons + 4 sticks + 1 shoulder)
- Stick discretization: 17 uniform buckets (default) or 21 K-means clusters (`--kmeans-centers`)
- Action modes: `:one_hot` (399 dims, default) or `:learned` (64-dim trainable embedding, saves ~670 dims)
- Character modes: `:one_hot` (33 dims, default) or `:learned` (64-dim trainable embedding, saves 64 dims)
- Stage modes:
  - `:one_hot_full` (default): 64 dims (full stage one-hot)
  - `:one_hot_compact`: 7 dims (6 competitive stages + "other", saves 57 dims)
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

**Test coverage:** 1459 tests passing

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
- Self-play GenServer infrastructure

**Next Phase:**
1. **Self-play Dolphin testing** - Run with real MeleePort games
2. **Large-scale self-play training** - GPU cluster deployment
3. **Character-specific rewards** - Mewtwo recovery, Ganon spacing, etc.

## Documentation

Detailed guides in `docs/`:

| Doc | Contents |
|-----|----------|
| [GOALS.md](docs/GOALS.md) | **Big picture goals, priorities, implementation log** |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, network structure, training pipeline |
| [architectures/](docs/architectures/ARCHITECTURES.md) | **Backbone docs: MLP, LSTM, GRU, Attention, Mamba, Jamba** |
| [TRAINING.md](docs/TRAINING.md) | Training commands, options, performance tips |
| [INFERENCE.md](docs/INFERENCE.md) | ONNX export, quantization, optimization |
| [DOLPHIN.md](docs/DOLPHIN.md) | Dolphin setup, running agents |
| [GOTCHAS.md](docs/GOTCHAS.md) | Technical pitfalls and fixes |
| [TRAINING_FEATURES.md](docs/TRAINING_FEATURES.md) | Feature roadmap, presets |
| [RESEARCH.md](docs/RESEARCH.md) | Prior art, papers, lessons learned, research roadmap |
| [docker-workflow.md](docs/docker-workflow.md) | Docker build, push, cloud GPU deployment |
| [REPLAY_STORAGE.md](docs/REPLAY_STORAGE.md) | Cloud storage setup for replays (B2/R2) |
| [RUNPOD_FILTER.md](docs/RUNPOD_FILTER.md) | Filter low-tier replays from large archives on RunPod |
| [RCLONE_GDRIVE.md](docs/RCLONE_GDRIVE.md) | Download large files from Google Drive with rclone |
| [TRAINING_IMPROVEMENTS.md](docs/TRAINING_IMPROVEMENTS.md) | Training-specific optimizations |
| [PROJECT_ROADMAP.md](docs/PROJECT_ROADMAP.md) | Comprehensive project improvement roadmap |
| [TESTING.md](docs/TESTING.md) | Test harness, tags, factories, coverage |

## Quick Start

```bash
# Install dependencies
mix deps.get

# NEW: Interactive setup wizard (recommended for beginners)
mix exphil.setup

# Using presets
mix run scripts/train_from_replays.exs --preset quick     # Fast iteration (~5 min)
mix run scripts/train_from_replays.exs --preset mewtwo    # Character-specific
mix run scripts/train_from_replays.exs --preset full      # Maximum quality

# Manual configuration
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
IO.write(:stderr, "\r#{progress_line}")  # Carriage return to overwrite
```

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
