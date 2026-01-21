# ExPhil - Elixir Phil

ExPhil is an Elixir-based successor to slippi-ai, creating high-ELO playable bots for lower-tier Melee characters (Mewtwo, G&W, Link, Ganondorf, Zelda).

## Quick Reference

**Embedding dimensions:**
- Player: 446 dims
- Game: ~1991 dims (2 players + stage + projectiles)
- Controller: 13 dims (8 buttons + 4 sticks + 1 shoulder)

**Policy network:** 6-head autoregressive (buttons, main_x/y, c_x/y, shoulder)

**Inference benchmarks:**
| Architecture | Inference | 60 FPS Ready |
|--------------|-----------|--------------|
| LSTM (Axon) | 220.95 ms | No |
| Mamba (Axon) | 8.93 ms | **Yes** |
| ONNX INT8 | 0.55 ms | **Yes** |

## Current Status

**Test coverage:** 900+ tests passing

**Completed:**
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
- Model evaluation script (`scripts/eval_model.exs`)

## Immediate Priorities

See [GOALS.md](docs/GOALS.md) for comprehensive roadmap. Current focus:

**High Impact (In Progress):**
1. **Projectile parsing** - 40-60% state info lost for Link/Samus/Falco
2. **Focal loss** - +30-50% accuracy on rare actions (Z/L/R)
3. **Embedding caching** - 2-3x training speedup

**Next Phase:**
4. **Self-play infrastructure** - BEAM concurrency + population-based training
5. **Character-specific rewards** - Mewtwo recovery, Ganon spacing, etc.

## Documentation

Detailed guides in `docs/`:

| Doc | Contents |
|-----|----------|
| [GOALS.md](docs/GOALS.md) | **Big picture goals, priorities, implementation log** |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, network structure, training pipeline |
| [TRAINING.md](docs/TRAINING.md) | Training commands, options, performance tips |
| [INFERENCE.md](docs/INFERENCE.md) | ONNX export, quantization, optimization |
| [DOLPHIN.md](docs/DOLPHIN.md) | Dolphin setup, running agents |
| [GOTCHAS.md](docs/GOTCHAS.md) | Technical pitfalls and fixes |
| [TRAINING_FEATURES.md](docs/TRAINING_FEATURES.md) | Feature roadmap, presets |
| [RESEARCH.md](docs/RESEARCH.md) | Prior art, papers, lessons learned, research roadmap |
| [docker-workflow.md](docs/docker-workflow.md) | Docker build, push, cloud GPU deployment |
| [RUNPOD_FILTER.md](docs/RUNPOD_FILTER.md) | Filter low-tier replays from large archives on RunPod |
| [RCLONE_GDRIVE.md](docs/RCLONE_GDRIVE.md) | Download large files from Google Drive with rclone |
| [TRAINING_IMPROVEMENTS.md](docs/TRAINING_IMPROVEMENTS.md) | Training-specific optimizations |
| [PROJECT_ROADMAP.md](docs/PROJECT_ROADMAP.md) | Comprehensive project improvement roadmap |
| [TESTING.md](docs/TESTING.md) | Test harness, tags, factories, coverage |

## Quick Start

```bash
# Install dependencies
mix deps.get

# Single-frame training (baseline)
mix run scripts/train_from_replays.exs --epochs 10 --max-files 50

# Temporal training with Mamba (recommended)
mix run scripts/train_from_replays.exs --temporal --backbone mamba --epochs 5

# Using presets
mix run scripts/train_from_replays.exs --preset quick     # Fast iteration
mix run scripts/train_from_replays.exs --preset full      # Maximum quality

# Evaluate a trained model
mix run scripts/eval_model.exs --checkpoint checkpoints/model.axon

# Play against Dolphin
source .venv/bin/activate
mix run scripts/play_dolphin_async.exs \
  --policy checkpoints/policy.bin \
  --dolphin ~/.config/Slippi\ Launcher/netplay \
  --iso ~/melee.iso
```

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
