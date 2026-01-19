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

**Test coverage:** 580 tests passing

**Completed:**
- Imitation learning (single-frame + temporal)
- All backbones: MLP, LSTM, GRU, Mamba, attention
- PPO trainer with clipped objective
- Dolphin integration (sync + async runners)
- Training features: early stopping, LR scheduling, gradient accumulation, validation split, checkpointing, memorable model names

## Immediate Priorities

1. **Self-play infrastructure** - Train agent vs agent using BEAM concurrency
2. **Character-specific rewards** - Mewtwo, G&W, Link specialization
3. **Data augmentation** - Mirror states, noise injection
4. **Model registry** - Track trained models with lineage

## Documentation

Detailed guides in `docs/`:

| Doc | Contents |
|-----|----------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, network structure, training pipeline |
| [TRAINING.md](docs/TRAINING.md) | Training commands, options, performance tips |
| [INFERENCE.md](docs/INFERENCE.md) | ONNX export, quantization, optimization |
| [DOLPHIN.md](docs/DOLPHIN.md) | Dolphin setup, running agents |
| [GOTCHAS.md](docs/GOTCHAS.md) | Technical pitfalls and fixes |
| [TRAINING_FEATURES.md](docs/TRAINING_FEATURES.md) | Feature roadmap, presets |

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

## References

- [slippi-ai](https://github.com/vladfi1/slippi-ai) - Primary reference
- [libmelee](https://github.com/altf4/libmelee) - Game interface
- [Nx](https://github.com/elixir-nx/nx) / [Axon](https://github.com/elixir-nx/axon) - ML framework
