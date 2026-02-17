# ExPhil - Neural Network Architecture Lab for Melee AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Experimenting with neural network architectures to build competitive Super Smash Bros. Melee bots in Elixir.**

ExPhil is a platform for training and evaluating AI agents that play Melee via [Slippi](https://slippi.gg/). It supports 30+ backbone architectures — from MLPs to Mamba SSMs to Transformers — all implemented in Elixir using Nx/Axon. The project uses imitation learning from human replays, with reinforcement learning refinement planned.

The long-term goal is competitive bots for Melee's lower-tier characters (Mewtwo, Ganondorf, Link, Zelda, Ice Climbers, Game & Watch), but high-tier replay data is more abundant, so that's where architecture iteration happens first.

---

## Why Elixir?

Most game AI research uses Python/PyTorch. ExPhil uses Elixir deliberately:

- **BEAM concurrency** for parallel self-play (multiple game instances, Elo matchmaking, population training)
- **Nx/Axon + EXLA** gives GPU acceleration via XLA without Python overhead
- **Fault tolerance** — OTP supervisors keep training alive through transient failures
- **[Edifice](https://github.com/blasphemetheus/edifice)** — our companion library with 92+ ML architecture implementations in pure Elixir

---

## Architecture Experiments

The core question: *which neural network architecture best captures Melee's temporal dynamics at 60 FPS?*

### Benchmarked Backbones

| Backbone | Type | Inference | 60 FPS | Val Loss | Notes |
|----------|------|-----------|--------|----------|-------|
| MLP | Single-frame | 9 ms | Yes | 3.11 | No temporal context |
| LSTM | Recurrent | 229 ms | No | **2.95** | Best accuracy, too slow |
| GRU | Recurrent | ~150 ms | No | ~3.0 | Faster LSTM variant |
| Attention | Transformer | 17 ms | Borderline | 3.07 | Parallelizable |
| Mamba | State Space | 8.9 ms | **Yes** | 3.00 | Best speed/accuracy tradeoff |
| Jamba | Hybrid SSM | ~20 ms | Needs opt | ~3.0 | Mamba + Attention layers |

*3-epoch benchmark, RTX 4090, batch_size=256, 50 replays*

### Additional Architectures

All available for experimentation: Griffin, xLSTM, RetNet, RWKV, GLA, HGRN, KAN, S4, S4D, S5, H3, Liquid, Decision Transformer, Zamba, Mamba-2, GatedSSM, Performer, DeltaNet, FNet, Perceiver, TTT, Hopfield, NTM, Reservoir, SNN, Bayesian — plus Diffusion, ACT, and Flow Matching policy types.

See [Architecture Guide](docs/reference/architectures/ARCHITECTURE_GUIDE.md) for beginner-friendly explanations, or [Architectures Reference](docs/reference/architectures/ARCHITECTURES.md) for benchmarks and CLI options.

---

## How It Works

```
Slippi Replays (.slp)
        |
    Peppi Parser (Rust NIF)
        |
    Frame Extraction + Learned Embedding (288 dims)
        |
    Temporal Backbone (Mamba / Attention / LSTM / ...)
        |
    Autoregressive Policy Head (6 heads)
        |
    Controller Output → Dolphin
```

**State Embedding**: Player positions, velocities, action states, damage, stocks, stage, projectiles. Uses learned embeddings by default (288 dims) — trainable action/character/stage embeddings replace one-hot vectors, enabling 6x larger networks at the same training speed.

**Temporal Backbone**: Processes 60 frames of game context. Swappable via `--backbone` flag.

**Policy Head**: 6-head autoregressive controller output:
- Buttons (8 independent), main stick (17x17), c-stick (17x17), shoulder (5 levels)

### Training Pipeline

1. **Behavioral Cloning** — learn from human replays via cross-entropy loss
2. **Self-Play RL** — PPO refinement against itself *(infrastructure complete, scaling next)*

---

## Quick Start

```bash
git clone https://github.com/blasphemetheus/exphil.git
cd exphil
mix deps.get

# Interactive setup (recommended for beginners)
mix exphil.setup

# Train with presets
mix run scripts/train_from_replays.exs --preset quick --replays ./replays
mix run scripts/train_from_replays.exs --preset production --online-robust

# Try different architectures
mix run scripts/train_from_replays.exs --backbone mamba --temporal --replays ./replays
mix run scripts/train_from_replays.exs --backbone attention --temporal --replays ./replays
mix run scripts/train_from_replays.exs --backbone griffin --temporal --replays ./replays

# Evaluate
mix run scripts/eval_model.exs --checkpoint checkpoints/model.axon

# Play against the AI (requires Dolphin + Slippi)
mix run scripts/play_dolphin_async.exs --policy checkpoints/model_policy.bin
```

### Training Presets

| Preset | Use Case | Time (RTX 4090) |
|--------|----------|-----------------|
| `quick` | Code testing | ~2 min |
| `gpu_standard` | Balanced training | ~30 min |
| `production` | Maximum quality | ~2-3 hours |
| `mewtwo` | Mewtwo specialist | ~2-3 hours |
| `ganondorf` | Ganondorf specialist | ~2-3 hours |

---

## Key Features

| Feature | Description |
|---------|-------------|
| **30+ Backbones** | Swap architectures with a single CLI flag |
| **Learned Embeddings** | 288-dim trainable embeddings (6x more efficient than one-hot) |
| **Frame Delay Augmentation** | Train with 0-18 frame random delays for online robustness |
| **Embedding Caching** | 2-3x training speedup, ~100x with augmented caching |
| **Model EMA** | Exponential moving average for better generalization |
| **ONNX Export** | INT8 quantization for 0.55ms inference |
| **Dolphin Integration** | Sync and async runners for live play |
| **Self-Play Infrastructure** | PPO, population training, Elo matchmaking |

---

## Tech Stack

- **Language**: [Elixir](https://elixir-lang.org/) on the BEAM VM
- **ML**: [Nx](https://github.com/elixir-nx/nx) / [Axon](https://github.com/elixir-nx/axon) with [EXLA](https://github.com/elixir-nx/nx/tree/main/exla) (XLA backend)
- **Architectures**: [Edifice](https://github.com/blasphemetheus/edifice) — 92+ architecture implementations in Elixir
- **Replay Parsing**: [Peppi](https://github.com/hohav/peppi) via Rustler NIF
- **Game Interface**: [libmelee](https://github.com/altf4/libmelee) (Python, via Port)
- **Inference**: ONNX Runtime for production deployment

---

## Current Status

- 2469 tests passing
- Imitation learning pipeline (single-frame + temporal)
- 30+ backbone architectures benchmarked or available
- 4 policy types (Standard, Diffusion, ACT, Flow Matching)
- Dolphin integration for live play
- PPO trainer and self-play infrastructure built
- Training features: EMA, cosine scheduling, augmentation, caching, checkpointing

**Next**: Large-scale self-play training on GPU clusters, scaling experiments with more data and larger models.

---

## We Want Your Replays

Training data makes or breaks imitation learning. If you play Melee competitively, your replays are valuable — especially for lower-tier characters where data is scarce.

**Characters we especially need**: Mewtwo, Ganondorf, Link, Zelda, Ice Climbers, Game & Watch

Find your replays at `~/.slippi/replays/` and reach out: **dleie** on Discord (spikenard)

---

## Research References

Builds on:
- [slippi-ai](https://github.com/vladfi1/slippi-ai) — Vladfi's BC+RL framework (TensorFlow)
- [Project Nabla](https://bycn.github.io/2022/08/19/project-nabla-writeup.html) — Lessons on policy collapse in fighting game AI
- [Phillip](https://github.com/vladfi1/phillip) — Original deep RL Melee bot

Key lessons applied:
1. **Two-stage training**: BC first, RL second (pure RL struggles with Melee's complexity)
2. **Autoregressive action heads**: Sample buttons, stick, c-stick sequentially
3. **Population-based training**: Maintain diverse agents to avoid policy collapse
4. **Frame delay training**: Essential for online play with 18+ frames of latency

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and guidelines. PRs welcome!

---

## License

MIT License — see [LICENSE](LICENSE) for details.
