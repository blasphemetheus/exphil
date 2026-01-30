# ExPhil - AI for Melee's Forgotten Characters

**Teaching neural networks to play Mewtwo, Ganondorf, Link, Zelda, Ice Climbers, and Game & Watch at a high level.**

ExPhil is building competitive AI agents for Melee's lower-tier characters using imitation learning from human replays, followed by reinforcement learning refinement. The goal: bots that play these characters *well*, not just bots that play them.

---

## 🎮 We Need Your Replays!

**If you play any of these characters, your replays are incredibly valuable:**

| Character | Why We Need Replays |
|-----------|---------------------|
| **Mewtwo** | Teleport recovery, tail hitboxes, Shadow Ball spacing |
| **Ganondorf** | Punish optimization, spacing, stomp setups |
| **Link** | Projectile zoning, bomb recovery, zair |
| **Zelda** | Transform decisions, lightning kick spacing |
| **Ice Climbers** | Desync tech, wobbling, handoffs, dual-character control |
| **Game & Watch** | Bucket, only fair/dair L-cancels, hammer RNG |

### How to Donate Replays

1. **Find your replays**: `~/.slippi/replays/` (Linux/Mac) or `%APPDATA%\Slippi\replays\` (Windows)
2. **Filter for your character**: We only need games where you played one of the characters above
3. **Any skill level helps**, but ranked/tournament games are especially valuable
4. **Send them**: dleie on discord (spikenard)

> 💡 **Why low-tier replays matter**: Top-tier characters have thousands of hours of pro gameplay to learn from. Mewtwo mains? Not so much. Your games—even if you're not a top player—contain crucial character-specific knowledge that doesn't exist elsewhere.

---

## 🤖 What We're Building

ExPhil learns to play Melee in two stages:

### Stage 1: Imitation Learning (Current Focus)
The AI watches human replays and learns to mimic the decisions players make. It sees the game state (positions, percents, actions) and learns to predict what controller inputs a human would use.

### Stage 2: Reinforcement Learning (Coming Soon)
Once the AI can play at a basic level, it plays against itself millions of times, getting rewarded for taking stocks and punished for losing them. This is how it goes from "plays like a human" to "plays better than most humans."

---

## 📊 Technical Overview

*For the ML folks who want the details.*

### Architecture

```
Game State → Embedding (288 dims) → Temporal Backbone → Policy Head → Controller Output
                                          ↓
                                    Mamba SSM / Attention
                                    (60-frame context)
```

**State Embedding**: Player positions, velocities, action states, damage, stocks, stage, projectiles (288 dims with learned embeddings, ~1204 dims with one-hot)

**Temporal Backbone**: [Mamba](https://arxiv.org/abs/2312.00752) state-space model processes 60 frames of context. Achieves 8.9ms inference (vs 220ms for LSTM) while maintaining temporal reasoning.

**Policy Head**: 6-head autoregressive controller output:
- Buttons: 8 independent Bernoulli (A, B, X, Y, Z, L, R, Start)
- Main stick: 17×17 discretized grid (X, Y sampled sequentially)
- C-stick: 17×17 discretized grid
- Shoulder: 5 levels (0, 0.35, 0.7, 0.9, 1.0)

### Training Pipeline

```
Slippi Replays (.slp)
        ↓
    Peppi Parser (Rust NIF)
        ↓
    Frame Extraction + Embedding
        ↓
    Behavioral Cloning (Cross-entropy loss)
        ↓
    Policy Network (Axon/Nx)
        ↓
    [Future] PPO Self-Play
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Frame Delay Augmentation** | Train with 0-18 frame random delays for online play robustness |
| **Mamba Backbone** | Linear-time sequence modeling, 25x faster than LSTM |
| **Cosine Annealing + Warm Restarts** | SGDR learning rate schedule for escaping local minima |
| **Model EMA** | Exponential moving average weights for better generalization |
| **Character Presets** | Tuned hyperparameters per character (window size, context length) |

### Inference Performance

| Backend | Latency | 60 FPS Ready |
|---------|---------|--------------|
| Mamba (Axon) | 8.9 ms | ✅ Yes |
| LSTM (Axon) | 220 ms | ❌ No |
| ONNX INT8 | 0.55 ms | ✅ Yes |

### Research References

This project builds on:
- [slippi-ai](https://github.com/vladfi1/slippi-ai) - Vladfi's BC+RL framework (TensorFlow)
- [Project Nabla](https://bycn.github.io/2022/08/19/project-nabla-writeup.html) - Lessons on policy collapse in fighting game AI
- [Phillip](https://github.com/vladfi1/phillip) - Original deep RL Melee bot

Key lessons applied:
1. **Two-stage training**: BC first, RL second (pure RL struggles with Melee's complexity)
2. **Autoregressive action heads**: Sample buttons → stick → c-stick sequentially
3. **Population-based training**: Avoid policy collapse by maintaining diverse agents
4. **Frame delay training**: Essential for online play where 18+ frames of latency exist

---

## 🛠 Tech Stack

- **Language**: Elixir (BEAM concurrency for future self-play)
- **ML Framework**: [Nx](https://github.com/elixir-nx/nx) / [Axon](https://github.com/elixir-nx/axon) with EXLA (XLA backend)
- **Replay Parsing**: [Peppi](https://github.com/hohav/peppi) via Rustler NIF
- **Game Interface**: [libmelee](https://github.com/altf4/libmelee) (Python, via Port)
- **Inference**: ONNX Runtime for production deployment

---

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/blasphemetheus/exphil.git
cd exphil
mix deps.get

# Train on replays (quick test)
mix run scripts/train_from_replays.exs --preset quick --replays ~/my_replays

# Train production model
mix run scripts/train_from_replays.exs --preset production --online-robust

# Evaluate model
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

## 📈 Current Status

- ✅ Imitation learning pipeline (single-frame + temporal)
- ✅ All backbones: MLP, LSTM, GRU, Mamba, Attention
- ✅ Dolphin integration for live play
- ✅ Training infrastructure (EMA, cosine scheduling, augmentation)
- 🔄 Collecting more low-tier replays (this is where you come in!)
- ⏳ Self-play reinforcement learning
- ⏳ Character-specific reward shaping

---

## 🙏 Contributing

### Replay Donations
The single most valuable contribution right now is **replay files** for the target characters. Even a few games help expand the training distribution.

### Processing Large Replay Archives
Have access to large Slippi replay archives? We have tools to filter them for low-tier characters:

```bash
# Filter replays from Google Drive archives on RunPod
python3 cloud_filter_replays.py --urls-file links.txt --output /workspace/lowtier --cleanup
```

See [RUNPOD_FILTER.md](docs/RUNPOD_FILTER.md) for the full guide on processing 100GB+ archives in the cloud.

### Code Contributions
PRs welcome! See the [issues](https://github.com/blasphemetheus/exphil/issues) for current priorities.

### Research Collaboration
If you're working on fighting game AI, game-playing agents, or imitation learning, let's talk!

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>Why doesn't slippi-ai have mewtwo?.</i>
</p>
