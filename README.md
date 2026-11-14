# ExPhil - Neural Network Architecture Lab for Melee AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Experimenting with neural network architectures to build competitive Super Smash Bros. Melee bots in Elixir.**

ExPhil is a platform for training and evaluating AI agents that play Melee via [Slippi](https://slippi.gg/). It supports 30+ backbone architectures — from MLPs to Mamba SSMs to Transformers — all implemented in Elixir using Nx/Axon. The project uses imitation learning from human replays, with reinforcement learning refinement planned.

The long-term goal is competitive bots for Melee's lower-tier characters (Mewtwo, Ganondorf, Link, Zelda, Ice Climbers, Game & Watch), but high-tier replay data is more abundant, so that's where architecture iteration happens first.

**Start here:** [Current status and supported workflow](docs/planning/CURRENT_STATUS.md),
[repository improvement checklist](docs/planning/REPO_IMPROVEMENTS.md).

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

### Historical Backbone Benchmarks

These results have no recorded benchmark date or code revision. They describe
an earlier evaluation setup, not current stateful inference or causal-policy
quality; use them as historical context rather than a deployment recommendation.

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
# Run in a parent directory for the three repositories.
git clone https://github.com/blasphemetheus/edifice.git
git clone https://github.com/blasphemetheus/libmelee_ex.git
git clone https://github.com/blasphemetheus/exphil.git
cd exphil
mix deps.get

# Interactive setup (recommended for beginners)
mix exphil.setup

# Train with presets
mix run scripts/train.exs --preset quick --replays ./replays

# Try different architectures
mix run scripts/train.exs --backbone gru --replays ./replays
mix run scripts/train.exs --backbone attention --replays ./replays

# Evaluate a checkpoint (see supported modes below)
mix run scripts/eval_model.exs --checkpoint checkpoints/model.axon
```

See [contributor setup](CONTRIBUTING.md#development-setup) for prerequisites and
local dependency overrides. `scripts/train.exs` is the current entry point;
`train_from_replays.exs` is legacy. Run setup/training commands in a separate
environment from any active training job.

**Evaluation:** `eval_model.exs` supports checkpoint-driven GRU BPTT evaluation;
see [supported modes and limits](docs/guides/BPTT_EVALUATION.md). Queued-action /
delay-ID multishine policies need gameplay evaluation; the
[multishine replay benchmark](docs/guides/MULTISHINE_BENCHMARK.md) measures chains,
re-entry, and teacher coverage. For Dolphin play, copy the checkpoint-specific
settings from the [deployment cards](docs/guides/DEPLOY_KNOBS.md); delay and
decoding settings affect behavior.

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

- Test suites cover training, embeddings, inference, and replay handling; the
  default test command excludes several native/integration categories.
- Imitation learning pipeline (single-frame + temporal)
- 30+ backbone architectures benchmarked or available
- 4 policy types (Standard, Diffusion, ACT, Flow Matching)
- Dolphin integration for live play
- PPO trainer and self-play infrastructure built
- Training features: EMA, cosine scheduling, augmentation, caching, checkpointing

**Next**: Evaluate causal-policy training and improve checkpoint evaluation,
configuration consistency, reproducibility, and semantic CI coverage. See
[current status](docs/planning/CURRENT_STATUS.md) and the
[engineering checklist](docs/planning/REPO_IMPROVEMENTS.md).

---

## Competing in the stomped.dev Arena

[stomped.dev](https://stomped.dev) is a ranked ladder for Melee bots. A bot there
is a web server: the arena opens a WebSocket to a URL you register, streams the
game as the raw `.slp` event stream (one message per frame), and reads 12-byte
controller pads back. Games are vanilla NTSC 1.02, 4 stocks, 8 minutes, on a
random legal stage, rated with OpenSkill. The full spec is at
[docs.stomped.dev](https://docs.stomped.dev/).

This suits ExPhil well: the whole bot runs on the BEAM, with no Dolphin,
libmelee, or Python on our side, and the observations are the same bytes our
training replays are made of.

### What is built

The arena server lives in `lib/exphil/stomped/`:

| Module | Role |
|--------|------|
| `ExPhil.Stomped.Server` | Bandit HTTP server: `GET <url>/ping` and the match WebSocket |
| `ExPhil.Stomped.Router` | Secret-path routing and signed-handshake verification (Ed25519) |
| `ExPhil.Stomped.Match` | The match protocol as a pure state machine (`character`, `ready`, `game`, `over`, reconnects) |
| `ExPhil.Stomped.Events` | `.slp` event framing: splits each message into events and frames |
| `ExPhil.Stomped.Decoder` | Frame events to the training-shaped `GameState`, plus the controller the game polled |
| `ExPhil.Stomped.Replay` | Cuts a stored `.slp` back into the arena's messages (offline stand-in for the socket) |
| `ExPhil.Stomped.Pad` | The 12-byte pad encoder and the policy-controller conversion |
| `ExPhil.Stomped.Bot` | The behaviour a bot implements (pick a character, answer each frame) |
| `ExPhil.Stomped.Bots.Policy` | Plays a checkpoint: decoder, warm agent, pad; scores landed and late pads |
| `ExPhil.Stomped.Seat` | Lends the one warm agent to one match at a time |
| `ExPhil.Stomped.Bots.Walker` | The arena's quickstart bot (walk at the opponent, tap A), for proving a deploy |

```bash
# Start the server (prints the secret path if you don't give one)
mix run scripts/stomped_serve.exs --path-secret "$(uuidgen)"
curl localhost:3000/<secret>/ping            # -> pong

# Put it on the internet, then register https://<host>/<secret> at stomped.dev/bots
cloudflared tunnel --url http://localhost:3000

# Once registered, verify the arena's signed handshake
mix run scripts/stomped_serve.exs --path-secret <secret> --url https://<host>/<secret>

# Play a checkpoint instead of the quickstart walker (CPU client: faster for a
# single-frame GRU step than the GPU, and it leaves the GPU to training)
env EXLA_TARGET=host mix run scripts/stomped_serve.exs \
  --path-secret <secret> --url https://<host>/<secret> \
  --policy eval_runs/0923_ppo/eval_candidate/candidate_policy.bin
```

The server times the arena's `ping` each match and aims just far enough ahead
for the measured round trip (`--lead-extra auto`, the default; a number pins
it). When a match closes it logs the lead it used, the round trip, and how
many pads landed or arrived late.

Measured per-frame compute for the PPO Fox (512-wide GRU, stateful step) on
one real arena game: 0.59 ms mean / 1.22 ms p99 on CPU, 1.80 ms mean / 3.02 ms
p99 on the RTX 5090. The arena's budget is 11 ms including the network.

A serving bot is a live ExPhil BEAM: the no-`mix`-next-to-it rule applies as
it does for a training run. If you do serve on the GPU, set
`EXPHIL_GPU_MEMORY_FRACTION=0.1` (the default reserves 75% of VRAM).

The registered URL is the bot's password, so it needs an unguessable path. A
tunnel exposes a port on your machine to the internet; the server answers only
the ping and the socket under the secret path.

### What is left

- [x] WebSocket server, match protocol, pad encoding, handshake verification
- [x] **Live `.slp` decoder** (`ExPhil.Stomped.Decoder`): each frame's events
      become the game state the policy embeds, in the same shape training
      builds from stored replays. Gate: on five real arena games (one per
      stage, both seats) every frame equals the `exphil_peppi` training frame
      (`mix test test/exphil/stomped --include nif`).
- [x] **Controller to pad** (`ExPhil.Stomped.Pad.from_controller/1`): the
      inverse of the game's stick processing, checked against the raw bytes
      recorded in those games. Trigger bytes are not recorded in replays, so
      that half is only checkable in a live game.
- [x] **Policy bot** (`ExPhil.Stomped.Bots.Policy`, `--policy PATH`): one warm
      `ExPhil.Agents.Agent` (stateful-step inference) lent to one match at a
      time. Each pad is also sent for the following frame, so a late decision
      repeats the last input instead of reading neutral, and every frame's
      echo is scored as landed or late. Timed offline on a real arena game
      with `scripts/stomped_replay_bench.exs`.
- [ ] **Offline arena client**: replay a stored `.slp` over the socket to test
      the full path, with timing, without waiting for a ladder game.
- [ ] **Delay choice**: a pad is due 11 ms after its frame is sent, network
      included. The arena is in Ashburn, Virginia; from further away the bot
      aims extra frames ahead (`--lead-extra N`), which is reaction delay
      `1 + N` in ExPhil's numbering. The checkpoint must be trained at that
      delay.
- [ ] **Stage coverage**: the stage is random among the six legal ones, so the
      deployed policy has to hold up on all of them.
- [ ] **Hosting**: a home tunnel to start; a US-East host if the measured round
      trip costs too many frames.

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
