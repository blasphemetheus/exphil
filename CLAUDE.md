# ExPhil - Elixir Phil

ExPhil is an Elixir-based successor to slippi-ai, designed to create high-ELO playable bots for lower-tier Melee characters.

## Target Characters (Initial Focus)
1. **Mewtwo** - Unique physics, teleport recovery, tail hurtbox mechanics
2. **Mr. Game & Watch** - No L-cancel, random hammer, unique shield
3. **Link** - Projectiles, tether recovery, bomb tech
4. **Ganondorf** - Slow but powerful, requires precise spacing/timing

## Project Structure

```
exphil/
├── lib/
│   ├── exphil/
│   │   ├── application.ex          # OTP Application
│   │   ├── networks/               # Neural network architectures
│   │   │   ├── policy.ex           # Policy network (actor)
│   │   │   ├── value.ex            # Value function (critic)
│   │   │   ├── transformer.ex      # Transformer-like temporal attention
│   │   │   └── recurrent.ex        # LSTM/GRU alternatives
│   │   ├── embeddings/             # Game state → tensor conversion
│   │   │   ├── player.ex           # Player state embedding
│   │   │   ├── game.ex             # Full game state embedding
│   │   │   ├── controller.ex       # Controller action embedding
│   │   │   └── character.ex        # Character-specific embeddings
│   │   ├── training/               # Training loops
│   │   │   ├── imitation.ex        # Behavioral cloning from replays
│   │   │   ├── ppo.ex              # Proximal Policy Optimization
│   │   │   ├── vtrace.ex           # V-trace for off-policy correction
│   │   │   └── distributed.ex      # Multi-GPU/node training
│   │   ├── agents/                 # Agent implementations
│   │   │   ├── basic.ex            # Single-step agent
│   │   │   └── delayed.ex          # Agent with frame delay handling
│   │   ├── rewards/                # Reward computation
│   │   │   ├── standard.ex         # KO/damage differential
│   │   │   └── shaped.ex           # Shaped rewards (approach, combo)
│   │   ├── data/                   # Data pipeline
│   │   │   ├── replay_parser.ex    # Parse .slp files (via Peppi)
│   │   │   ├── dataset.ex          # Training dataset management
│   │   │   └── batch.ex            # Batching utilities
│   │   └── eval/                   # Evaluation
│   │       ├── metrics.ex          # Performance metrics
│   │       └── matchups.ex         # Character matchup tracking
│   └── exphil_bridge/              # Python interop
│       ├── libmelee_port.ex        # libmelee communication
│       └── dolphin.ex              # Dolphin process management
├── priv/
│   └── python/
│       ├── melee_bridge.py         # Python-side libmelee wrapper
│       └── replay_converter.py     # Convert .slp to training format
├── test/
├── config/
│   ├── config.exs
│   ├── dev.exs
│   └── prod.exs
├── scripts/
│   ├── train_imitation.exs         # Imitation learning script
│   ├── train_rl.exs                # RL fine-tuning script
│   └── eval.exs                    # Evaluation script
├── docs/
│   └── ARCHITECTURE.md
├── notebooks/                      # Livebook notebooks for analysis
├── mix.exs
└── CLAUDE.md
```

## Implementation Plan

### Phase 1: Foundation (Weeks 1-3)
Core infrastructure and Python bridge

#### 1.1 Project Setup
- [ ] Initialize Mix project with proper deps (Nx, Axon, EXLA)
- [ ] Configure GPU support (CUDA/ROCm via EXLA)
- [ ] Set up Pythonx or Erlang Port for libmelee communication
- [ ] Create basic OTP supervision tree

#### 1.2 Data Pipeline
- [ ] Port Peppi replay parser bindings (or use Pythonx)
- [ ] Implement game state type structs (mirror slippi-ai's types.py)
- [ ] Create training data format (efficient tensor storage)
- [ ] Build dataset streaming for large replay collections

#### 1.3 Python Bridge
- [ ] Implement libmelee wrapper in Python
- [ ] Create Elixir Port for bidirectional communication
- [ ] Handle game state serialization/deserialization
- [ ] Test with Dolphin + Slippi

### Phase 2: Embeddings & Networks (Weeks 4-6)
Neural network architecture in Nx/Axon

#### 2.1 State Embeddings (Port from slippi-ai/embed.py)
- [ ] Player embedding: position, action, damage, character, etc.
- [ ] Stage embedding: ID, platform positions
- [ ] Item embedding (for Link bombs, etc.)
- [ ] One-hot and continuous embedding primitives

#### 2.2 Controller Embeddings
- [ ] Button embedding (8 legal buttons as Bernoulli)
- [ ] Stick embedding (discretized axis positions)
- [ ] Autoregressive structure for sampling

#### 2.3 Network Architecture
- [ ] Implement MLP backbone
- [ ] Implement LSTM/GRU recurrent layers
- [ ] **NEW**: Add Transformer-like temporal attention layer
  - Self-attention over recent frames
  - More efficient than pure LSTM for capturing dependencies
- [ ] Policy head with controller output
- [ ] Value head for RL

### Phase 3: Training Infrastructure (Weeks 7-10)
Imitation learning and RL training loops

#### 3.1 Imitation Learning
- [ ] Behavioral cloning loss (cross-entropy on actions)
- [ ] Value function bootstrapping
- [ ] Unroll-based training (handle frame delays)
- [ ] Wandb integration for metrics logging
- [ ] Checkpointing and model saving

#### 3.2 Reinforcement Learning
- [ ] PPO implementation in Axon
- [ ] V-trace for off-policy correction
- [ ] Reward computation (KO diff, damage ratio)
- [ ] Teacher KL regularization (stay close to imitation policy)
- [ ] Self-play infrastructure

#### 3.3 Distributed Training
- [ ] Data parallelism across GPUs
- [ ] Gradient accumulation
- [ ] Async evaluation during training

### Phase 4: Character-Specific Features (Weeks 11-14)
Tailored embeddings and rewards per character

#### 4.1 Mewtwo
- [ ] Teleport recovery state tracking
- [ ] Tail hurtbox considerations in reward
- [ ] Confusion (side-B) combo reward shaping

#### 4.2 Mr. Game & Watch
- [ ] Handle no L-cancel (simplifies action space)
- [ ] Bucket (down-B) projectile absorption tracking
- [ ] Hammer RNG awareness in value function

#### 4.3 Link
- [ ] Bomb tracking and self-damage handling
- [ ] Projectile zoning reward shaping
- [ ] Tether recovery state machine

#### 4.4 Ganondorf
- [ ] Spacing-focused reward shaping
- [ ] Combo optimization (stomp chains)
- [ ] Edge guarding emphasis

### Phase 5: Evaluation & Deployment (Weeks 15-18)
Testing, optimization, and Slippi integration

#### 5.1 Evaluation Framework
- [ ] CPU opponent benchmarks (Level 9)
- [ ] Self-play ELO tracking
- [ ] Human evaluation sessions
- [ ] Matchup-specific metrics

#### 5.2 Slippi Online Integration
- [ ] Handle 18+ frame delay properly
- [ ] Buffer donation support
- [ ] Stable netplay performance (< 2ms inference)

#### 5.3 Optimization
- [ ] Inference optimization for real-time play
- [ ] Model quantization (INT8 if needed)
- [ ] Frame skip strategies for slower hardware

## Key Improvements Over slippi-ai

### 1. Modern Architecture (Decision Transformer-inspired)
```elixir
# Instead of pure LSTM, use temporal attention
defmodule ExPhil.Networks.TemporalTransformer do
  # Self-attention over recent N frames
  # Better at capturing long-range dependencies
  # More parallelizable than LSTM during training
end
```

### 2. Character-Specific Modules
```elixir
# Character-aware embeddings and rewards
defmodule ExPhil.Embeddings.Character.Mewtwo do
  # Mewtwo-specific state features:
  # - Tail hurtbox extension
  # - Teleport charge state
  # - Shadow Ball charge level
end
```

### 3. Elixir Concurrency for Evaluation
```elixir
# Parallel self-play games using BEAM
defmodule ExPhil.Eval.Arena do
  use GenServer

  # Run many parallel games for faster evaluation
  # Natural fit for Elixir's concurrency model
end
```

### 4. World Model (Future Enhancement)
Inspired by Dreamer v3/v4:
- Learn environment dynamics model
- Train policy in "imagination" (faster than real rollouts)
- Better sample efficiency

## Current Research to Incorporate

### Decision Transformers
- Treat RL as sequence modeling
- Condition on desired return
- May improve sample efficiency

### Graph Neural Networks for Game State
- Model relationships between entities
- Better generalization across characters/stages

### Multi-Agent Training
- Train against diverse opponent pool
- Prevents overfitting to self-play strategies

## Dependencies (mix.exs)

```elixir
defp deps do
  [
    # ML Core
    {:nx, "~> 0.9"},
    {:axon, "~> 0.7"},
    {:exla, "~> 0.9"},  # GPU backend
    {:polaris, "~> 0.1"},  # Optimizers

    # Data
    {:explorer, "~> 0.9"},  # DataFrames
    {:rustler, "~> 0.33"},  # For Peppi bindings if needed

    # Python Interop
    {:pythonx, "~> 0.3"},  # Or use Erlport

    # Observability
    {:telemetry, "~> 1.2"},
    {:wandb, github: "..."},  # Wandb Elixir client if available

    # Dev
    {:kino, "~> 0.14", only: :dev},  # Livebook
  ]
end
```

## Getting Started

```bash
# Install dependencies
mix deps.get

# Configure GPU (in config/config.exs)
# config :exla, :clients, default: [platform: :cuda]

# Download replay dataset
mix exphil.download_replays

# Parse replays to training format
mix exphil.parse_replays --input ./replays --output ./parsed

# Start imitation training
mix exphil.train --mode imitation --character mewtwo

# Continue with RL
mix exphil.train --mode rl --checkpoint ./checkpoints/latest.axon
```

## Open Questions

1. **Python Bridge Strategy**: Port vs Pythonx vs NIFs?
   - Port: Simple but serialization overhead
   - Pythonx: In-process Python, lower overhead
   - NIFs: Maximum performance but complex

2. **Replay Parser**: Use Peppi directly or port to Elixir?
   - Peppi is Rust, could use Rustler
   - Or use Pythonx to call py-slippi

3. **Distributed Training**:
   - Single node multi-GPU first
   - Consider libcluster for multi-node later

4. **Inference Latency Target**:
   - Need < 2ms for smooth 60 FPS play
   - May need INT8 quantization or model distillation

## References

- [slippi-ai](https://github.com/vladfi1/slippi-ai) - Primary reference
- [libmelee](https://github.com/altf4/libmelee) - Game interface
- [Decision Transformer](https://arxiv.org/abs/2106.01345) - Sequence modeling for RL
- [DreamerV3](https://danijar.com/project/dreamerv3/) - World model RL
- [Nx](https://github.com/elixir-nx/nx) - Numerical Elixir
- [Axon](https://github.com/elixir-nx/axon) - Neural networks for Elixir
