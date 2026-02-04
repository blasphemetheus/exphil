# ExPhil Architecture

## Overview

ExPhil reimplements slippi-ai's architecture in Elixir, with modern enhancements including learned embeddings (6x dimension reduction), 15 temporal backbones (Mamba, Attention, LSTM, Liquid, RWKV, and more), and async inference for real-time play. This document details the technical architecture and key design decisions.

**Key architectural choices:**
- **Default 287 dims** (learned embeddings) vs legacy 1204 dims (one-hot) - enables 6x larger networks
- **15 backbone architectures**: See [Architecture Guide](architectures/ARCHITECTURE_GUIDE.md) for details
- **AsyncRunner**: Decouples 60fps frame reading from variable-latency inference
- **Self-play infrastructure**: GenServer-based with Elo matchmaking and opponent pools

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              ExPhil System                                    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         Training Pipeline                               │ │
│  │                                                                        │ │
│  │   ┌──────────┐    ┌──────────────┐    ┌──────────────┐                │ │
│  │   │  Replay  │───→│   Parser     │───→│  Embedding   │                │ │
│  │   │  Files   │    │ (Peppi/SLP)  │    │   Cache      │                │ │
│  │   │  (.slp)  │    │              │    │  (2-3x fast) │                │ │
│  │   └──────────┘    └──────────────┘    └──────┬───────┘                │ │
│  │                                              │                         │ │
│  │                    ┌─────────────────────────┴──────────────────┐      │ │
│  │                    │                                            │      │ │
│  │                    ▼                                            ▼      │ │
│  │   ┌────────────────────────────┐    ┌────────────────────────────┐    │ │
│  │   │    Imitation Learning      │    │    Reinforcement Learning   │    │ │
│  │   │                            │    │                             │    │ │
│  │   │  • Behavioral Cloning      │    │  • PPO with Clipped Obj     │    │ │
│  │   │  • Focal Loss (rare acts)  │    │  • Self-Play GenServer      │    │ │
│  │   │  • Embedding Caching       │    │  • Elo Matchmaking          │    │ │
│  │   │  • Data Augmentation       │    │  • Opponent Pool Sampling   │    │ │
│  │   └─────────────┬──────────────┘    └──────────────┬──────────────┘    │ │
│  │                 │                                   │                  │ │
│  │                 └───────────────┬───────────────────┘                  │ │
│  │                                 │                                      │ │
│  │                                 ▼                                      │ │
│  │                    ┌───────────────────────┐                          │ │
│  │                    │    Trained Model      │                          │ │
│  │                    │    (.axon checkpoint) │                          │ │
│  │                    └───────────┬───────────┘                          │ │
│  │                                │                                      │ │
│  └────────────────────────────────┼──────────────────────────────────────┘ │
│                                   │                                        │
│  ┌────────────────────────────────┼──────────────────────────────────────┐ │
│  │                    Inference Pipeline                                  │ │
│  │                                │                                      │ │
│  │                                ▼                                      │ │
│  │   ┌────────────────────────────────────────────────────────────────┐ │ │
│  │   │                      AsyncRunner                                │ │ │
│  │   │                                                                 │ │ │
│  │   │  ┌─────────────┐   ┌────────────┐   ┌─────────────────────┐   │ │ │
│  │   │  │ FrameLoop   │   │  SharedETS │   │  InferenceLoop      │   │ │ │
│  │   │  │ (60 FPS)    │──→│  (state)   │←──│  (async)            │   │ │ │
│  │   │  │             │   │            │   │                     │   │ │ │
│  │   │  └─────────────┘   └────────────┘   └──────────┬──────────┘   │ │ │
│  │   │                                                │              │ │ │
│  │   └────────────────────────────────────────────────┼──────────────┘ │ │
│  │                                                    │                │ │
│  │                           ┌────────────────────────┘                │ │
│  │                           │                                         │ │
│  │                           ▼                                         │ │
│  │   ┌────────────────────────────────────────────────────────────┐   │ │
│  │   │                  MeleePort (GenServer)                      │   │ │
│  │   │                                                             │   │ │
│  │   │   ┌─────────────┐        ┌────────────────────────────┐    │   │ │
│  │   │   │   Elixir    │◀──────▶│   libmelee (Python)        │    │   │ │
│  │   │   │   Port      │  JSON  │   melee_bridge.py          │    │   │ │
│  │   │   │             │        │                            │    │   │ │
│  │   │   └─────────────┘        └─────────────┬──────────────┘    │   │ │
│  │   │                                        │                   │   │ │
│  │   └────────────────────────────────────────┼───────────────────┘   │ │
│  │                                            │                       │ │
│  └────────────────────────────────────────────┼───────────────────────┘ │
│                                               │                         │
└───────────────────────────────────────────────┼─────────────────────────┘
                                                │
                                                ▼
                                   ┌────────────────────────┐
                                   │   Dolphin + Slippi     │
                                   │   (Game Emulator)      │
                                   └────────────────────────┘
```

## Neural Network Architecture

### Backbone Comparison

| Backbone | Inference | 60 FPS Ready | Val Loss | Memory | Best For |
|----------|-----------|--------------|----------|--------|----------|
| MLP | 2-5ms | Yes | 3.11 | 50MB | Fast iteration, baseline |
| LSTM | 220ms | No | **2.95** | 500MB | Best accuracy (offline only) |
| GRU | ~150ms | No | ~3.0 | 400MB | Faster recurrent alternative |
| Mamba | 8.9ms | **Yes** | 3.00 | 800MB | Real-time temporal |
| Attention | 17ms | Borderline | 3.07 | 2.5GB | Long-range patterns |
| Jamba | ~20ms | Borderline | 3.0 | 1.2GB | Hybrid approach |

**Recommended:** Mamba for real-time play (8.9ms inference, 60 FPS capable). ONNX INT8 export achieves 0.55ms.

### Embedding Dimensions

ExPhil uses two embedding modes:

| Mode | Total Dims | Per Player | Stage | Controller | Notes |
|------|------------|------------|-------|------------|-------|
| **Learned (default)** | 287 | 56 | 7 | 13 | 6x smaller, modern |
| One-hot (legacy) | 1204 | 488 | 64 | 13 | slippi-ai compatible |

**Learned embedding breakdown (287 dims):**
```
Player 0:           56 dims (8 base + 9 optional + 39 Nana compact)
Player 1:           56 dims
Stage (compact):    7 dims (6 competitive + "other")
Prev action:        13 dims (8 buttons + 4 sticks + 1 shoulder)
Player names:       112 dims (tag identification)
Spatial:            4 dims (distance, relative pos, frame)
Projectiles:        35 dims (5 slots × 7 dims)
Action IDs:         2 dims (appended for network embedding)
Character IDs:      2 dims (appended for network embedding)
Padding:            1 dim (alignment to 8 for tensor cores)
─────────────────────────────────────────────────────────
Total:              288 dims (287 raw + 1 padding)
```

### Network Structure

```
Input: Game State (t-N to t) or single frame
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│                    State Embedding Layer                     │
│                                                              │
│  Player0: [pos, facing, percent, jumps, shield, speeds, ...] │
│  Player1: [same as Player0]                                  │
│  Nana:    [compact 39 dims or enhanced 14 dims + action ID]  │
│  Stage:   [compact 7-dim or learned ID]                      │
│  Prev:    [buttons(8), sticks(4), shoulder(1)]               │
│                                                              │
│  Learned IDs appended: [action_p0, action_p1, char_p0, ...]  │
│                                                              │
│  Total: 287 dims (learned) or 1204 dims (one-hot)           │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backbone Network                          │
│                                                              │
│  MLP:       Dense → ReLU → Dropout (single-frame)           │
│  LSTM/GRU:  Recurrent → LayerNorm (sequential)              │
│  Mamba:     Selective SSM with parallel scan (O(L))         │
│  Attention: Multi-head self-attention (O(L²))               │
│  Jamba:     Mamba blocks + Attention every N layers         │
│                                                              │
│  Learned embeddings: IDs extracted, embedded via lookup,    │
│  then concatenated with continuous features                  │
│                                                              │
│  Output: 256-512 dimensional hidden state                    │
└─────────────────────────────────────────────────────────────┘
           │
           ├──────────────────────┐
           ▼                      ▼
┌─────────────────────┐  ┌─────────────────────┐
│    Policy Head      │  │    Value Head       │
│                     │  │                     │
│  Autoregressive:    │  │  Dense(hidden, 128) │
│  1. Buttons (8)     │  │  ReLU               │
│  2. Main X (17)     │  │  Dense(128, 1)      │
│  3. Main Y (17)     │  │                     │
│  4. C-Stick X (17)  │  │  Output: V(s)       │
│  5. C-Stick Y (17)  │  │                     │
│  6. Shoulder (5)    │  └─────────────────────┘
│                     │
│  Each conditioned   │
│  on previous        │
└─────────────────────┘
```

### Autoregressive Controller Head

The policy outputs 6 heads sequentially, each conditioned on previous samples:

```elixir
# Sampling sequence:
# 1. Sample 8 buttons (independent Bernoulli, sigmoid logits)
# 2. Concat buttons embedding, sample main_x (17-way categorical)
# 3. Concat main_x embedding, sample main_y
# 4. Concat main_y embedding, sample c_x
# 5. Concat c_x embedding, sample c_y
# 6. Concat c_y embedding, sample shoulder (5-way categorical)

# Output sizes:
# - Buttons: 8 logits (binary per button: A, B, X, Y, Z, L, R, D_UP)
# - Stick axes: 17 values each (discretized -1.0 to +1.0)
# - Shoulder: 5 values (discretized 0.0 to 1.0)
```

## Training Pipeline

### Phase 1: Imitation Learning (Behavioral Cloning)

Learn to mimic human play from Slippi replay data:

```
Loss = Button_BCE + Stick_CE + Shoulder_CE

With enhancements:
  - Focal loss: (1 - p_t)^γ × CE  (γ=2.0 focuses on hard examples)
  - Label smoothing: Prevents overconfidence
  - K-means discretization: ~5% better stick accuracy
```

**Key features:**
- **Embedding caching**: 2-3x speedup by precomputing embeddings
- **Augmented caching**: ~100x speedup with `--cache-augmented --augment`
- **Frame delay training**: `--online-robust` for Slippi online (18+ frame delay)
- **Early stopping**: Monitor validation loss with patience

### Phase 2: Reinforcement Learning (PPO)

Improve beyond human level through self-play:

```
L_total = -L_clip + c1 × L_value + c2 × entropy

Where:
  L_clip = min(r × A, clip(r, 1±ε) × A)  # Clipped surrogate
  L_value = (V - V_target)²               # Value loss
  entropy = H(π)                          # Exploration bonus
```

**Self-play infrastructure (complete):**
- `lib/exphil/self_play/supervisor.ex` - Top-level supervisor
- `lib/exphil/self_play/game_runner.ex` - Per-game GenServer
- `lib/exphil/self_play/population_manager.ex` - Policy versioning
- `lib/exphil/self_play/matchmaker.ex` - Elo ratings
- `lib/exphil/self_play/experience_collector.ex` - Batched experience

## Reward Design

### Standard Rewards

```elixir
# Primary: Stock differential (+1 KO, -1 death)
# Secondary: Damage differential (0.01 × net damage)
reward = stock_diff + 0.01 × (damage_dealt - damage_taken)
```

### Shaped Rewards

| Reward | Formula | Range | Purpose |
|--------|---------|-------|---------|
| Approach | (prev_dist - curr_dist) / 3 | -1 to +1 | Encourage engagement |
| Combo | min(hitstun/30, 1) | 0 to 1 | Reward combos |
| Edge guard | +1.0 | 0 or 1 | Punish recovery |
| Recovery risk | horizontal + vertical | 0 to 2 | Penalize offstage |

**Weights (default):**
- Stock: 1.0 (primary signal)
- Damage: 0.01 (frame-by-frame)
- Approach: 0.001 (avoid turtling)
- Combo: 0.05 (encourage offense)
- Edge guard: 0.1 (punish recovery)
- Recovery: 0.02 (weak penalty)

## Inference Architecture

### AsyncRunner (Real-Time Play)

Decouples frame reading from inference to maintain 60 FPS gameplay:

```
┌─────────────────────┐    ┌───────────────────┐    ┌──────────────────┐
│   FrameLoop         │───▶│  SharedState      │◀───│  InferenceLoop   │
│   (60fps, reads)    │    │  (ETS table)      │    │  (async, slow)   │
└─────────────────────┘    │                   │    └──────────────────┘
         │                 │  :latest_state    │             │
         │                 │  :latest_action   │             │
         │                 │  :in_game         │             │
         └────────────────▶│  :frame_count     │◀────────────┘
                           └───────────────────┘
```

**Key benefits:**
- FrameLoop never blocks waiting for inference
- InferenceLoop uses latest state, updates latest action
- ETS provides lock-free concurrent access
- Game maintains 60 FPS even with 200ms LSTM inference

### Agent State Management

```elixir
%Agent{
  policy_params: map(),        # Trained model weights
  predict_fn: fun(),           # Compiled inference function
  frame_buffer: queue(),       # Temporal window (60 frames)
  mamba_cache: map() | nil,    # Incremental Mamba state
  deterministic: boolean(),    # Argmax vs sampling
  temperature: float(),        # Softmax temperature
  action_repeat: integer()     # Cache action for N frames
}
```

### Inference Optimization

| Strategy | Speedup | Use Case |
|----------|---------|----------|
| JIT warmup | Avoids first-frame latency | Before game start |
| Incremental Mamba | 60x for Mamba | Real-time temporal |
| Action repeat | N× | Fast models |
| ONNX INT8 | ~5x | Production deployment |

## Python Bridge (MeleePort)

### Communication Protocol

- **Transport**: Erlang Port with line-delimited JSON on stdin/stdout
- **Python**: `priv/python/melee_bridge.py` (498 lines)
- **Commands**: `init`, `step`, `send_controller`, `ping`, `stop`

### Game State Structure

```elixir
%GameState{
  frame: integer(),
  stage: integer(),           # Stage ID
  menu_state: integer(),      # 2=IN_GAME
  players: %{
    1 => %Player{
      character: integer(),
      x: float(), y: float(),
      percent: float(),
      stock: integer(),
      action: integer(),      # Action state ID
      action_frame: integer(),
      jumps_left: integer(),
      on_ground: boolean(),
      nana: %Nana{} | nil     # Ice Climbers partner
    }
  },
  projectiles: [%Projectile{}],
  distance: float()
}
```

## Testing Strategy

### Test Organization (1933 tests)

| Category | Files | Coverage |
|----------|-------|----------|
| Embeddings | 7 | Continuous/discrete encoding, shapes |
| Networks | 10 | All 6 backbones, policy, value |
| Training | 37+ | Config, data, imitation, PPO |
| Self-Play | 6 | Elo, matchmaking, population |
| Integration | 2 | Full pipeline, Dolphin |

### Test Tags

- `:slow` - Tests >1s (excluded by default)
- `:integration` - External dependencies
- `:gpu` - Requires CUDA
- `:benchmark` - Performance regression
- `:snapshot` - Embedding output stability

```bash
mix test                    # Fast unit tests
mix test.slow               # Include slow tests
mix test.all                # Everything
mix test.benchmark          # Performance tests
```

## Monitoring & Observability

### Training Metrics (W&B)

- Loss: total, button, stick, shoulder
- Per-action accuracy: buttons, rare actions (Z, L, R)
- Learning rate, gradient norm
- GPU memory utilization

### Inference Metrics

- Inference latency (ms/frame)
- FPS achieved vs target (60)
- Confidence scores
- Frame buffer depth

## File Organization

```
lib/exphil/
├── networks/           # Policy, Value, all backbones
│   ├── policy.ex       # Main policy network
│   ├── mamba.ex        # Selective SSM
│   ├── attention.ex    # Multi-head attention
│   ├── recurrent.ex    # LSTM/GRU
│   └── hybrid.ex       # Jamba (Mamba+Attention)
├── embeddings/         # State → tensor conversion
│   ├── player.ex       # Player embedding
│   ├── game.ex         # Full game state
│   └── controller.ex   # Controller I/O
├── training/           # Training infrastructure
│   ├── imitation.ex    # Behavioral cloning
│   ├── ppo.ex          # PPO algorithm
│   ├── data.ex         # Dataset handling
│   └── config.ex       # CLI parsing
├── agents/             # Inference agents
│   ├── agent.ex        # Policy inference
│   └── supervisor.ex   # Agent management
├── bridge/             # Dolphin integration
│   ├── melee_port.ex   # Python bridge
│   └── async_runner.ex # Real-time play
├── self_play/          # RL infrastructure
│   ├── supervisor.ex   # Self-play supervisor
│   ├── game_runner.ex  # Per-game GenServer
│   └── matchmaker.ex   # Elo system
└── rewards/            # Reward computation
    ├── standard.ex     # Stock/damage rewards
    └── shaped.ex       # Approach/combo/edge
```

## References

- [slippi-ai](https://github.com/vladfi1/slippi-ai) - Original TensorFlow implementation
- [libmelee](https://github.com/altf4/libmelee) - Python game state API
- [Mamba paper](https://arxiv.org/abs/2312.00752) - Selective state space models
- [Nx](https://github.com/elixir-nx/nx) / [Axon](https://github.com/elixir-nx/axon) - Elixir ML
