# slippi-ai Case Study

**Repository**: https://github.com/vladfi1/slippi-ai
**Author**: vladfi1 (@x_pilot on Twitch)
**Status**: Active (primary reference implementation)
**Language**: Python 3.10+ / TensorFlow 2.x / Sonnet

## Overview

slippi-ai (Phillip II) is vladfi1's successor to the original Phillip bot. It combines behavioral cloning from Slippi replays with PPO reinforcement learning refinement. This two-stage approach achieves competitive play against top professionals with human-like technique.

**Key Achievement**: Played top competitive players (Zain, Amsa, Moky, Aklo) via Twitch netplay with 18+ frame online delay, winning majority of games.

## Architecture

```
┌─────────────────────────────────────────┐
│         Slippi Replays (.slp)           │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│    Parsing & Preprocessing Layer        │
│  - parse_peppi: Read .slp files         │
│  - preprocessing: Validate replays      │
│  - PyArrow: Efficient storage (Parquet) │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│    Game State Embedding Layer           │
│  - Player embedding (484+ dims)         │
│  - Game state embedding (1200+ dims)    │
│  - Controller embedding (13 dims)       │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│    Policy Network Architectures         │
│  - MLP, LSTM, GRU, ResNet, Transformer │
│  - Recurrent: hidden state management   │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│    Controller Head (Autoregressive)     │
│  Samples: buttons → main stick → c-stick│
│  Independent or sequential sampling     │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│    Training Pipeline                    │
│  - Imitation: Cross-entropy on BC      │
│  - RL: PPO with KL divergence teacher  │
│  - Value function for advantage calc    │
└─────────────────────────────────────────┘
```

## State Embedding

### Player Embedding (per player, per frame)

**Base dimensions**: 440 dims (without speeds)

| Feature | Dims | Notes |
|---------|------|-------|
| `percent` | 1 | Scaled 0.01 |
| `facing` | 1 | Boolean -1/+1 |
| `x, y position` | 2 | Scaled 0.05 |
| `action` | 399 | One-hot (0x18F max action states) |
| `character` | 33 | One-hot (0x21 characters) |
| `invulnerable` | 1 | Boolean |
| `jumps_left` | 7 | One-hot (0-6 jumps) |
| `shield_strength` | 1 | Scaled 0.01 |
| `on_ground` | 1 | Boolean |

**Optional additions**:
- Speed fields (5 dims): air_x_self, ground_x_self, y_self, x_attack, y_attack
- Nana (Ice Climbers): 39 dims with `exists` boolean

**Total Player Embedding**:
- Base: 440 dims
- With speeds: 445 dims
- With Nana: 479 dims
- With Nana + speeds: 484 dims

### Game State Embedding

| Component | Dims | Notes |
|-----------|------|-------|
| p0 player | 484 | Full config |
| p1 player | 484 | Full config |
| stage | 64 | One-hot |
| Randall (Yoshi's) | 2 | x, y position |
| FoD platforms | 2 | left, right height |
| Items (15 slots) | 480 | MLP-compressed (32 dims/item) |

**Total**: ~1200 dims with all features

### Controller Embedding

| Component | Dims | Notes |
|-----------|------|-------|
| buttons | 8 | Independent Bernoulli (A, B, X, Y, Z, L, R, D_UP) |
| main_stick | 2 | x, y continuous or discretized |
| c_stick | 2 | x, y |
| shoulder | 1 | L/R trigger |

**Total**: 13 dims continuous, ~20 dims discretized

## Network Architectures

### Available Architectures

| Architecture | Config | Use Case |
|--------------|--------|----------|
| **MLP** | `mlp` | Baseline, single-frame |
| **LSTM** | `lstm` | Standard recurrent |
| **GRU** | `gru` | Lightweight recurrent |
| **DeepResLSTM** | `res_lstm` | Deep LSTM with residuals |
| **TransformerLike** | `tx_like` | Temporal attention (recommended) |

### TransformerLike Architecture (Recommended)

```
Input (embedded state-action)
    ↓
Linear encoder → hidden_size dims
    ↓
[For each layer]:
    Recurrent layer (LSTM/GRU) with residual
        ↓
    FFW residual block (hidden_size → 4×hidden_size → hidden_size)
    ↓
Output (final hidden state)
```

**Typical config**:
```bash
--config.network.name=tx_like \
--config.network.tx_like.num_layers=3 \
--config.network.tx_like.hidden_size=512 \
--config.network.tx_like.ffw_multiplier=2 \
--config.network.tx_like.recurrent_layer=lstm \
--config.network.tx_like.activation=gelu
```

## Controller Head

### Autoregressive Sampling (Recommended)

Rather than modeling all control components independently, the network predicts sequentially:

```
Frame t-1 controller state → (fed back as context)
    ↓
1. Generate buttons (8 independent Bernoulli)
    ↓
2. Generate main_stick X given buttons
    ↓
3. Generate main_stick Y given buttons + main_x
    ↓
4. Generate c_stick X given all prior
    ↓
5. Generate c_stick Y given all prior
    ↓
6. Generate shoulder given all prior
```

**Benefits**:
- Captures correlations (e.g., button + stick combos for wavedashing)
- Significantly better accuracy than independent sampling
- Each component's embedding feeds into residual block for next

## Training Pipeline

### Stage 1: Behavioral Cloning

**Entry Point**: `scripts/train.py` → `train_lib.py`

```
Training loop:
1. Load replays via DatasetConfig
2. Create name map (top N player names for conditioning)
3. Embed all states offline into tensors
4. Sample batch (T, B) frames [T+1 time steps, B batch size]
5. Network unroll: compute hidden states
6. Policy head: compute logits for actions
7. Compute BC loss: cross-entropy on buttons, MSE on sticks
8. Backprop through network + heads
9. Log metrics to wandb
```

**Example command**:
```bash
python scripts/train.py \
  --config.tag=fox_delay_18 \
  --config.data.batch_size=512 \
  --config.data.unroll_length=80 \
  --config.learner.learning_rate=1e-4 \
  --config.network.name=tx_like \
  --config.network.tx_like.num_layers=3 \
  --config.dataset.allowed_characters=fox \
  --wandb.mode=online
```

### Stage 2: PPO Self-Play

**Entry Point**: `slippi_ai/rl/run.py`

```
Teacher (BC-trained policy) [frozen]
    ↓
Actor Process(es):
  - Parallel Dolphin environments
  - Run agents using policy network
  - Collect trajectories
    ↓
Rollouts (T frames × B batch):
  - States, actions, rewards, resets
    ↓
Learner Process:
  - Compute advantages using value function
  - PPO loss with KL divergence to teacher
  - Update policy + value function
    ↓
Sync policy back to actors
```

**PPO Config**:
```python
@dataclasses.dataclass
class PPOConfig:
    num_epochs: int = 1      # Reuse trajectories N times
    num_batches: int = 1     # Mini-batches per epoch
    epsilon: float = 1e-2    # PPO clip epsilon
    beta: float = 0          # Entropy coefficient
    max_mean_actor_kl: float = 1e-4  # KL divergence cap
```

**Example command**:
```bash
python slippi_ai/rl/run.py \
  --config.learner.learning_rate=3e-5 \
  --config.learner.kl_teacher_weight=3e-3 \
  --config.learner.ppo.num_epochs=2 \
  --config.actor.rollout_length=240 \
  --config.actor.num_envs=96 \
  --config.opponent.type=self \
  --config.teacher=models/fox_d18_imitation
```

## Reward System

```python
@dataclasses.dataclass
class RewardConfig:
    damage_ratio: float = 0.01       # Damage % weight vs KO
    ledge_grab_penalty: float = 0    # Bad ledge grabs
    approaching_factor: float = 0    # Approach reward
    stalling_penalty: float = 0      # Offstage stalling
    stalling_threshold: float = 20   # Offstage distance
    nana_ratio: float = 0.5          # IC Nana weight
```

**Reward computation**:
- KO difference: +1 opponent death, -1 self death
- Damage: `damage_ratio × damage_dealt`
- Ledge penalty: Bad defensive ledge grabs
- Nana: Apply same logic to Ice Climbers partner

## Frame Delay Handling

**Critical for online play**:
- Melee actions have ~1 frame latency input → execution
- Online (netplay) adds 18+ frames additional delay

**Solution**: Delay policy actions by N frames during training
- `policy.delay`: 0 for offline, 18 for online
- Buffer stores additional frames
- Value function operates on non-delayed state

```python
# Policy applies delay internally
delayed_action = action_buffer[frame - delay]
value = value_function(current_state)  # No delay
```

## Key Technical Details

### Dynamic Shape Handling
- Unroll length variable but bounded (64-240 frames typical)
- Batch size fixed during training
- Time-major format: (T, B, features)

### State Reset
- Episodes end when either player dies
- Value bootstrap: use final predicted value for returns
- `is_resetting` mask: zero out advantages across boundaries
- Recurrent state reset on new episodes

### Player Name Conditioning
- Model optionally conditions on player identity
- Top N frequent players embedded one-hot
- Allows learning player-specific styles

### Observation Filtering
- Character-specific action ambiguities handled
- Fox/Falco/Sheik tech actions indistinguishable first N frames
- Masks to most specific determinable action

## Performance

**Inference Speed**:
- Single-frame MLP: ~milliseconds
- Batch (512 frames): ~0.5-2s depending on network
- Real-time requirement: <16ms for 60 FPS

**Training Speed**:
- BC full dataset: Days to weeks
- Typical: 3080Ti trains 6-7 days to convergence
- RL: Runs indefinitely, quality improves over time

**Memory**:
- Replay dataset: ~1-100 GB
- Model parameters: ~5-50M
- Training batch: 50-100MB

## Key Files

| File | Purpose |
|------|---------|
| `embed.py` | State embeddings (688 lines) |
| `networks.py` | Architectures (449 lines) |
| `controller_heads.py` | Action generation (200 lines) |
| `policies.py` | Policy wrapper (300+ lines) |
| `learner.py` | BC trainer (128 lines) |
| `data.py` | Data loading (600+ lines) |
| `observations.py` | Observation filtering (150+ lines) |
| `reward.py` | Reward computation (200+ lines) |
| `dolphin.py` | Console interface (400+ lines) |
| `train_lib.py` | BC entry point (500+ lines) |
| `rl/run_lib.py` | RL entry point (600+ lines) |

## Lessons for ExPhil

### Applicable Concepts

1. **Embedding Strategy**: Clean abstraction maps well to Elixir structs
2. **Autoregressive Sampling**: Core insight applicable to any framework
3. **Temporal Handling**: Sonnet RNN interface → Axon/Nx temporal ops
4. **Name Conditioning**: Optional feature, easy to add/remove
5. **Data Format**: PyArrow → Elixir, Parquet still usable
6. **Mirror Augmentation**: Trivial in Elixir pattern matching
7. **Delay Handling**: Requires careful state management
8. **Value Function**: Can be separate or multi-head output

### What to Adopt

- Two-stage BC → RL pipeline
- Autoregressive controller heads
- KL regularization to teacher during RL
- Comprehensive embedding with optional features

### What to Improve

- Mamba backbone for O(n) vs O(n²) attention
- Elixir/BEAM for concurrent self-play
- ONNX export for deployment flexibility

## References

- [Repository](https://github.com/vladfi1/slippi-ai)
- [x_pilot Twitch](https://twitch.tv/x_pilot) - Live matches vs professionals
- [Phillip Paper](https://arxiv.org/abs/1702.06230) - Original research
