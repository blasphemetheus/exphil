# ExPhil - Elixir Phil

ExPhil is an Elixir-based successor to slippi-ai, designed to create high-ELO playable bots for lower-tier Melee characters.

## Target Characters (Initial Focus)
1. **Mewtwo** - Unique physics, teleport recovery, tail hurtbox mechanics
2. **Mr. Game & Watch** - No L-cancel, random hammer, unique shield
3. **Link** - Projectiles, tether recovery, bomb tech
4. **Ganondorf** - Slow but powerful, requires precise spacing/timing
5. **Zelda** - Transform mechanic, Din's Fire zoning, Lightning Kick spacing

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
│   │   │   ├── agent.ex            # GenServer holding trained policy
│   │   │   └── supervisor.ex       # DynamicSupervisor for agents
│   │   ├── bridge/                 # Game communication
│   │   │   ├── async_runner.ex     # Async frame reader + inference
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
│       ├── replay_converter.py     # Convert .slp to training format
│       └── quantize_onnx.py        # INT8 quantization for ONNX models
├── test/
├── config/
│   ├── config.exs
│   ├── dev.exs
│   └── prod.exs
├── scripts/
│   ├── train_from_replays.exs      # Imitation learning script
│   ├── train_ppo.exs               # PPO fine-tuning script
│   ├── play_dolphin.exs            # Sync gameplay (fast MLP)
│   ├── play_dolphin_async.exs      # Async gameplay (slow LSTM)
│   ├── eval_model.exs              # Model evaluation
│   ├── export_onnx.exs             # Export to ONNX (needs axon_onnx)
│   └── export_numpy.exs            # Export weights to NumPy format
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
- [x] Initialize Mix project with proper deps (Nx, Axon, EXLA)
- [x] Configure EXLA backend (CPU w/ XLA optimizations, ~2000x faster)
- [x] Set up Pythonx or Erlang Port for libmelee communication
- [x] Create basic OTP supervision tree

#### 1.2 Data Pipeline
- [x] Port Peppi replay parser bindings (or use Pythonx)
- [x] Implement game state type structs (mirror slippi-ai's types.py)
- [x] Create training data format (efficient tensor storage)
- [x] Build dataset streaming for large replay collections

#### 1.3 Python Bridge
- [x] Implement libmelee wrapper in Python
- [x] Create Elixir Port for bidirectional communication
- [x] Handle game state serialization/deserialization
- [x] Test with Dolphin + Slippi (async runner working!)

### Phase 2: Embeddings & Networks (Weeks 4-6)
Neural network architecture in Nx/Axon

#### 2.1 State Embeddings (Port from slippi-ai/embed.py)
- [x] Player embedding: position, action, damage, character, etc.
- [x] Stage embedding: ID, platform positions
- [x] Item embedding (Link bombs, Peach turnips, Mr. Saturn, Bob-ombs)
- [x] One-hot and continuous embedding primitives

#### 2.2 Controller Embeddings
- [x] Button embedding (8 legal buttons as Bernoulli)
- [x] Stick embedding (discretized axis positions)
- [x] Autoregressive structure for sampling

#### 2.3 Network Architecture
- [x] Implement MLP backbone
- [x] Implement LSTM/GRU recurrent layers (`ExPhil.Networks.Recurrent`)
- [x] Transformer-like temporal attention (`ExPhil.Networks.Attention`)
  - Sliding window attention (O(K²) efficient)
  - Hybrid LSTM + attention architecture
  - Sinusoidal positional encoding
- [x] Policy head with controller output
- [x] Value head for RL
- [x] Temporal policy (`Policy.build_temporal/1`) integrating attention

### Phase 3: Training Infrastructure (Weeks 7-10)
Imitation learning and RL training loops

#### 3.1 Imitation Learning
- [x] Behavioral cloning loss (cross-entropy on actions)
- [ ] Value function bootstrapping
- [x] Temporal/sequence training with attention backbones
- [x] Wandb integration for metrics logging
- [x] Checkpointing and model saving

#### 3.2 Reinforcement Learning
- [x] PPO implementation in Axon
- [x] PPO training script (`scripts/train_ppo.exs`)
- [ ] V-trace for off-policy correction
- [x] Reward computation (KO diff, damage ratio)
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

#### 4.5 Zelda
- [ ] Transform state machine (Zelda ↔ Sheik)
- [ ] Din's Fire zoning and tracking
- [ ] Lightning Kick sweetspot spacing rewards

### Phase 5: Evaluation & Deployment (Weeks 15-18)
Testing, optimization, and Slippi integration

#### 5.1 Evaluation Framework
- [ ] CPU opponent benchmarks (Level 9)
- [ ] Self-play ELO tracking
- [ ] Human evaluation sessions
- [ ] Matchup-specific metrics

#### 5.2 Slippi Online Integration
- [x] Handle 18+ frame delay properly (via `--frame-delay` training option)
- [ ] Buffer donation support
- [x] Stable netplay performance (< 2ms inference via Mamba or ONNX INT8)

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

## Implementation Notes

### Current Status (January 2025)

**Completed modules:**
- `ExPhil.Networks.Policy` - 6-head autoregressive policy + temporal variants
- `ExPhil.Networks.Value` - Value function with GAE computation
- `ExPhil.Networks.ActorCritic` - Combined actor-critic with PPO loss
- `ExPhil.Networks.Recurrent` - LSTM/GRU layers for temporal processing
- `ExPhil.Networks.Attention` - Sliding window & hybrid attention mechanisms
- `ExPhil.Embeddings.Primitives` - One-hot, float, bool embedding utilities
- `ExPhil.Embeddings.Player` - Player state embedding (446 dims base)
- `ExPhil.Embeddings.Game` - Full game state embedding (~1991 dims)
- `ExPhil.Embeddings.Controller` - Controller action embedding
- `ExPhil.Training.Imitation` - Behavioral cloning trainer (single-frame + temporal)
- `ExPhil.Training.Data` - Dataset batching with sequence support
- `ExPhil.Training.PPO` - PPO trainer with clipped objective
- `ExPhil.Rewards` - Reward computation (damage, KO, combo, recovery)
- `ExPhil.Data.ReplayParser` - Slippi replay parsing via py-slippi
- `ExPhil.Data.Dataset` - Dataset management and batching
- `ExPhil.Bridge.*` - Game state structs (GameState, Player, ControllerState, etc.)
- `ExPhil.Bridge.Supervisor` - DynamicSupervisor for MeleePort bridge processes
- `ExPhil.Agents.Supervisor` - DynamicSupervisor for inference agents
- `ExPhil.Agents.Agent` - GenServer holding trained policy for inference (single-frame + temporal)
- `ExPhil.Agents` - High-level facade for agent management
- `ExPhil.Telemetry` - Telemetry events and metrics collector
- `ExPhil.Integrations.Wandb` - Weights & Biases experiment tracking

**Test coverage:** 580 tests passing

### Next Steps (Priority Order)

#### Recently Completed
- [x] **Mamba backbone** - 24.75x faster than LSTM, achieves <16ms for 60 FPS
- [x] **ONNX export + INT8 quantization** - `sequence_last` layer, full pipeline working
- [x] **Temporal training infrastructure** - LSTM/attention backbones working
- [x] **Batch Nx embedding optimization** - 20x faster preprocessing
- [x] **Truncated BPTT option** - `--truncate-bptt N` for 2-3x faster training
- [x] **XLA multi-threading** - Auto-enabled for CPU training
- [x] **O(1) batch lookup** - `:array` instead of `Enum.at`
- [x] **Async game runner** - Decouple frame reading from slow LSTM inference
- [x] **Dolphin integration** - Full gameplay loop working with auto-restart

#### Immediate: Inference Optimization

Current bottleneck: LSTM inference ~500ms on CPU (need <16ms for 60fps).

**1. Architecture Simplification** (Quick wins, low effort)
| Change | Expected Speedup | Status |
|--------|------------------|--------|
| Reduce hidden size (256→64) | 4-16x | [ ] TODO |
| Reduce window size (60→20) | ~3x | [ ] TODO |
| Use GRU instead of LSTM | 1.3x | [ ] TODO |
| Single layer vs stacked | 2x | [ ] TODO |

```bash
# Train smaller LSTM
mix run scripts/train_from_replays.exs \
  --temporal --backbone lstm \
  --hidden 64 --window-size 20 \
  --epochs 5 --max-files 20
```

**2. Knowledge Distillation** (Best accuracy/speed tradeoff) - IMPLEMENTED
- [x] Generate soft labels from trained teacher model
- [x] Train small MLP student to match teacher distributions
- [x] Expected: MLP with LSTM-like behavior, <5ms inference

**Step 1: Generate soft labels from teacher model:**
```bash
mix run scripts/generate_soft_labels.exs \
  --teacher checkpoints/mamba_policy.bin \
  --replays replays \
  --output soft_labels.bin \
  --temperature 2.0 \
  --max-files 50
```

**Step 2: Train distilled MLP student:**
```bash
mix run scripts/train_distillation.exs \
  --soft-labels soft_labels.bin \
  --hidden 64,64 \
  --epochs 10 \
  --alpha 0.7 \
  --output distilled_policy.bin
```

**Distillation options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--hidden` | 64,64 | Student MLP hidden sizes |
| `--temperature` | 2.0 | Soft label temperature (higher = softer) |
| `--alpha` | 0.7 | Weight for soft vs hard labels |
| `--epochs` | 10 | Training epochs |

**Expected results by hidden size:**
| Hidden | Parameters | Inference | Accuracy |
|--------|------------|-----------|----------|
| 32,32 | ~70K | ~1ms | Good |
| 64,64 | ~270K | ~2ms | Better |
| 128,64 | ~400K | ~3ms | Best |

**3. BF16 Training** (2x speedup, minimal accuracy loss) - IMPLEMENTED
- [x] Add `--precision` option to training script (default: bf16)
- [x] Update Imitation trainer to use configurable precision
- [ ] Benchmark actual speedup on your hardware

```bash
# BF16 is now default - just run training normally
mix run scripts/train_from_replays.exs --temporal --backbone lstm

# To use F32 (full precision) instead:
mix run scripts/train_from_replays.exs --precision f32 --temporal --backbone lstm
```

**4. GPU Acceleration** (10-100x speedup)
- [ ] Install CUDA toolkit and cuDNN
- [ ] Configure EXLA with CUDA backend
- [ ] Expected: LSTM inference <10ms on GPU

```elixir
# In config/config.exs
config :exla, :clients,
  cuda: [platform: :cuda, memory_fraction: 0.8],
  default: [platform: :host]
config :exla, default_client: :cuda
```

**5. Mamba/SSM Architecture** - IMPLEMENTED
- [x] Implement Mamba layer in Axon (`ExPhil.Networks.Mamba`)
- [x] 24.75x faster inference than LSTM (8.93ms vs 220.95ms)
- [x] Linear O(L) scaling with sequence length
- [x] Achieves <16ms target for 60 FPS real-time gameplay
- Reference: https://arxiv.org/abs/2312.00752

**Train with Mamba backbone:**
```bash
mix run scripts/train_from_replays.exs \
  --temporal --backbone mamba \
  --hidden 256 --window-size 60 \
  --state-size 16 --expand-factor 2 --conv-size 4 \
  --num-layers 2 --epochs 5
```

**Memory-optimized Mamba training (for constrained systems):**
```bash
mix run scripts/train_from_replays.exs \
  --temporal --backbone mamba \
  --hidden 128 --window-size 30 \
  --state-size 16 --expand-factor 2 --conv-size 4 \
  --num-layers 2 --epochs 3 --max-files 5 --batch-size 32
```

**Memory usage trade-offs:**
| Config | Embedding | Training | Notes |
|--------|-----------|----------|-------|
| Full (hidden=256, window=60, files=10) | ~10GB | ~12GB | Best accuracy |
| Medium (hidden=128, window=30, files=5) | ~5GB | ~9GB | Good balance |
| Minimal (hidden=64, window=20, files=3) | ~3GB | ~5GB | Fast iteration |

**Benchmark (ExPhil production dimensions):**
| Architecture | Inference | 60 FPS Ready |
|--------------|-----------|--------------|
| LSTM (Axon)  | 220.95 ms | No |
| Mamba (Axon) | 8.93 ms   | **Yes** |
| ONNX INT8    | 0.55 ms   | **Yes** |

**6. ONNX Export + INT8 Quantization** (Production deployment) - PARTIAL
- [x] INT8 quantization script via ONNX Runtime
- [ ] Direct ONNX export via `axon_onnx` (compiles but has runtime API issues)
- [x] NumPy export workaround for Python-side ONNX conversion
- Expected: 2-4x speedup with ~1% accuracy loss

**Current Status:** Using `blasphemetheus/axon_onnx` fork (`runtime-fixes` branch) via GitHub:
- [x] sequence_last layer support
- [x] LSTM/GRU serialization
- [x] Multi-output model support (Policy's 6-head structure)
- API mismatch with Axon 0.8+ partially resolved

**Recommended path:** Use NumPy export + Python ONNX conversion (see `scripts/export_numpy.exs`)

**Export to ONNX:**
```bash
# From checkpoint
mix run scripts/export_onnx.exs --checkpoint checkpoints/imitation_latest.axon --output policy.onnx

# From exported policy
mix run scripts/export_onnx.exs --policy checkpoints/imitation_latest_policy.bin --output policy.onnx
```

**Verify the ONNX model:**
```bash
python -c "import onnxruntime as ort; sess = ort.InferenceSession('policy.onnx'); print('OK:', [i.name for i in sess.get_inputs()])"
```

**Quantize to INT8** (2-4x faster inference):
```bash
# Dynamic quantization (no calibration needed, fast)
python priv/python/quantize_onnx.py policy.onnx policy_int8.onnx

# Static quantization (more accurate, requires calibration data)
python priv/python/quantize_onnx.py policy.onnx policy_int8.onnx --static --calibration-data calibration.npz
```

**Benchmark quantized model:**
```bash
python -c "
import onnxruntime as ort
import numpy as np
import time

sess = ort.InferenceSession('policy_int8.onnx')
inp = sess.get_inputs()[0]
shape = [1 if d is None else d for d in inp.shape]
x = np.random.randn(*shape).astype(np.float32)

# Warmup + benchmark
for _ in range(10): sess.run(None, {inp.name: x})
start = time.time()
for _ in range(100): sess.run(None, {inp.name: x})
print(f'Average: {(time.time() - start) * 10:.2f} ms/inference')
"
```

**For Elixir inference with ONNX**, add `ortex` dependency:
```elixir
{:ortex, "~> 0.1"}
```
```elixir
{:ok, model} = Ortex.load("policy_int8.onnx")
output = Ortex.run(model, input_tensor)
```

**Complete ONNX Deployment Workflow (Mamba model):**

After training a Mamba model, follow these steps to get sub-millisecond inference:

```bash
# Step 1: Training produces checkpoint and policy files
ls checkpoints/
# imitation_latest.axon (full checkpoint)
# imitation_latest_policy.bin (exported policy)

# Step 2: Export to NumPy format (workaround for axon_onnx issues)
mix run scripts/export_numpy.exs \
  --policy checkpoints/imitation_latest_policy.bin \
  --output weights.npz

# Step 3: Rebuild model in Python and export to ONNX
# (See priv/python/convert_to_onnx.py for reference implementation)
python priv/python/convert_to_onnx.py \
  --weights weights.npz \
  --config checkpoints/imitation_latest_policy.bin \
  --output policy.onnx

# Step 4: Quantize to INT8 for maximum speed
python priv/python/quantize_onnx.py policy.onnx policy_int8.onnx

# Step 5: Benchmark to verify performance
python -c "
import onnxruntime as ort
import numpy as np
import time

sess = ort.InferenceSession('policy_int8.onnx')
inp = sess.get_inputs()[0]
x = np.random.randn(1, 60, 1991).astype(np.float32)  # Mamba: [batch, seq, embed]

for _ in range(10): sess.run(None, {inp.name: x})
start = time.time()
for _ in range(100): sess.run(None, {inp.name: x})
print(f'ONNX INT8 inference: {(time.time() - start) * 10:.2f} ms')
"

# Expected: ~0.5ms per inference (200x faster than Axon LSTM)
```

**When to use ONNX vs Axon:**
| Use Case | Recommendation |
|----------|----------------|
| Training | Axon (native Elixir, easy debugging) |
| Prototyping | Axon Mamba (8.93ms, good enough) |
| Production gameplay | ONNX INT8 (0.55ms, maximum performance) |
| Integration with Python/Rust | ONNX (cross-platform) |

#### Priority Order for Optimization

1. **Mamba backbone** - Available now, 24.75x faster than LSTM, <16ms inference
2. **ONNX + INT8 quantization** - Available now, 0.55ms inference
3. **Train smaller LSTM** (hidden=64, window=20) - Quick experiment
4. **Knowledge distillation to MLP** - Best accuracy/speed tradeoff
5. **GPU acceleration** (if NVIDIA GPU available) - 1 hour setup
6. **BF16 training** - Available now (default)

#### Evaluation & Fine-tuning

1. **Evaluate temporal model** - Run the trained LSTM policy through evaluation
   - Use `notebooks/evaluation_dashboard.livemd` to visualize predictions
   - Compare action distribution to replay ground truth

2. **Larger dataset training** - Scale up with more replays
   - `--max-files 50 --epochs 5 --truncate-bptt 20` for faster iteration
   - Monitor with Wandb: `--wandb --wandb-project exphil`

3. **PPO fine-tuning in mock mode** - Test the RL training loop
   - `mix run scripts/train_ppo.exs --mock --pretrained checkpoints/imitation_latest_policy.bin`
   - Verify gradient flow and loss computation

#### Open Source Contributions

4. **Fix axon_onnx for Nx 0.10+** - PR SUBMITTED (partial fix)
   - PR: https://github.com/mortont/axon_onnx/compare/master...blasphemetheus:axon_onnx:support-nx-0.10
   - Issue: Version constraints in mix.exs were too restrictive (`~> 0.5`)
   - Fix: Widen version constraints (nx, axon, exla, req)
   - **Note:** PR enables compilation, but runtime API issues remain:
     - Axon 0.8+ changed APIs (shapes vs tensors)
     - Multi-output models and LSTM layers not supported
   - **Workaround active:** ExPhil uses `{:axon_onnx, github: "mortont/axon_onnx"}`
   - Forum thread: https://elixirforum.com/t/error-using-axononnx-v0-4-0-undefined-function-transform-2/63326

#### Other Next Steps

5. **Frame delay training** - IMPLEMENTED
   Simulates Slippi online conditions where there's 18+ frame delay between
   observing game state and your input taking effect.

   **How it works:**
   - Normal training: (state_t, action_t) pairs
   - With delay N: (state_{t-N}, action_t) pairs
   - Teaches the model to act on "stale" information, just like online play

   **Training with frame delay:**
   ```bash
   # Simulate 18-frame online delay (Slippi standard)
   mix run scripts/train_from_replays.exs \
     --temporal --backbone mamba \
     --frame-delay 18 \
     --epochs 5

   # Higher delay for worse connections
   mix run scripts/train_from_replays.exs \
     --temporal --backbone mamba \
     --frame-delay 24 \
     --epochs 5
   ```

   **Recommended workflow:**
   1. First train with `--frame-delay 0` to learn basic gameplay
   2. Fine-tune with `--frame-delay 18` for online adaptation
   3. At inference, use matching `--frame-delay 18` in play scripts

   **Frame delay values:**
   | Connection | Delay | Notes |
   |------------|-------|-------|
   | LAN/Local | 0-4 | Near-instant |
   | Good online | 12-18 | Standard Slippi |
   | Poor online | 24-36 | High latency |

6. **Self-play infrastructure** - Train agent vs agent
   - Use BEAM concurrency for parallel games
   - Prevents overfitting to human replay patterns

7. **Character-specific rewards** - Mewtwo, G&W, Link specialization

8. **Docker: Precompiled Rustler NIF** - Avoid runtime NIF compilation
    - Configure Rustler with `skip_compilation?: true` or `force_build: false` in prod
    - Or use Rustler precompilation to download prebuilt binaries
    - Currently Dockerfile includes full Rust toolchain (~500MB) as workaround
    - See: https://hexdocs.pm/rustler_precompiled/precompilation_guide.html

### Dolphin Integration Setup

To run the trained agent against Dolphin:

#### Prerequisites
1. **Slippi Dolphin** - Download from https://slippi.gg/downloads
   - AppImage location (after Slippi Launcher install): `~/.config/Slippi Launcher/netplay/Slippi_Online-x86_64.AppImage`

2. **Melee ISO** - NTSC 1.02 (Rev 2) required
   - Configure in Slippi Launcher settings

3. **Python 3.12** with venv (pyenet requires <3.13)
   ```bash
   # Install Python 3.12 via asdf
   asdf install python 3.12.12
   asdf local python 3.12.12

   # Create and activate venv
   python3 -m venv .venv
   source .venv/bin/activate

   # Install dependencies (enet system library required for pyenet)
   sudo pacman -S enet  # Arch/Manjaro
   pip install -r priv/python/requirements.txt
   ```

4. **System library**: `enet` (for pyenet networking)
   - Arch/Manjaro: `sudo pacman -S enet`
   - Ubuntu/Debian: `sudo apt install libenet-dev`

#### Running the Agent

Two scripts are available for running trained agents:

**Async Runner (Recommended for LSTM/temporal models):**
```bash
source .venv/bin/activate

# Async version - separates frame reading from inference
# Best for slow models (LSTM ~500ms inference) - game runs at 60fps
mix run scripts/play_dolphin_async.exs \
  --policy checkpoints/lstm_policy.bin \
  --dolphin ~/.config/Slippi\ Launcher/netplay \
  --iso ~/path/to/melee.iso \
  --character mewtwo \
  --stage final_destination \
  --on-game-end restart  # or 'stop' to exit after one game
```

**Sync Runner (For fast MLP models):**
```bash
source .venv/bin/activate

# Sync version - inference on every frame
# Best for fast models (<16ms inference)
mix run scripts/play_dolphin.exs \
  --policy checkpoints/mlp_policy.bin \
  --dolphin ~/.config/Slippi\ Launcher/netplay \
  --iso ~/path/to/melee.iso \
  --character mewtwo \
  --action-repeat 3  # Only compute new action every N frames
```

**Common Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--policy PATH` | required | Path to exported policy file |
| `--dolphin PATH` | required | Path to Slippi/Dolphin folder |
| `--iso PATH` | required | Path to Melee 1.02 ISO |
| `--port N` | 1 | Agent controller port |
| `--opponent-port N` | 2 | Human/opponent controller port |
| `--character NAME` | mewtwo | Agent character |
| `--stage NAME` | final_destination | Stage |
| `--frame-delay N` | 0 | Simulated online delay |
| `--deterministic` | false | Use argmax instead of sampling |

**Async-only Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--on-game-end MODE` | restart | `restart` = auto-start next game, `stop` = exit after one game |

**Sync-only Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--action-repeat N` | 1 | Cache action and reuse for N frames (reduces inference calls) |

#### Async Architecture

The async runner (`AsyncRunner`) uses Elixir's concurrency to decouple slow inference from fast frame reading:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   FrameLoop     │────▶│   SharedState    │◀────│   Inference     │
│   (fast, 60fps) │     │   (ETS table)    │     │   (slow, async) │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

- **FrameLoop**: Reads frames at 60fps, sends last known action
- **InferenceLoop**: Runs model async (~2/s for LSTM), updates action when ready
- **ETS table**: Lock-free shared state between processes

This allows the game to run smoothly even with 500ms LSTM inference times.

#### JIT Warmup

Both scripts include JIT warmup during Step 4, which pre-compiles the model before gameplay:
- Runs dummy inference with zero-filled tensors
- Avoids compilation stutter on first game frame
- Takes ~10s for MLP, ~60s for LSTM on first run

#### Wayland Notes (Hyprland, Sway, etc.)
Slippi Dolphin may need environment flags:
```bash
# Run Slippi Launcher with Wayland support
./Slippi-Launcher.AppImage --ozone-platform=wayland
```

### Technical Gotchas

#### 1. Polaris.Updates.apply_updates nil issue
`Polaris.Updates.apply_updates/2` is a `defn` with a default nil parameter (`state \\ nil`).
In Nx 0.10.0, calling it directly fails during lazy container traversal.

**Fix:** Wrap with `Nx.Defn.jit/1`:
```elixir
apply_updates_fn = Nx.Defn.jit(&Polaris.Updates.apply_updates/2)
new_params = apply_updates_fn.(params, updates)
```

#### 2. Axon training vs inference mode
- `mode: :train` returns `%{prediction: ..., state: ...}` for stateful layers (dropout)
- `mode: :inference` returns predictions directly
- For gradient computation, use `mode: :inference` to avoid pattern matching complexity

#### 3. Nx.to_number with {1} shaped tensors
`Nx.to_number/1` only works on scalar tensors (shape `{}`). When `Nx.slice` returns
a single element, it has shape `{1}` which must be squeezed first:
```elixir
tensor |> Nx.squeeze() |> Nx.to_number()
```

#### 4. Axon.ModelState deprecation
Pass full `%Axon.ModelState{}` to predict functions, not just the `.data` map:
```elixir
# Good
predict_fn.(model_state, input)

# Deprecated (triggers warning)
predict_fn.(model_state.data, input)
```

#### 5. JIT compilation time
First batch takes 2-5 minutes on CPU for large models (1991 input dims).
This is normal - subsequent batches are fast after compilation.

#### 6. EXLA Backend Configuration
EXLA provides ~2000x speedup over Nx.BinaryBackend. Configured in `config/config.exs`:
```elixir
config :nx, default_backend: EXLA.Backend
config :exla, default_client: :host  # CPU with XLA optimizations
```

For CUDA GPU support (if available):
```elixir
config :exla, :clients,
  cuda: [platform: :cuda, memory_fraction: 0.8],
  default: [platform: :host]
config :exla, default_client: :cuda
```

#### 7. EXLA/Defn.Expr tensor mismatch in closures
When using `Nx.Defn.value_and_grad` with closures that capture tensors, you get:
```
cannot invoke Nx function because it relies on two incompatible tensor implementations: EXLA.Backend and Nx.Defn.Expr
```

**Fix:** Copy ALL captured tensors before using them in gradient computation:
```elixir
# Copy batch data
states = Nx.backend_copy(states)
actions = Map.new(actions, fn {k, v} -> {k, Nx.backend_copy(v)} end)

# Also copy model parameters to avoid closure capture issues
model_state = deep_backend_copy(trainer.policy_params)
```

#### 8. Elixir struct pattern matching order
Structs are maps, so `is_map(%Nx.Tensor{})` returns `true`. When using guards with
struct patterns, the struct clause MUST come before the `is_map` guard:
```elixir
# CORRECT order - struct patterns first
defp deep_backend_copy(%Nx.Tensor{} = tensor), do: Nx.backend_copy(tensor)
defp deep_backend_copy(%Axon.ModelState{data: data} = state) do
  %{state | data: deep_backend_copy(data)}
end
defp deep_backend_copy(map) when is_map(map) and not is_struct(map) do
  Map.new(map, fn {k, v} -> {k, deep_backend_copy(v)} end)
end
defp deep_backend_copy(other), do: other
```

#### 9. EXLA tensor serialization in checkpoints
EXLA tensors contain device buffer references that are process/session-specific.
When serializing checkpoints with `:erlang.term_to_binary/1`, these buffers become
invalid after the training process ends.

**Fix:** Always convert tensors to `Nx.BinaryBackend` before saving:
```elixir
def save_checkpoint(trainer, path) do
  checkpoint = %{
    policy_params: to_binary_backend(trainer.policy_params),
    optimizer_state: to_binary_backend(trainer.optimizer_state),
    # ... other fields
  }
  File.write(path, :erlang.term_to_binary(checkpoint))
end

defp to_binary_backend(%Nx.Tensor{} = t), do: Nx.backend_copy(t, Nx.BinaryBackend)
defp to_binary_backend(%Axon.ModelState{data: data, state: state} = ms) do
  %{ms | data: to_binary_backend(data), state: to_binary_backend(state)}
end
defp to_binary_backend(map) when is_map(map) and not is_struct(map) do
  Map.new(map, fn {k, v} -> {k, to_binary_backend(v)} end)
end
defp to_binary_backend(other), do: other
```

**Symptoms of stale EXLA buffers:**
- `ArgumentError: unable to get buffer. It may belong to another node`
- `ArgumentError: decode failed, none of the variant types could be decoded`

#### 10. Policy architecture mismatch on load
When loading a policy, the model architecture must match exactly. If training used
`hidden_sizes: [64, 64]` but loading uses default `[512, 512]`, you get shape errors.

**Fix:** Always save and restore `hidden_sizes`, `embed_size`, `dropout` in policy config:
```elixir
# In export_policy
config = %{
  embed_size: trainer.config[:embed_size],
  hidden_sizes: trainer.config[:hidden_sizes],
  dropout: trainer.config[:dropout],
  # ... other config
}
```

#### 11. Dynamic sequence length in attention models causes infinite JIT compilation
When building attention models with `shape: {nil, nil, embed_size}` (dynamic batch AND
sequence length), XLA tries to compile kernels that handle arbitrary sequence lengths.
This can take 30+ minutes or never complete.

**Symptoms:**
- Training process stuck at "Step 4: Training..." with high CPU, no progress
- JIT compilation takes >10 minutes for attention models

**Fix:** Use concrete sequence length (`:seq_len` option) in attention model builders:
```elixir
# GOOD - concrete seq_len (fast JIT, ~2-3 minutes)
Attention.build_sliding_window(
  embed_size: 1991,
  window_size: 30,
  seq_len: 30  # Defaults to window_size, which is correct for training
)

# BAD - dynamic seq_len (can hang indefinitely)
Attention.build_sliding_window(
  embed_size: 1991,
  window_size: 30,
  seq_len: nil  # Forces dynamic shape - avoid for training
)
```

The attention modules now default `seq_len` to `window_size` for training efficiency.
For inference with variable-length sequences, pass `seq_len: nil` explicitly and
expect longer JIT compilation on first run.

#### 12. Pre-computed tensors in Axon.nx closures need BinaryBackend
When pre-computing tensors (like attention masks) that are captured in `Axon.nx`
closures, they must be in `Nx.BinaryBackend` to avoid EXLA/Defn.Expr mismatch.

**Symptom:**
```
cannot invoke Nx function because it relies on two incompatible tensor
implementations: EXLA.Backend and Nx.Defn.Expr
```

**Fix:** Convert pre-computed tensors to BinaryBackend before capture:
```elixir
# GOOD - explicit BinaryBackend
mask = window_mask(seq_len, window_size) |> Nx.backend_copy(Nx.BinaryBackend)
Axon.nx(input, fn tensor ->
  # mask is now safely inlined into the defn expression
  scaled_dot_product_attention(query, key, value, mask: mask)
end)

# BAD - EXLA tensor captured in closure
mask = window_mask(seq_len, window_size)  # Uses default EXLA backend
Axon.nx(input, fn tensor ->
  # CRASH: mask is EXLA but tensor is Defn.Expr during tracing
  scaled_dot_product_attention(query, key, value, mask: mask)
end)
```

### Performance Tips

#### CPU Training Optimization

**XLA Multi-threading (2-3x speedup):**
Auto-enabled in `train_from_replays.exs`. Can also be set manually:
```bash
XLA_FLAGS="--xla_cpu_multi_thread_eigen=true" mix run scripts/train_from_replays.exs
```

**Batch size tuning:**
- Larger batches (128, 256) reduce per-batch overhead
- Monitor RAM usage: `free -h` during training
- If swap usage increases, reduce batch size

**System tuning (optional):**
```bash
# Lower swappiness if swap thrashing occurs
sudo sysctl vm.swappiness=10

# Check process priority (nice value)
ps -o nice,pid,comm -p <PID>

# Increase priority (requires sudo for negative values)
sudo renice -n -5 -p <PID>
```

#### Training Time Expectations

| Dataset Size | Epochs | Estimated Time (CPU) |
|--------------|--------|---------------------|
| 1 file (~14K frames) | 1 | ~5-6 minutes |
| 10 files (~140K frames) | 1 | ~15-20 minutes |
| 100 files (~1.4M frames) | 10 | ~3-4 hours |

*First epoch includes ~5 min JIT compilation overhead*

#### Temporal Training Overhead

Temporal training (`--temporal`) has embedding preprocessing that scales with dataset size:
- Each sequence of `window_size` frames must be embedded once during Step 2
- Pre-computed embeddings use batch Nx operations for speed
- Progress is visible: "Embedding: X% (Y/Z)"

**Performance after optimization (batch Nx embedding):**
| Dataset | Sequences | Old Time | New Time |
|---------|-----------|----------|----------|
| 1 file (~14K frames, window=60) | ~13,818 | ~20 min | ~1 min |
| 10 files (~140K frames) | ~138K | ~3.5 hrs | ~10 min |

**Key optimizations applied:**
1. `embed_states_fast/3` - Batch embed all frames in a sequence at once
2. `Player.embed_batch/2` - Vectorized player embedding with Nana support
3. `batch_float_embed/2`, `batch_bool_embed/2` - Primitives for list→tensor conversion
4. `:array.get` instead of `Enum.at` - O(1) vs O(n) index lookup in batch creation
5. Cached `predict_fn` and `apply_updates_fn` in trainer - built once, reused every step

The `Data.precompute_embeddings/2` function does embedding once during dataset
creation, and `batched_sequences/2` uses O(1) array lookup for fast batch assembly.

#### GPU Support

Currently using CPU with XLA optimizations. For NVIDIA GPU:
```elixir
# config/config.exs
config :exla, :clients,
  cuda: [platform: :cuda, memory_fraction: 0.8],
  default: [platform: :host]
config :exla, default_client: :cuda
```

Intel Iris Xe / AMD integrated GPUs have limited EXLA support.

### Architecture Decisions

#### Embedding Structure
- **Player embedding:** 446 dimensions (position, action one-hot, character, stocks, etc.)
- **Game embedding:** ~1991 dimensions (2 players + stage + optional projectiles + name ID)
- **Controller embedding:** 8 buttons + 4 stick axes + 1 shoulder = 13 dimensions

#### Policy Network
- MLP backbone with configurable hidden sizes (default: [512, 512])
- 6 output heads: buttons (8 Bernoulli), main_x/y, c_x/y (17-way categorical each), shoulder (5-way)
- Autoregressive sampling during inference

#### Training
- AdamW optimizer with weight decay
- Imitation learning first, then PPO fine-tuning
- Gradient clipping for stability

### Running Training

```bash
# Single-frame imitation learning (baseline)
mix run scripts/train_from_replays.exs --epochs 10 --max-files 100

# Temporal training with sliding window attention
mix run scripts/train_from_replays.exs --temporal --backbone sliding_window \
  --window-size 60 --epochs 10

# Temporal training with hybrid LSTM + attention
mix run scripts/train_from_replays.exs --temporal --backbone hybrid

# Temporal training with LSTM only
mix run scripts/train_from_replays.exs --temporal --backbone lstm

# Temporal training with Mamba (fastest inference, recommended)
mix run scripts/train_from_replays.exs --temporal --backbone mamba

# With all options
mix run scripts/train_from_replays.exs \
  --replays /path/to/replays \
  --epochs 5 \
  --batch-size 64 \
  --player-port 1 \
  --temporal \
  --backbone sliding_window \
  --window-size 60 \
  --stride 1

# Faster training with truncated BPTT (2-3x speedup)
mix run scripts/train_from_replays.exs --temporal --backbone lstm \
  --window-size 60 --truncate-bptt 20
```

**Temporal training tradeoffs:**
- Slower per-epoch (sequences are larger than single frames)
- Better at learning temporal patterns (combos, reactions, habits)
- Recommended after establishing baseline with single-frame training

**Truncated BPTT (`--truncate-bptt N`):**
Limits how far gradients flow back through time during backpropagation.
- `nil` (default): Full BPTT - all 60 frames get gradients
- `20`: Only last 20 frames get gradients (early frames forward-pass only)
- `10`: Maximum speed, may lose long-range temporal patterns

| Setting | Speed | Accuracy | Use Case |
|---------|-------|----------|----------|
| Full BPTT | 1x | Best | Final training, complex combos |
| `--truncate-bptt 30` | ~1.5x | Good | Balanced training |
| `--truncate-bptt 20` | ~2x | Moderate | Fast iteration, prototyping |
| `--truncate-bptt 10` | ~3x | Lower | Quick experiments |

Recommended: Start with `window_size/3` (e.g., 20 for window=60)

### PPO Fine-tuning

```bash
# Test PPO loop with mock environment (no Dolphin needed)
mix run scripts/train_ppo.exs --mock \
  --pretrained checkpoints/imitation_latest_policy.bin \
  --timesteps 10000

# Full PPO training with Dolphin
mix run scripts/train_ppo.exs \
  --pretrained checkpoints/imitation_latest_policy.bin \
  --dolphin /path/to/slippi \
  --iso /path/to/melee.iso \
  --character mewtwo \
  --opponent cpu3 \
  --timesteps 100000
```

### Playing and Evaluation

```bash
# Evaluate model on replay frames
mix run scripts/eval_model.exs --policy checkpoints/imitation_latest_policy.bin

# Play against human in Dolphin (async - best for LSTM)
mix run scripts/play_dolphin_async.exs \
  --policy checkpoints/lstm_policy.bin \
  --dolphin /path/to/slippi \
  --iso /path/to/melee.iso \
  --on-game-end restart

# Play against human in Dolphin (sync - best for fast MLP)
mix run scripts/play_dolphin.exs \
  --policy checkpoints/mlp_policy.bin \
  --dolphin /path/to/slippi \
  --iso /path/to/melee.iso \
  --action-repeat 3

# Interactive analysis with Livebook
livebook server notebooks/evaluation_dashboard.livemd
```

#### Evaluation Dashboard Livebook

The `notebooks/evaluation_dashboard.livemd` provides interactive model analysis:

1. **Load Policy** - Point to any `.bin` policy file and see its config
2. **View Architecture** - See layer sizes and total parameter count
3. **Test Inference** - Run the model on sample game states
4. **Compare to Replays** - Measure prediction accuracy vs human play
5. **Visualize Actions** - Charts showing button press rates and stick positions

**To use:**
```bash
# Start Livebook server (runs on http://localhost:8080)
livebook server notebooks/evaluation_dashboard.livemd --port 8080
```

Then open browser to the URL shown and run cells interactively.

### Temporal Inference in Agents

Agents automatically detect temporal policies and handle frame buffering:

```elixir
# Load a temporal policy - agent auto-detects temporal config
{:ok, agent} = ExPhil.Agents.start_with_policy(:my_agent, "checkpoints/temporal_policy.bin")

# Check temporal config
config = ExPhil.Agents.Agent.get_config(agent)
# => %{temporal: true, backbone: :sliding_window, window_size: 60, buffer_size: 0, ...}

# Each get_action call adds to the frame buffer
{:ok, action} = ExPhil.Agents.get_action(:my_agent, game_state)

# Reset buffer when starting a new game
ExPhil.Agents.Agent.reset_buffer(agent)
```

**How temporal inference works:**
1. Each `get_action` call embeds the game state and adds it to a rolling buffer
2. If buffer has fewer than `window_size` frames, pad with first frame (warmup)
3. Build sequence tensor `[1, window_size, embed_size]`
4. Feed to temporal policy network
5. Sample action from output logits

**Policy export includes temporal metadata:**
```elixir
# When exporting, temporal config is preserved
Imitation.export_policy(trainer, "policy.bin")
# Saved config includes: temporal, backbone, window_size, num_heads, head_dim, etc.
```

### Test Commands

```bash
# Run all tests
mix test

# Run with coverage
mix test --cover

# Run specific test file
mix test test/exphil/training/imitation_test.exs

# Run slow tests (tagged @tag :slow)
mix test --include slow
```

## References

- [slippi-ai](https://github.com/vladfi1/slippi-ai) - Primary reference
- [libmelee](https://github.com/altf4/libmelee) - Game interface
- [Decision Transformer](https://arxiv.org/abs/2106.01345) - Sequence modeling for RL
- [DreamerV3](https://danijar.com/project/dreamerv3/) - World model RL
- [Nx](https://github.com/elixir-nx/nx) - Numerical Elixir
- [Axon](https://github.com/elixir-nx/axon) - Neural networks for Elixir
